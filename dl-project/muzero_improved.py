"""
MuZero 개선된 인터페이스
Train과 Inference를 명확히 분리하여 사용성 향상
"""
import torch
import numpy as np
from pathlib import Path
import datetime
import argparse
import json

# MuZero 관련 import
from models.MuZero.models import MuZeroFullyConnectedNetwork
from models.MuZero.config import MuZeroConfig
from models.MuZero.mcts import MCTS
from models.MuZero.action import Action
from models.MuZero.environment import DrivingEnvironment
from models.MuZero.observation_provider import GTObservationProvider, ROS2ObservationProvider, MockObservationProvider
from models.MuZero.self_play import SelfPlayManager
from models.MuZero.replay_buffer import ReplayBuffer
from models.MuZero.train import train_network


def get_network(config, device):
    """설정에 따라 적절한 MuZero 네트워크를 생성합니다."""
    if config.network == "fullyconnected":
        return MuZeroFullyConnectedNetwork(
            config.observation_shape,
            config.stacked_observations,
            len(config.action_space),
            config.encoding_size,
            config.fc_reward_layers,
            config.fc_value_layers,
            config.fc_policy_layers,
            config.fc_representation_layers,
            config.fc_dynamics_layers,
            config.support_size,
        ).to(device)
    else:
        raise NotImplementedError(
            '지원되는 네트워크 설정은 "fullyconnected"입니다.'
        )


def create_observation_provider(config, mode="train"):
    """모드에 따라 적절한 ObservationProvider 생성"""
    if mode == "train":
        if config.observation_provider_type == "gt":
            print(f"🎯 GT 데이터 사용: {config.gt_data_path}")
            return GTObservationProvider(config, config.gt_data_path)
        else:
            print("⚠️  Mock 데이터 사용 (학습용 실제 데이터가 준비되지 않음)")
            return MockObservationProvider(config)
    else:  # inference
        if config.observation_provider_type == "ros2":
            print("🤖 ROS2 실시간 데이터 사용")
            return ROS2ObservationProvider(config)
        else:
            print("⚠️  Mock 데이터 사용 (추론용 ROS2 연결이 준비되지 않음)")
            return MockObservationProvider(config)


def train_mode(config, checkpoint_path=None):
    """학습 모드 실행"""
    print("\n" + "="*60)
    print("🎓 MuZero 학습 모드")
    print("="*60)
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Device: {device}")
    
    # 결과 디렉토리 설정 (config.results_path 사용)
    print(f"📁 결과 저장 경로: {config.results_path}")
    
    # 네트워크 생성
    network = get_network(config, device)
    
    # 체크포인트 로드 (있는 경우)
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"📂 체크포인트 로드: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        network.load_state_dict(checkpoint['network_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"   → {start_epoch}번째 에폭부터 재개")
    else:
        start_epoch = 0
    
    # Observation Provider 생성 (학습용)
    observation_provider = create_observation_provider(config, mode="train")
    
    # 환경 생성
    environment = DrivingEnvironment(config, observation_provider)
    
    # Self-play 매니저 생성
    self_play_manager = SelfPlayManager(
        network=network,
        environment=environment,
        config=config,
        device=device
    )
    
    # Replay Buffer 생성
    replay_buffer = ReplayBuffer(capacity=config.replay_buffer_size)
    
    # 학습 루프
    print(f"\n🚀 학습 시작 (총 {config.num_training_epochs}회)")
    
    for epoch in range(start_epoch, config.num_training_epochs):
        print(f"\n📍 Epoch {epoch + 1}/{config.num_training_epochs}")
        
        # 1. Self-play로 데이터 수집
        print("  1️⃣ Self-play 데이터 수집...")
        games = self_play_manager.play_games(
            num_games=config.num_selfplay_games, 
            temperature=1.0
        )
        
        # 게임 통계
        game_lengths = [game.get_length() for game in games]
        total_rewards = [sum(game.rewards) for game in games]
        print(f"     - 평균 길이: {np.mean(game_lengths):.1f}")
        print(f"     - 평균 보상: {np.mean(total_rewards):.2f}")
        
        # Replay Buffer에 저장
        for game in games:
            replay_buffer.save_game(game)
        
        # 2. 네트워크 학습
        print("  2️⃣ 네트워크 학습...")
        train_network(network, replay_buffer, config, device)
        
        # 3. 주기적으로 체크포인트 저장
        if (epoch + 1) % config.checkpoint_interval == 0:
            checkpoint_dir = config.results_path / config.checkpoint_save_dir
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            save_path = checkpoint_dir / f"{config.checkpoint_epoch_prefix}{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'network_state_dict': network.state_dict(),
                'config': config.__dict__
            }, save_path)
            print(f"  💾 체크포인트 저장: {save_path}")
    
    print("\n✅ 학습 완료!")
    
    # 최종 모델 저장
    checkpoint_dir = config.results_path / config.checkpoint_save_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_path = checkpoint_dir / config.checkpoint_final_name
    torch.save({
        'network_state_dict': network.state_dict(),
        'config': config.__dict__
    }, final_path)
    print(f"💾 최종 모델 저장: {final_path}")
    
    # 설정 파일 저장
    config_path = config.results_path / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2, default=str)
    print(f"📄 설정 파일 저장: {config_path}")
    
    print(f"\n📊 TensorBoard 확인: tensorboard --logdir {config.results_path / 'logs'}")


def inference_mode(config, checkpoint_path, num_episodes=1, render=True):
    """추론 모드 실행"""
    print("\n" + "="*60)
    print("🚗 MuZero 추론 모드")
    print("="*60)
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Device: {device}")
    
    # 네트워크 생성 및 체크포인트 로드
    network = get_network(config, device)
    
    if not Path(checkpoint_path).exists():
        print(f"❌ 체크포인트를 찾을 수 없습니다: {checkpoint_path}")
        return
    
    print(f"📂 체크포인트 로드: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    network.load_state_dict(checkpoint['network_state_dict'])
    network.eval()  # 평가 모드로 전환
    
    # Observation Provider 생성 (추론용)
    observation_provider = create_observation_provider(config, mode="inference")
    
    # 환경 생성
    environment = DrivingEnvironment(config, observation_provider)
    
    # MCTS 생성 (추론용)
    mcts = MCTS(config)
    
    print(f"\n🎮 {num_episodes}개 에피소드 실행")
    
    for episode in range(num_episodes):
        print(f"\n📍 Episode {episode + 1}/{num_episodes}")
        
        # 환경 리셋
        observation = environment.reset()
        done = False
        total_reward = 0
        steps = 0
        actions_taken = []
        
        while not done and steps < config.max_game_length:
            # MCTS로 최적 액션 선택
            with torch.no_grad():
                # 네트워크 추론
                network_output = network.initial_inference(observation.unsqueeze(0))
                
                # MCTS 실행
                mcts_result = mcts.run(
                    observation,
                    add_exploration_noise=False  # 추론 시에는 탐색 노이즈 없음
                )
                
                # 최적 액션 선택 (가장 높은 확률)
                action = np.argmax(mcts_result['policy_targets'])
            
            # 액션 실행
            observation, reward, done, info = environment.step(action)
            
            # 통계 업데이트
            total_reward += reward
            steps += 1
            actions_taken.append(Action.from_index(action))
            
            # 렌더링 (옵션)
            if render:
                print(f"  Step {steps}: Action={Action.from_index(action)}, Reward={reward:.2f}")
        
        # 에피소드 결과
        print(f"\n  📊 Episode {episode + 1} 결과:")
        print(f"     - 총 스텝: {steps}")
        print(f"     - 총 보상: {total_reward:.2f}")
        print(f"     - 종료 이유: {info.get('done_reason', 'unknown')}")
        
        # 액션 통계
        action_counts = {}
        for action in actions_taken:
            key = str(action)
            action_counts[key] = action_counts.get(key, 0) + 1
        print(f"     - 액션 분포: {json.dumps(action_counts, indent=8)}")
    
    # 리소스 정리
    environment.close()
    print("\n✅ 추론 완료!")


def evaluate_mode(config, checkpoint_path, num_episodes=100):
    """평가 모드 - 모델 성능 평가"""
    print("\n" + "="*60)
    print("📊 MuZero 평가 모드")
    print("="*60)
    
    # 추론 모드와 유사하지만 통계 수집에 집중
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = get_network(config, device)
    
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)
    network.load_state_dict(checkpoint['network_state_dict'])
    network.eval()
    
    # 평가용 ObservationProvider 생성 (기본적으로 GT 사용하지만 설정 유지)
    if config.observation_provider_type != "gt":
        print(f"⚠️  평가 모드에서는 일반적으로 GT 데이터를 사용합니다. 현재: {config.observation_provider_type}")
    observation_provider = create_observation_provider(config, mode="train")  # GT나 Mock
    environment = DrivingEnvironment(config, observation_provider)
    
    # 평가 통계
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    print(f"\n🎮 {num_episodes}개 에피소드로 평가 시작...")
    
    for episode in range(num_episodes):
        observation = environment.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < config.max_game_length:
            with torch.no_grad():
                # 간단한 추론 (MCTS 없이)
                network_output = network.initial_inference(observation.unsqueeze(0))
                policy = torch.softmax(network_output.policy_logits, dim=1)
                action = torch.argmax(policy).item()
            
            observation, reward, done, info = environment.step(action)
            total_reward += reward
            steps += 1
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        if info.get('done_reason') == 'goal_reached':
            success_count += 1
        
        if (episode + 1) % 10 == 0:
            print(f"  진행: {episode + 1}/{num_episodes}")
    
    # 최종 통계
    print(f"\n📈 평가 결과:")
    print(f"   - 평균 보상: {np.mean(episode_rewards):.2f} (±{np.std(episode_rewards):.2f})")
    print(f"   - 평균 에피소드 길이: {np.mean(episode_lengths):.1f}")
    print(f"   - 성공률: {success_count/num_episodes*100:.1f}%")
    print(f"   - 최고 보상: {np.max(episode_rewards):.2f}")
    print(f"   - 최저 보상: {np.min(episode_rewards):.2f}")
    
    environment.close()


def find_latest_checkpoint(config):
    """가장 최근의 체크포인트 파일 경로를 찾아 반환"""
    results_dir = Path("models/MuZero/results")
    
    if not results_dir.exists():
        return None, "❌ results 디렉토리가 없습니다. 먼저 학습을 실행하세요."
    
    # 가장 최근 날짜 폴더 찾기
    dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    if not dirs:
        return None, "❌ 학습된 모델이 없습니다. 먼저 학습을 실행하세요."
    
    latest_dir = max(dirs, key=lambda x: x.stat().st_mtime)
    
    # config에서 체크포인트 설정 가져오기
    config_path = latest_dir / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
            checkpoint_save_dir = saved_config.get('checkpoint_save_dir', 'checkpoints')
            checkpoint_final_name = saved_config.get('checkpoint_final_name', 'muzero_final.pth')
    else:
        # config.json이 없으면 현재 config 사용
        checkpoint_save_dir = config.checkpoint_save_dir
        checkpoint_final_name = config.checkpoint_final_name
    
    checkpoint_path = latest_dir / checkpoint_save_dir / checkpoint_final_name
    
    if not checkpoint_path.exists():
        return None, f"❌ 체크포인트를 찾을 수 없습니다: {checkpoint_path}"
    
    print(f"✅ 체크포인트 발견: {checkpoint_path}")
    return checkpoint_path, None


def main():
    parser = argparse.ArgumentParser(description="MuZero 자율주행 AI")
    
    # 모드 선택
    parser.add_argument("mode", choices=["train", "inference", "evaluate"],
                       help="실행 모드 선택")
    
    # 공통 옵션
    parser.add_argument("--config", type=str, default=None,
                       help="설정 파일 경로 (JSON)")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="체크포인트 파일 경로 (기본값: 가장 최근 체크포인트)")
    
    # 학습 옵션
    parser.add_argument("--epochs", type=int, default=None,
                       help="학습 에폭 수 (기본값: config 설정 사용)")
    parser.add_argument("--resume", action="store_true",
                       help="이전 학습 재개")
    
    # 추론 옵션
    parser.add_argument("--episodes", type=int, default=1,
                       help="실행할 에피소드 수")
    parser.add_argument("--no-render", action="store_true",
                       help="렌더링 비활성화")
    
    # ObservationProvider 옵션
    parser.add_argument("--provider", choices=["mock", "gt", "ros2"],
                       default=None, help="Observation Provider 타입")
    parser.add_argument("--gt-data", type=str,
                       help="GT 데이터 파일 경로")
    
    args = parser.parse_args()
    
    # 설정 로드
    config = MuZeroConfig()
    
    # 설정 파일이 제공된 경우
    if args.config:
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
            for key, value in custom_config.items():
                setattr(config, key, value)
    
    # CLI 옵션으로 설정 덮어쓰기
    if args.provider:
        config.observation_provider_type = args.provider
    if args.gt_data:
        config.gt_data_path = Path(args.gt_data)
    if args.epochs is not None:
        config.num_training_epochs = args.epochs
    
    # 모드별 실행
    if args.mode == "train":
        checkpoint_path = args.checkpoint if args.resume else None
        train_mode(config, checkpoint_path)
    
    elif args.mode == "inference":
        # checkpoint 경로 확인
        checkpoint_path = args.checkpoint
        if checkpoint_path is None:
            checkpoint_path, error = find_latest_checkpoint(config)
            if error:
                print(error)
                return
        
        inference_mode(config, checkpoint_path, args.episodes, not args.no_render)
    
    elif args.mode == "evaluate":
        # checkpoint 경로 확인
        checkpoint_path = args.checkpoint
        if checkpoint_path is None:
            checkpoint_path, error = find_latest_checkpoint(config)
            if error:
                print(error)
                return
                
        evaluate_mode(config, checkpoint_path, args.episodes)


if __name__ == "__main__":
    main()