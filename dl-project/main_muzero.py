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
# MCTS는 SelfPlayManager 내부에서 사용됨
from models.MuZero.action import Action
from models.MuZero.environment import DrivingEnvironment
from models.MuZero.observation_provider import GTObservationProvider, ROS2ObservationProvider, MockObservationProvider
from models.MuZero.self_play import SelfPlayManager
from models.MuZero.replay_buffer import ReplayBuffer
from models.MuZero.data_parser import build_observation_tensor
from models.MuZero.train import train_network
from models.MuZero.utils import calculate_game_statistics, print_game_statistics, print_episode_summary
from models.MuZero.config_manager import load_config, save_config, print_config
from models.MuZero.logger import get_logger
from models.MuZero.expert_data_loader import ExpertDataLoader


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
    logger = get_logger("MuZero.Train", config.results_path / "logs")
    
    print("\n" + "="*60)
    print("🎓 MuZero 학습 모드")
    print("="*60)
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Device: {device}")
    logger.info(f"Device: {device}")
    
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
    # GT 인지, Mock 인지, ROS2 인지 설정에 따라 다름
    observation_provider = create_observation_provider(config, mode="train")
    # parsing_result = observation_provider.get_observation()  # 디버깅용 - 불필요

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
    
    # 파싱한 Json 을 활용하여 Episode 구성 단계
    if config.observation_provider_type == "gt":
        print("\n📚 전문가 데이터 로드 중...")
        expert_loader = ExpertDataLoader(config, observation_provider)
        expert_episodes = expert_loader.create_episodes_from_json(
            num_episodes=config.gt_num_episodes,  # config에서 읽기
            episode_step_length=config.max_episode_step_length,  # config에서 읽기
            overlap=config.gt_episode_overlap  # config에서 읽기
        )
    elif config.observation_provider_type == "ros2":
        print("\n🤖 ROS2 실시간 데이터 로드 중...")
        expert_episodes = []
    elif config.observation_provider_type == "mock":
        print("\n⚠️  Mock 데이터 사용 (학습용 실제 데이터가 준비되지 않음)")
        expert_episodes = []
    else:
        raise ValueError(f"지원하지 않는 observation_provider_type: {config.observation_provider_type}")
    
    print(f"   Episode 생성 완료: {len(expert_episodes)}개")
        
    # Episode에 MCTS 결과 추가
    if len(expert_episodes) > 0:
        print("\n🧠 Episode에 MCTS 적용 중...")
        enriched_episodes = self_play_manager.enrich_episodes_with_mcts(expert_episodes)
    else:
        enriched_episodes = []
    
    # Enriched Episode를 Replay Buffer에 추가
    for episode in enriched_episodes:
        replay_buffer.save_episode(episode)
    
    print(f"   전문가 데이터 + MCTS 결과 추가 완료: {len(enriched_episodes)}개 에피소드")
    
    # 학습 루프
    print(f"\n🚀 학습 시작 (총 {config.num_training_epochs}회)")
    
    for epoch in range(start_epoch, config.num_training_epochs):
        print(f"\n📍 Epoch {epoch + 1}/{config.num_training_epochs}")
        
        # 네트워크 학습
        print("  📖 네트워크 학습...")
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
            logger.checkpoint_saved(save_path)
    
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
    save_config(config, config_path)
    
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
    
    # SelfPlayManager 활용 (MCTS 포함)
    self_play_manager = SelfPlayManager(
        network=network,
        environment=environment,
        config=config,
        device=device
    )
    
    print(f"\n🎮 {num_episodes}개 에피소드 실행")
    
    # 추론용 설정: temperature=0 (결정적 선택), 탐색 노이즈 없음
    games = self_play_manager.play_games(
        num_games=num_episodes,
        temperature=0.0  # 결정적 선택
    )
    
    # 결과 통계 출력
    for i, game in enumerate(games):
        print_episode_summary(i, game, show_actions=True)
        
        if render:
            print("\n   상세 스텝:")
            for step, (action_idx, reward) in enumerate(zip(game.actions, game.rewards)):
                print(f"     Step {step+1}: Action={Action.from_index(action_idx)}, Reward={reward:.2f}")
    
    # 리소스 정리
    environment.close()
    print("\n✅ 추론 완료!")


def evaluate_mode(config, checkpoint_path, num_episodes=100, use_mcts=True, dataset_type="validation", save_results=False):
    """평가 모드 - GT 데이터에 대한 모델 성능 평가
    
    GT 데이터의 observation에 대해 MCTS를 실행하여 선택한 action과
    expert action을 비교하여 모델의 성능을 평가합니다.
    
    Args:
        config: 설정 객체
        checkpoint_path: 체크포인트 경로
        num_episodes: 평가할 에피소드 수
        use_mcts: MCTS 사용 여부 (True: MCTS 사용, False: 네트워크 출력만 사용)
        dataset_type: 사용할 데이터셋 ("train", "validation", "test")
        save_results: 평가 결과를 파일로 저장할지 여부
    """
    print("\n" + "="*60)
    print("📊 MuZero 평가 모드 - GT 데이터 기반 평가")
    print("="*60)
    print(f"🎯 MCTS 사용: {'예' if use_mcts else '아니오'}")
    print(f"📁 데이터셋: {dataset_type}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 네트워크 로드
    network = get_network(config, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    network.load_state_dict(checkpoint['network_state_dict'])
    network.eval()
    print(f"✅ 체크포인트 로드 완료: {checkpoint_path}")
    
    # GT 데이터 로드 - 데이터셋 타입에 따라 다른 경로 사용
    config.observation_provider_type = "gt"  # 평가는 항상 GT 데이터로
    
    # 데이터셋별 경로 설정
    original_gt_dir = config.gt_jsonl_dir
    if dataset_type == "validation":
        # validation 데이터 경로 (예: gt_jsonl_dir + "_val")
        validation_dir = Path(original_gt_dir).parent / f"{Path(original_gt_dir).name}_val"
        if validation_dir.exists():
            config.gt_jsonl_dir = str(validation_dir)
            print(f"📂 Validation 데이터 사용: {validation_dir}")
        else:
            print(f"⚠️  Validation 데이터 경로가 없습니다: {validation_dir}")
            print(f"📂 Train 데이터로 대체: {original_gt_dir}")
    elif dataset_type == "test":
        # test 데이터 경로
        test_dir = Path(original_gt_dir).parent / f"{Path(original_gt_dir).name}_test"
        if test_dir.exists():
            config.gt_jsonl_dir = str(test_dir)
            print(f"📂 Test 데이터 사용: {test_dir}")
        else:
            print(f"⚠️  Test 데이터 경로가 없습니다: {test_dir}")
            print(f"📂 Train 데이터로 대체: {original_gt_dir}")
    else:  # train
        print(f"📂 Train 데이터 사용: {original_gt_dir}")
    
    observation_provider = create_observation_provider(config, mode="train")
    environment = DrivingEnvironment(config, observation_provider)
    
    # Expert 데이터 로더 생성
    expert_loader = ExpertDataLoader(observation_provider)
    
    # GT Episode 로드
    print(f"\n📁 GT 데이터 로드 중...")
    expert_episodes = expert_loader.load_episodes(max_episodes=num_episodes)
    actual_episodes = len(expert_episodes)
    print(f"✅ {actual_episodes}개 GT Episode 로드 완료")
    
    if use_mcts:
        # SelfPlayManager로 MCTS 평가
        self_play_manager = SelfPlayManager(
            network=network,
            environment=environment,
            config=config,
            device=device
        )
        
        print(f"\n🎮 GT 데이터에 MCTS 적용하여 평가 중...")
        # enrich_episodes_with_mcts를 사용하여 GT observation에 대해 MCTS 실행
        evaluated_episodes = self_play_manager.enrich_episodes_with_mcts(expert_episodes)
        
        # 평가 메트릭 계산
        total_steps = 0
        total_distance_error = 0.0
        episode_distance_errors = []
        episode_values = []
        
        for episode in evaluated_episodes:
            episode_error = 0.0
            episode_steps = 0
            
            for step_idx in range(len(episode)):
                total_steps += 1
                episode_steps += 1
                
                # 거리 오차 계산 (simulated vs expert position)
                if 'simulated_ego_pos' in episode.states[step_idx]:
                    sim_pos = episode.states[step_idx]['simulated_ego_pos']
                    expert_pos = episode.expert_ego_pos[step_idx]
                    if sim_pos and expert_pos:
                        distance = np.sqrt((sim_pos[0] - expert_pos[0])**2 + 
                                         (sim_pos[1] - expert_pos[1])**2)
                        total_distance_error += distance
                        episode_error += distance
            
            # Episode별 평균 거리 오차
            if episode_steps > 0:
                episode_distance_errors.append(episode_error / episode_steps)
            
            # Episode의 평균 value
            if len(episode.values) > 0:
                episode_values.append(np.mean(episode.values))
        
        # 최종 통계 출력
        print(f"\n📈 MCTS 평가 결과:")
        print(f"   - 평가한 Episode 수: {actual_episodes}개")
        print(f"   - 총 평가 스텝 수: {total_steps}")
        print(f"   - 평균 거리 오차: {total_distance_error/total_steps:.2f} 그리드 셀")
        print(f"   - Episode별 평균 거리 오차: {np.mean(episode_distance_errors):.2f} (±{np.std(episode_distance_errors):.2f})")
        print(f"   - 평균 Episode Value: {np.mean(episode_values):.3f} (±{np.std(episode_values):.3f})")
        
    else:
        # 네트워크 직접 추론 (MCTS 없이)
        print(f"\n🎮 네트워크 직접 추론으로 평가 중...")
        
        total_steps = 0
        total_distance_error = 0.0
        
        for episode in expert_episodes:
            for step_idx in range(len(episode)):
                state_dict = episode.states[step_idx]
                
                # 6채널 observation 생성
                observation = build_observation_tensor(state_dict, config.observation_shape, device)
                
                with torch.no_grad():
                    # 네트워크 추론
                    value, _, policy_logits, _ = network.initial_inference(observation.unsqueeze(0))
                    
                    # 정책에서 가장 높은 확률의 action 선택
                    action = torch.argmax(policy_logits).item()
                    
                    # value 추출 (categorical distribution에서)
                    support_size = (config.support_size * 2) + 1
                    support = torch.arange(-config.support_size, config.support_size + 1, device=device)
                    value_scalar = torch.sum(value[0] * support).item()
                    
                    # Action에 따른 simulated position 계산
                    # (실제 시뮬레이션 없이 단순 계산)
                    # 여기서는 평가 메트릭만 계산
                    
                total_steps += 1
        
        print(f"\n📈 네트워크 직접 추론 결과:")
        print(f"   - 평가한 Episode 수: {actual_episodes}개")
        print(f"   - 총 평가 스텝 수: {total_steps}")
    
    # 평가 결과 저장 (MCTS 사용 시에만)
    if save_results and use_mcts:
        import pickle
        from datetime import datetime
        
        # 저장 경로 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("evaluation_results")
        results_dir.mkdir(exist_ok=True)
        
        save_path = results_dir / f"eval_{dataset_type}_{timestamp}.pkl"
        
        # Episode 저장
        with open(save_path, 'wb') as f:
            pickle.dump(evaluated_episodes, f)
        
        print(f"\n💾 평가 결과 저장됨: {save_path}")
        print(f"   시각화: python visualize_evaluation.py --results {save_path}")
    
    environment.close()
    print("\n✅ 평가 완료!")


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
    
    # 평가 옵션
    parser.add_argument("--dataset", choices=["train", "validation", "test"],
                       default="validation", help="평가할 데이터셋 선택")
    parser.add_argument("--no-mcts", action="store_true",
                       help="MCTS 없이 네트워크 출력만으로 평가")
    parser.add_argument("--save", action="store_true",
                       help="평가 결과를 파일로 저장 (시각화용)")
    
    # ObservationProvider 옵션
    parser.add_argument("--provider", choices=["mock", "gt", "ros2"],
                       default=None, help="Observation Provider 타입")
    parser.add_argument("--gt-data", type=str,
                       help="GT 데이터 파일 경로")
    
    args = parser.parse_args()
    
    # 설정 로드
    if args.config:
        config = load_config(Path(args.config))
    else:
        config = MuZeroConfig()
    
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
                
        evaluate_mode(config, checkpoint_path, args.episodes, 
                     use_mcts=not args.no_mcts, dataset_type=args.dataset)


if __name__ == "__main__":
    main()