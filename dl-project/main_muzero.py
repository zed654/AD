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
from models.MuZero.train import train_network
from models.MuZero.utils import build_observation_tensor, calculate_game_statistics, print_game_statistics, print_episode_summary
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
            # 새로운 파라미터들
            use_coordconv=config.use_coordconv,
            use_goal_map=config.use_goal_map,
            use_film=config.use_film,
            alpha_init=config.alpha_init,
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
        





    




    for iteration in range(config.num_iterations):
        print(f"\n🔄 Iteration {iteration + 1}/{config.num_iterations}")
        enriched_episodes = [] # Replay Buffer에 추가할 에피소드 리스트 초기화
        # Episode에 MCTS 결과 추가
        if len(expert_episodes) > 0:
            print("\n🧠 Episode에 MCTS 적용 중...")
            import copy # enrich_episodes_with_cmts() 에 DeepCopy 위해 임시 추가
            enriched_episodes = self_play_manager.enrich_episodes_with_mcts(copy.deepcopy(expert_episodes))
        else:
            enriched_episodes = []
        
        # Enriched Episode를 Replay Buffer에 추가
        for episode in enriched_episodes:
            # ReplayBuffer.save_episode()이 내부적으로 FIFO 방식으로 오래된 episode 제거 처리
            replay_buffer.save_episode(episode)
        
        print(f"   전문가 데이터 + MCTS 결과 추가 완료: {len(enriched_episodes)}개 에피소드")
        
        # Replay Buffer 상세 정보 출력
        replay_buffer.print_buffer_summary()

        # 네트워크 학습
        print("  📖 네트워크 학습...")
        loss_dict = train_network(network, replay_buffer, config, device, logger)
        


    # 최종 모델 저장
    print("\n✅ 학습 완료!")
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
    # TODO: 학습 설계 완료 이후 구현하기


def evaluate_mode(config, checkpoint_path=None):
    """평가 모드 - GT 데이터에 대한 모델 성능 평가
    
    GT 데이터의 observation에 대해 MCTS를 실행하여 선택한 action과
    expert action을 비교하여 모델의 성능을 평가합니다.
    """
    # TODO: 학습 설계 완료 이후 구현하기

# def find_latest_checkpoint(config):
#     """가장 최근의 체크포인트 파일 경로를 찾아 반환"""
#     results_dir = Path("models/MuZero/results")
    
#     if not results_dir.exists():
#         return None, "❌ results 디렉토리가 없습니다. 먼저 학습을 실행하세요."
    
#     # 가장 최근 날짜 폴더 찾기
#     dirs = [d for d in results_dir.iterdir() if d.is_dir()]
#     if not dirs:
#         return None, "❌ 학습된 모델이 없습니다. 먼저 학습을 실행하세요."
    
#     latest_dir = max(dirs, key=lambda x: x.stat().st_mtime)
    
#     # config에서 체크포인트 설정 가져오기
#     config_path = latest_dir / "config.json"
#     if config_path.exists():
#         with open(config_path, 'r') as f:
#             saved_config = json.load(f)
#             checkpoint_save_dir = saved_config.get('checkpoint_save_dir', 'checkpoints')
#             checkpoint_final_name = saved_config.get('checkpoint_final_name', 'muzero_final.pth')
#     else:
#         # config.json이 없으면 현재 config 사용
#         checkpoint_save_dir = config.checkpoint_save_dir
#         checkpoint_final_name = config.checkpoint_final_name
    
#     checkpoint_path = latest_dir / checkpoint_save_dir / checkpoint_final_name
    
#     if not checkpoint_path.exists():
#         return None, f"❌ 체크포인트를 찾을 수 없습니다: {checkpoint_path}"
    
#     print(f"✅ 체크포인트 발견: {checkpoint_path}")
#     return checkpoint_path, None


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
    
    # 모드별 실행
    if args.mode == "train":
        # 체크포인트 경로 결정: config에 설정되어 있으면 자동 재개
        checkpoint_path = None
        if args.resume:
            # --resume 플래그가 있으면 우선적으로 명령줄 체크포인트 사용
            checkpoint_path = args.checkpoint or config.resume_checkpoint_path
        elif config.resume_checkpoint_path:
            # config에 체크포인트가 설정되어 있으면 자동으로 재개
            checkpoint_path = config.resume_checkpoint_path
            print(f"🔄 config.py에서 체크포인트 자동 재개: {checkpoint_path}")
        train_mode(config, checkpoint_path)
    
    elif args.mode == "inference":
        # checkpoint 경로 확인
        checkpoint_path = args.checkpoint
        # if checkpoint_path is None:
        #     checkpoint_path, error = find_latest_checkpoint(config)
        #     if error:
        #         print(error)
        #         return
        
        inference_mode(config, checkpoint_path, args.episodes, not args.no_render)
    
    elif args.mode == "evaluate":
        # checkpoint 경로 확인
        checkpoint_path = args.checkpoint
        # if checkpoint_path is None:
        #     checkpoint_path, error = find_latest_checkpoint(config)
        #     if error:
        #         print(error)
                # return
                
        # CLI 옵션으로 config 설정 덮어쓰기
        if args.episodes is not None:
            config.eval_num_episodes = args.episodes
        if args.no_mcts:
            config.eval_use_mcts = False
        if args.dataset:
            config.eval_dataset_type = args.dataset
        if args.save:
            config.eval_save_results = True
            
        evaluate_mode(config, checkpoint_path)


if __name__ == "__main__":
    main()