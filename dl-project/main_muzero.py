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
from models.MuZero.models import get_network
from models.MuZero.config import MuZeroConfig
# MCTS는 SelfPlayManager 내부에서 사용됨
from models.MuZero.action import Action
# from models.MuZero.environment import DrivingEnvironment
from models.MuZero.observation_provider import GTObservationProvider, ROS2ObservationProvider, MockObservationProvider
from models.MuZero.replay_buffer import ReplayBuffer
from models.MuZero.train import train_network
from models.MuZero.utils import build_observation_tensor, calculate_game_statistics, print_game_statistics
from models.MuZero.config_manager import load_config, save_config, print_config
from models.MuZero.logger import get_logger
from models.MuZero.expert_data_loader import ExpertDataLoader
from models.MuZero.self_play import enrich_episode_with_mcts_parallel_worker
from models.MuZero.utils import visualize_training_progress

import torch.multiprocessing as mp
import time
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn

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

    # Multiprocessing spawn method 설정 (CUDA 안전성)
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 이미 설정됨

    # 결과 디렉토리 설정 (config.results_path 사용)
    print(f"📁 결과 저장 경로: {config.results_path}")

    # 네트워크 생성
    network = get_network(config, device)
    
    # 체크포인트 로드 (있는 경우)
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"📂 체크포인트 로드: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        network.load_state_dict(checkpoint['network_state_dict'])
        start_epoch = checkpoint.get('iteration', 0)
        print(f"   → {start_epoch}번째 에폭부터 재개")
    else:
        start_epoch = 0
    
    # Observation Provider 생성 (학습용)
    # GT 인지, Mock 인지, ROS2 인지 설정에 따라 다름
    observation_provider = create_observation_provider(config, mode="train")
    # parsing_result = observation_provider.get_observation()  # 디버깅용 - 불필요

    # 환경 생성
    # environment = DrivingEnvironment(config, observation_provider)

    # Replay Buffer 생성
    replay_buffer = ReplayBuffer(capacity=config.replay_buffer_size)
    
    # 파싱한 Json 을 활용하여 Episode 구성 단계
    if config.observation_provider_type == "gt":
        print("\n📚 전문가 데이터 로드 중...")
        expert_loader = ExpertDataLoader(config, observation_provider)
        expert_episodes = expert_loader.create_episodes_from_json(
            num_episodes=config.gt_num_episodes,  # config에서 읽기
            episode_step_length=config.max_episode_step_length,  # config에서 읽기
            episode_overlap_steps=config.gt_episode_overlap  # config에서 읽기
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
        





    




    # Loss history 전역 관리 (연속된 그래프를 위해)
    loss_history = {
        'steps': [],
        'total_loss': [],
        'value_loss': [],
        'policy_loss': [],
        'reward_loss': [],
        'latest_loss': None
    }

    # TensorBoard Writer 전역 관리 (연속된 로그를 위해)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=config.results_path / "logs")
    # pool = mp.Pool(processes=config.selfplay_parallel_workers)
    for iteration in range(config.num_iterations):
        print(f"\n🔄 Iteration {iteration + 1}/{config.num_iterations}")
        enriched_episodes = [] # Replay Buffer에 추가할 에피소드 리스트 초기화

        # Episode에 MCTS 결과 추가
        if len(expert_episodes) > 0:
            # Progress Bar가 시작 정보를 표시하므로 중복 출력 제거
            start_time = time.time()

            # 1. Network state_dict 추출 (pickle 가능)
            network_state_dict = network.state_dict()

            # 3. Worker arguments 준비 (spawn 방식은 자동으로 deepcopy)
            args_list = [
                (idx, network_state_dict, config, device, episode)
                for idx, episode in enumerate(expert_episodes)
            ]

            # 4. 병렬 처리 (mp.Pool.imap 사용 - 메모리 효율적)
            # imap 사용 시, 순서 보장 + 하나씩 yield (메모리 절약)
            # 진행률 표시 -> 🧠 MCTS Self-play (Enrich Episodes) ━━━━━━━━━━━━━━━━━━ 45% (150/330) 0:00:18 성공: 148개
            enriched_episodes = []
            pool = mp.Pool(processes=config.selfplay_parallel_workers)
            try:
                with Progress(
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TextColumn("({task.completed}/{task.total})"),
                    TimeElapsedColumn(),
                    TextColumn("[green]성공: [bold yellow]{task.fields[success_count]}개"),
                ) as progress:
                    task = progress.add_task(
                        "[cyan]🧠 Self-play (Enrich Episodes) ",
                        total=len(args_list),
                        success_count=0
                    )

                    for i, ep in enumerate(pool.imap(
                        enrich_episode_with_mcts_parallel_worker,
                        args_list
                    )):
                        # None이 아닌 Episode만 수집
                        if ep is not None:
                            enriched_episodes.append(ep)

                        # Progress bar 업데이트
                        progress.update(task, advance=1, success_count=len(enriched_episodes))

            except KeyboardInterrupt:
                print("\n\n⚠️  사용자에 의해 중단되었습니다. Worker 프로세스를 정리하는 중...")
                pool.terminate()  # 즉시 종료
                pool.join()       # 종료 대기
                print("✅ 정리 완료!")
                raise  # 상위로 전파하여 프로그램 종료
            finally:
                # pool.terminate()  # 즉시 종료
                print("   모든 Worker 프로세스 정리 중...")
                pool.close()
                pool.join()
                # print("   worker 작업 완료. 모든 Worker 프로세스 유지")

            elapsed = time.time() - start_time

            # Iteration 종료 후 GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # GPU 연산 완료 대기
                time.sleep(0.1)
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                 # 메모리 사용량 출력
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1e9
                    print(f"  GPU {i} allocated: {allocated:.2f}GB")
            # # 4. 병렬 처리 (mp.Pool.map 사용)
            # # map 사용 시, worker 순서 보장됨 (worker 2가 먼저 끝나도, 결과는 1, 2, 3, ... 순으로 반환)
            # with mp.Pool(processes=config.selfplay_parallel_workers) as pool:
            #     enriched_episodes = pool.map(
            #         enrich_episode_with_mcts_parallel_worker,
            #         args_list
            #     )

            # # 5. None 제거 (실패한 Episode)
            # enriched_episodes = [ep for ep in enriched_episodes if ep is not None]

            print(f"Parallel processing completed in {elapsed:.2f} seconds")
            print(f"Successfully enriched {len(enriched_episodes)}/{len(args_list)} episodes")

        else:
            enriched_episodes = []
        

        if (iteration + 1) % config.visualization_interval_iteration == 0 and config.enable_training_visualization:
            try:
                visualize_training_progress(network, enriched_episodes, config, device, loss_history, iteration)
            except Exception as e:
                print(f"    ⚠️ 시각화 중 오류 발생: {e}")
                logger.info(f"Visualization error at Iteration {iteration+1}: {e}")

        # Enriched Episode를 Replay Buffer에 추가
        for episode in enriched_episodes:
            # ReplayBuffer.save_episode()이 내부적으로 FIFO 방식으로 오래된 episode 제거 처리
            # Replay_Buffer 저장 시, 내부적으로 GPU 텐서를 CPU 로 이동
            replay_buffer.save_episode(episode)

        print(f"   전문가 데이터 + MCTS 결과 추가 완료: {len(enriched_episodes)}개 에피소드")

        # Self-play 후 GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 🔍 Replay Buffer 텐서 분석 (첫 Iteration만)
        if iteration == 0 and len(replay_buffer.buffer) > 0:
            print(f"\n=== [Replay Buffer 분석] 첫 번째 Episode ===")
            first_ep = replay_buffer.buffer[0]
            for step_idx in range(min(3, len(first_ep.states))):
                state_dict = first_ep.states[step_idx]
                print(f"  Step {step_idx}:")
                if isinstance(state_dict, dict):
                    for key in ['obstacles', 'static_environment', 'ground_class_id']:
                        if key in state_dict:
                            value = state_dict[key]
                            if isinstance(value, torch.Tensor):
                                print(f"    ✓ {key}: Tensor(device={value.device}, dtype={value.dtype}, shape={value.shape})")
                            elif isinstance(value, np.ndarray):
                                print(f"    - {key}: ndarray(dtype={value.dtype}, shape={value.shape})")
                            else:
                                print(f"    ? {key}: {type(value).__name__}")

        # Replay Buffer 상세 정보 출력
        replay_buffer.print_buffer_summary(True if iteration == 0 else False)

        # GPU 메모리 사용량 출력
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(device) / 1024**3   # GB
            print(f"  🎮 GPU Memory 1: {allocated:.3f}GB allocated, {reserved:.3f}GB reserved")

        # 네트워크 학습
        print("  📖 네트워크 학습...")
        base_step = iteration * config.training_steps
        loss_dict = train_network(network, replay_buffer, config, device, logger, iteration_num=iteration+1, base_step=base_step, loss_history=loss_history, writer=writer)


        # GPU 메모리 사용량 출력 (동기화 후 측정)
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # 모든 GPU 작업 완료 대기
            torch.cuda.empty_cache()  # 캐시 한번 더 정리
            allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(device) / 1024**3   # GB
            print(f"  🎮 GPU Memory 2: {allocated:.3f}GB allocated, {reserved:.3f}GB reserved")

        


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
    
    # TensorBoard Writer 종료
    writer.close()

    # 설정 파일 저장
    config_path = config.results_path / "config.json"
    save_config(config, config_path)

    print(f"\n📊 TensorBoard 확인: tensorboard --logdir {config.results_path / 'logs'}")


def inference_mode(config, checkpoint_path=None):
    """추론 모드 실행"""
    # TODO: 학습 설계 완료 이후 구현하기


def evaluate_mode(config, checkpoint_path=None):
    """평가 모드 - GT 데이터에 대한 모델 성능 평가
    
    GT 데이터의 observation에 대해 MCTS를 실행하여 선택한 action과
    expert action을 비교하여 모델의 성능을 평가합니다.

    실제 추론 모드로 전환하여 성능 확인. (여기선 GT로, inference_mode 에서 ROS2 Topic 수신받아 수행)
    """
    # TODO: 학습 설계 완료 이후 구현하기
    """학습 모드 실행"""
    logger = get_logger("MuZero.Train", config.results_path / "logs")
    
    print("\n" + "="*60)
    print("🎓 MuZero 학습 모드")
    print("="*60)
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Device: {device}")
    logger.info(f"Device: {device}")

    # Multiprocessing spawn method 설정 (CUDA 안전성)
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 이미 설정됨

    # 결과 디렉토리 설정 (config.results_path 사용)
    print(f"📁 결과 저장 경로: {config.results_path}")

    # 네트워크 생성
    network = get_network(config, device)
    
    # 체크포인트 로드 (있는 경우)
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"📂 체크포인트 로드: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        network.load_state_dict(checkpoint['network_state_dict'])
        start_epoch = checkpoint.get('iteration', 0)
        print(f"   → {start_epoch}번째 에폭부터 재개")
    else:
        start_epoch = 0
    
    # Observation Provider 생성 (학습용)
    # GT 인지, Mock 인지, ROS2 인지 설정에 따라 다름
    observation_provider = create_observation_provider(config, mode="train")
    # parsing_result = observation_provider.get_observation()  # 디버깅용 - 불필요
    
    # 파싱한 Json 을 활용하여 Episode 구성 단계
    if config.observation_provider_type == "gt":
        print("\n📚 전문가 데이터 로드 중...")
        expert_loader = ExpertDataLoader(config, observation_provider)
        expert_episodes = expert_loader.create_episodes_from_json(
            num_episodes=config.gt_num_episodes,  # config에서 읽기
            episode_step_length=config.max_episode_step_length,  # config에서 읽기
            episode_overlap_steps=config.gt_episode_overlap  # config에서 읽기
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
        
    
    
    # 기존 코드 (무한 루프로 추론 시간 측정)
    episode_idx = 0
    while True:
        # import random
        # rand_num = random.randint(0, len(expert_episodes)-1)
        # episode = expert_episodes[rand_num]
    
        episode = expert_episodes[episode_idx]
        args = (0, network.state_dict(), config, device, episode)
        start_time = time.time()
        enriched_episode = enrich_episode_with_mcts_parallel_worker(args)
        elapsed = time.time() - start_time
        print(f"Episode {episode_idx} enriched in {elapsed:.2f} seconds")
    
        episode_idx = (episode_idx+1) % len(expert_episodes)
        time.sleep(0.1)

    # # TODO: 여기서 시각화...?
    # print(f"Parallel processing completed in {elapsed:.2f} seconds")



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
        # # checkpoint 경로 확인
        # checkpoint_path = args.checkpoint
        # # if checkpoint_path is None:
        # #     checkpoint_path, error = find_latest_checkpoint(config)
        # #     if error:
        # #         print(error)
        #         # return
                
        # # CLI 옵션으로 config 설정 덮어쓰기
        # if args.episodes is not None:
        #     config.eval_num_episodes = args.episodes
        # if args.no_mcts:
        #     config.eval_use_mcts = False
        # if args.dataset:
        #     config.eval_dataset_type = args.dataset
        # if args.save:
        #     config.eval_save_results = True

        # 체크포인트 경로 결정: config에 설정되어 있으면 자동 재개
        checkpoint_path = None
        if args.resume:
            # --resume 플래그가 있으면 우선적으로 명령줄 체크포인트 사용
            checkpoint_path = args.checkpoint or config.resume_checkpoint_path
        elif config.resume_checkpoint_path:
            # config에 체크포인트가 설정되어 있으면 자동으로 재개
            checkpoint_path = config.resume_checkpoint_path
            print(f"🔄 config.py에서 체크포인트 자동 재개: {checkpoint_path}")
        evaluate_mode(config, checkpoint_path)


if __name__ == "__main__":
    main()