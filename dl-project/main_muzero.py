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
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_training_progress(network, replay_buffer, epoch, config, device, loss_history=None):
    """학습 중 모델의 예측 성능과 Loss를 시각화
    
    replay_buffer에서 Episode를 샘플링하여 현재 네트워크로 다시 추론
    loss_history가 있으면 Loss 그래프도 함께 표시
    """
    print(f"\n  📊 Epoch {epoch} 학습 진행 상황 시각화...")
    
    # 시각화할 Episode 샘플링 (최대 3개)
    episodes = replay_buffer.sample_episodes(min(3, len(replay_buffer)))
    
    if not episodes:
        print("    시각화할 Episode가 없습니다.")
        return
    
    # Self-play 매니저 생성 (MCTS enrichment를 위해)
    from models.MuZero.self_play import SelfPlayManager
    from models.MuZero.environment import DrivingEnvironment
    
    # Mock environment와 provider (시각화용이므로 실제 환경 불필요)
    class MockObservationProvider:
        def get_observation(self):
            return None
    
    env = DrivingEnvironment(config, MockObservationProvider())
    self_play_manager = SelfPlayManager(
        network=network,
        environment=env,
        config=config,
        device=device
    )
    
    # 각 Episode에 대해 시각화
    for ep_idx, episode in enumerate(episodes):
        if len(episode) < 10:  # 너무 짧은 Episode는 스킵
            continue
            
        # 시각화할 부분만 추출 (첫 10 스텝)
        visualization_steps = min(10, len(episode))
        
        # 추출한 Episode 부분을 새로운 Episode로 만듦
        from models.MuZero.replay_buffer import Episode
        partial_episode = Episode()
        for i in range(visualization_steps):
            partial_episode.states.append(episode.states[i])
            partial_episode.expert_ego_pos.append(episode.expert_ego_pos[i])
            partial_episode.actions.append(0)  # 임시 action
            partial_episode.rewards.append(0.0)  # 임시 reward
            partial_episode.values.append(0.0)  # 임시 value
            partial_episode.policies.append(None)  # 임시 policy
        
        # 현재 네트워크로 MCTS enrichment 수행
        enriched_episodes = self_play_manager.enrich_episodes_with_mcts([partial_episode])
        enriched_episode = enriched_episodes[0]
        
        # 시각화 - Loss 정보가 있으면 2개 subplot, 없으면 1개
        if loss_history and len(loss_history['epochs']) > 0:
            fig, (ax, ax_loss) = plt.subplots(1, 2, figsize=(20, 10))
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # 그리드 설정 (99x99)
        ax.set_xlim(0, 99)
        ax.set_ylim(0, 99)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # y축 반전 (row가 아래로 증가)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Epoch {epoch} - Episode {ep_idx+1} (First {visualization_steps} steps)')
        
        # Expert 경로 (파란색)
        expert_rows = [pos[0] for pos in enriched_episode.expert_ego_pos]
        expert_cols = [pos[1] for pos in enriched_episode.expert_ego_pos]
        ax.plot(expert_cols, expert_rows, 'b-', linewidth=2, label='Expert Path', alpha=0.7)
        ax.scatter(expert_cols, expert_rows, c='blue', s=50, zorder=5)
        
        # MCTS로 계산된 simulated 경로 (빨간색)
        # enrich_episodes_with_mcts에서 이미 simulated_ego_pos가 계산됨
        sim_rows = []
        sim_cols = []
        current_pos = (49, 49)  # 시작 위치
        
        for i in range(len(enriched_episode)):
            # enriched_episode의 action을 사용하여 위치 업데이트
            if i > 0:  # 첫 스텝은 항상 (49, 49)
                from models.MuZero.action import Action
                action = enriched_episode.actions[i-1]  # 이전 action으로 현재 위치 결정
                action_obj = Action.from_index(action)
                dx, dy = action_obj.get_movement_vector()
                
                # 180도 회전 좌표계에서 그리드 좌표 업데이트 (self_play.py와 동일):
                # dy > 0 (상단) → row 증가 (전방)
                # dx > 0 (우측) → col 감소 (좌측)
                next_row = current_pos[0] + dy  # 전방/후방
                next_col = current_pos[1] - dx  # 좌/우
                
                # 경계 체크
                if 0 <= next_row < 99 and 0 <= next_col < 99:
                    current_pos = (next_row, next_col)
            
            sim_rows.append(current_pos[0])
            sim_cols.append(current_pos[1])
        
        # 예측 경로 그리기
        ax.plot(sim_cols, sim_rows, 'r--', linewidth=2, label='Predicted Path (Current Network)', alpha=0.7)
        ax.scatter(sim_cols, sim_rows, c='red', s=50, zorder=5)
        
        # 시작점과 끝점 표시
        ax.scatter(expert_cols[0], expert_rows[0], c='green', s=100, marker='o', label='Start', zorder=6)
        if len(expert_cols) > 1:
            ax.scatter(expert_cols[-1], expert_rows[-1], c='orange', s=100, marker='s', label='End', zorder=6)
        
        # 범례
        ax.legend(loc='upper right')
        
        # Loss 정보 텍스트 박스 추가 (최신 Loss)
        if loss_history and 'latest_loss' in loss_history:
            latest = loss_history['latest_loss']
            loss_text = f"Latest Loss (Epoch {epoch}):\n"
            loss_text += f"Total: {latest['total_loss']:.4f}\n"
            loss_text += f"Value: {latest['value_loss']:.4f}\n"
            loss_text += f"Policy: {latest['policy_loss']:.4f}\n"
            loss_text += f"Reward: {latest['reward_loss']:.4f}"
            ax.text(0.02, 0.98, loss_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # Loss 그래프 그리기 (있는 경우)
        if loss_history and len(loss_history['epochs']) > 0 and 'ax_loss' in locals():
            epochs = loss_history['epochs']
            ax_loss.plot(epochs, loss_history['total_loss'], 'b-', label='Total Loss', linewidth=2)
            ax_loss.plot(epochs, loss_history['value_loss'], 'g--', label='Value Loss', alpha=0.7)
            ax_loss.plot(epochs, loss_history['policy_loss'], 'r--', label='Policy Loss', alpha=0.7)
            ax_loss.plot(epochs, loss_history['reward_loss'], 'y--', label='Reward Loss', alpha=0.7)
            
            ax_loss.set_xlabel('Epoch', fontsize=12)
            ax_loss.set_ylabel('Loss', fontsize=12)
            ax_loss.set_title(f'Training Loss History (up to Epoch {epoch})', fontsize=14)
            ax_loss.legend(loc='upper right')
            ax_loss.grid(True, alpha=0.3)
            
            # 현재 Epoch 표시
            if epoch in epochs:
                idx = epochs.index(epoch)
                ax_loss.scatter([epoch], [loss_history['total_loss'][idx]], 
                              c='red', s=100, zorder=5, marker='o')
                ax_loss.annotate(f'Current\n{loss_history["total_loss"][idx]:.4f}',
                                xy=(epoch, loss_history['total_loss'][idx]),
                                xytext=(10, 10), textcoords='offset points',
                                fontsize=9, ha='left',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # 저장
        save_dir = config.results_path / "visualizations"
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / f"epoch_{epoch}_episode_{ep_idx+1}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    💾 시각화 저장: {save_path}")


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
    
    # Loss history 저장용
    loss_history = {
        'epochs': [],
        'total_loss': [],
        'value_loss': [],
        'policy_loss': [],
        'reward_loss': [],
        'latest_loss': None
    }
    
    # 학습 루프
    print(f"\n🚀 학습 시작 (총 {config.num_training_epochs}회)")
    
    for epoch in range(start_epoch, config.num_training_epochs):
        print(f"\n📍 Epoch {epoch + 1}/{config.num_training_epochs}")
        
        # 네트워크 학습
        print("  📖 네트워크 학습...")
        loss_dict = train_network(network, replay_buffer, config, device)
        
        # Loss history 업데이트
        if loss_dict:
            loss_history['epochs'].append(epoch + 1)
            loss_history['total_loss'].append(loss_dict['total_loss'])
            loss_history['value_loss'].append(loss_dict['value_loss'])
            loss_history['policy_loss'].append(loss_dict['policy_loss'])
            loss_history['reward_loss'].append(loss_dict['reward_loss'])
            loss_history['latest_loss'] = loss_dict
            
            print(f"  📊 Loss: Total={loss_dict['total_loss']:.4f}, "
                  f"Value={loss_dict['value_loss']:.4f}, "
                  f"Policy={loss_dict['policy_loss']:.4f}, "
                  f"Reward={loss_dict['reward_loss']:.4f}")
        
        # 2.5 학습 진행 상황 시각화 (선택적)
        if (epoch + 1) % config.checkpoint_interval == 0:
            visualize_training_progress(network, replay_buffer, epoch + 1, config, device, loss_history)
        
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


def evaluate_mode(config, checkpoint_path=None):
    """평가 모드 - GT 데이터에 대한 모델 성능 평가
    
    GT 데이터의 observation에 대해 MCTS를 실행하여 선택한 action과
    expert action을 비교하여 모델의 성능을 평가합니다.
    """
    print("\n" + "="*60)
    print("📊 MuZero 평가 모드 - GT 데이터 기반 평가")
    print("="*60)
    print(f"🎯 MCTS 사용: {'예' if config.eval_use_mcts else '아니오'}")
    print(f"📁 데이터셋: {config.eval_dataset_type}")
    
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
    original_gt_path = config.gt_data_path
    if config.eval_dataset_type == "validation":
        # validation 데이터 경로 (예: gt_data_path에서 _val 추가)
        validation_path = Path(original_gt_path).parent / (Path(original_gt_path).stem + "_val" + Path(original_gt_path).suffix)
        if validation_path.exists():
            config.gt_data_path = validation_path
            print(f"📂 Validation 데이터 사용: {validation_path}")
        else:
            print(f"⚠️  Validation 데이터 경로가 없습니다: {validation_path}")
            print(f"📂 Train 데이터로 대체: {original_gt_path}")
    elif config.eval_dataset_type == "test":
        # test 데이터 경로
        test_path = Path(original_gt_path).parent / (Path(original_gt_path).stem + "_test" + Path(original_gt_path).suffix)
        if test_path.exists():
            config.gt_data_path = test_path
            print(f"📂 Test 데이터 사용: {test_path}")
        else:
            print(f"⚠️  Test 데이터 경로가 없습니다: {test_path}")
            print(f"📂 Train 데이터로 대체: {original_gt_path}")
    else:  # train
        print(f"📂 Train 데이터 사용: {original_gt_path}")
    
    observation_provider = create_observation_provider(config, mode="train")
    environment = DrivingEnvironment(config, observation_provider)
    
    # Expert 데이터 로더 생성
    expert_loader = ExpertDataLoader(config, observation_provider)
    
    # GT Episode 로드
    print(f"\n📁 GT 데이터 로드 중...")
    expert_episodes = expert_loader.create_episodes_from_json(
        num_episodes=config.eval_num_episodes,
        episode_step_length=config.max_episode_step_length,
        overlap=0  # 평가 시에는 겹침 없이
    )
    actual_episodes = len(expert_episodes)
    print(f"✅ {actual_episodes}개 GT Episode 로드 완료")
    
    if config.eval_use_mcts:
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
                observation = build_observation_tensor(state_dict, config.observation_shape, device, config.use_obstacle_channels)
                
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
    if config.eval_save_results and config.eval_use_mcts:
        import pickle
        from datetime import datetime
        
        # 저장 경로 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("evaluation_results")
        results_dir.mkdir(exist_ok=True)
        
        save_path = results_dir / f"eval_{config.eval_dataset_type}_{timestamp}.pkl"
        
        # Episode 저장
        with open(save_path, 'wb') as f:
            pickle.dump(evaluated_episodes, f)
        
        print(f"\n💾 평가 결과 저장됨: {save_path}")
        
        # 간단한 결과 시각화 (처음 몇 Episode)
        print("\n📊 평가 결과 예시 (처음 3개 Episode):")
        for i, episode in enumerate(evaluated_episodes[:3]):
            print(f"\nEpisode {i+1}:")
            print(f"  스텝 수: {len(episode)}")
            
            # GT vs 예측 경로 비교
            total_error = 0.0
            for step in range(len(episode)):
                if 'simulated_ego_pos' in episode.states[step]:
                    sim_pos = episode.states[step]['simulated_ego_pos']
                    expert_pos = episode.expert_ego_pos[step]
                    error = np.sqrt((sim_pos[0] - expert_pos[0])**2 + (sim_pos[1] - expert_pos[1])**2)
                    total_error += error
                    
                    if step < 5:  # 처음 5스텝만 출력
                        action = Action.from_index(episode.actions[step])
                        print(f"  Step {step}: GT {expert_pos} -> Pred {sim_pos} (오차: {error:.2f}, Action: {action})")
            
            avg_error = total_error / len(episode)
            print(f"  평균 오차: {avg_error:.2f} 그리드 셀")
    
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