import torch
import numpy as np
from pathlib import Path
import datetime
import argparse
from torch.utils.tensorboard import SummaryWriter

# MuZero 관련 import
from models.MuZero.models import MuZeroFullyConnectedNetwork
from models.MuZero.config import MuZeroConfig
from models.MuZero.mcts import MCTS
from models.MuZero.action import Action
from models.MuZero.environment import DrivingEnvironment
from models.MuZero.self_play import SelfPlayManager
from models.MuZero.replay_buffer import ReplayBuffer

def get_network(config):
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
        )
    else:
        raise NotImplementedError(
            '지원되는 네트워크 설정은 "fullyconnected"입니다.'
        )

def main(debug_mode=False):
    """MuZero Self-play 예제"""
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 설정 생성
    config = MuZeroConfig()
    
    # 네트워크 생성
    network = get_network(config).to(device)
    
    # 환경 생성 (config 사용)
    environment = DrivingEnvironment(config)
    
    # Self-play 매니저 생성
    self_play_manager = SelfPlayManager(
        network=network,
        environment=environment,
        config=config,
        device=device
    )
    
    if debug_mode:
        print("\n=== Single Game Self-play (with rendering) ===")
        game_history = self_play_manager.play_game(temperature=1.0, render=True)
        print(f"\n게임 요약:")
        print(f"- 총 스텝: {game_history.get_length()}")
        print(f"- 총 보상: {sum(game_history.rewards):.2f}")
        print(f"- 평균 가치: {np.mean(game_history.values):.2f}")
    
    # 여러 게임 실행 (메인 데이터 수집)
    print("\n=== Multiple Games Self-play ===")
    games = self_play_manager.play_games(num_games=config.num_selfplay_games, temperature=1.0)
    
    # 통계 출력
    game_lengths = [game.get_length() for game in games]
    total_rewards = [sum(game.rewards) for game in games]
    
    print(f"\n전체 게임 통계 ({config.num_selfplay_games}게임):")
    print(f"- 평균 길이: {np.mean(game_lengths):.1f} (±{np.std(game_lengths):.1f})")
    print(f"- 평균 보상: {np.mean(total_rewards):.2f} (±{np.std(total_rewards):.2f})")
    print(f"- 최대 보상: {np.max(total_rewards):.2f}")
    print(f"- 최소 보상: {np.min(total_rewards):.2f}")
    
    # Replay Buffer에 게임 데이터 저장
    print("\n=== Replay Buffer 데이터 수집 ===")
    replay_buffer = ReplayBuffer(capacity=100)
    
    # 디버그 모드에서 단일 게임도 저장
    if debug_mode:
        replay_buffer.save_game(game_history)
        print(f"단일 게임 저장 완료")
    
    # 여러 게임 저장
    for game in games:
        replay_buffer.save_game(game)
    print(f"{len(games)}개 게임 저장 완료")
    
    print(f"\nReplay Buffer 상태:")
    print(f"  - 저장된 에피소드: {replay_buffer.size()}")
    print(f"  - 총 누적 에피소드: {replay_buffer.total_episodes()}")
    
    # Replay Buffer에서 배치 샘플링 테스트
    print("\n=== 배치 샘플링 테스트 ===")
    batch = replay_buffer.get_batch(batch_size=2)
    if batch:
        print(f"샘플링된 배치 크기: {len(batch)}")
        for i, episode in enumerate(batch):
            print(f"  - Episode {i+1}: {len(episode)} 스텝, 총 보상: {sum(episode.rewards):.2f}")
            
            # 첫 번째 스텝의 상세 정보 출력
            if len(episode) > 0:
                print(f"    첫 번째 스텝:")
                print(f"      - Action: {Action.from_index(episode.actions[0])}")
                print(f"      - Reward: {episode.rewards[0]:.2f}")
                print(f"      - Value: {episode.values[0]:.2f}")
                print(f"      - Policy shape: {episode.policies[0].shape}")
    else:
        print("배치 샘플링 실패 (데이터 부족)")

    # 모델 학습 수행
    print("\n=== MuZero 모델 학습 ===")
    train_network(network, replay_buffer, config, device)


def support_to_scalar(logits, support_size):
    """
    Support distribution을 스칼라 값으로 변환
    """
    # Logits를 확률로 변환
    probabilities = torch.softmax(logits, dim=1)
    # Support 범위 생성
    support = torch.arange(-support_size, support_size + 1, 
                          dtype=torch.float32, device=logits.device)
    support = support.expand(logits.shape[0], -1)
    # 기대값 계산
    return torch.sum(support * probabilities, dim=1)


def scalar_to_support(scalars, support_size):
    """
    스칼라 값을 support distribution으로 변환 (cross_entropy용 인덱스 반환)
    """
    batch_size = scalars.shape[0]
    
    # Clamp scalars to the valid support range
    scalars = torch.clamp(scalars, -support_size, support_size)
    
    # Transform scalar to support index
    # This is a simplified version - MuZero paper uses a more sophisticated method
    floor_idx = torch.floor(scalars + support_size).long()
    floor_idx = torch.clamp(floor_idx, 0, 2 * support_size)
    
    return floor_idx


def train_network(network, replay_buffer, config, device):
    """MuZero 네트워크 학습
    
    Args:
        network: MuZero 신경망
        replay_buffer: 경험 재생 버퍼
        config: MuZero 설정
        device: 학습 디바이스 (cuda/cpu)
    
    학습 과정:
    1. Replay Buffer에서 에피소드 배치 샘플링
    2. 각 에피소드에서 랜덤 위치 선택하고 K-step unroll
    3. Value, Reward, Policy loss 계산
    4. 역전파 및 가중치 업데이트
    """
    
    # Optimizer 설정
    if config.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=config.lr_init,
            weight_decay=config.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            network.parameters(),
            lr=config.lr_init,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
    
    # 학습률 스케줄러
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.lr_decay_steps,
        gamma=config.lr_decay_rate
    )
    
    # 학습 설정
    K = 5  # Unroll 스텝 수
    training_steps = min(config.training_steps, 100)  # 데모를 위해 100스텝으로 제한
    batch_size = min(config.batch_size, replay_buffer.size())  # 배치 크기 조정
    
    print(f"학습 설정:")
    print(f"  - Optimizer: {config.optimizer}")
    print(f"  - 학습률: {config.lr_init}")
    print(f"  - 배치 크기: {batch_size}")
    print(f"  - 학습 스텝: {training_steps}")
    print(f"  - Unroll 스텝: {K}")
    
    # TensorBoard 설정
    writer = SummaryWriter(log_dir=config.results_path / "logs")
    
    # 학습 루프
    for step in range(training_steps):
        # 배치 샘플링
        batch = replay_buffer.get_batch(batch_size)
        if batch is None:
            print(f"Step {step}: 배치 샘플링 실패 (데이터 부족)")
            continue
        
        # 손실 누적 변수
        total_loss = 0
        value_loss = 0
        reward_loss = 0
        policy_loss = 0
        
        # 배치의 각 에피소드에 대해
        for episode in batch:
            # 에피소드가 너무 짧으면 스킵
            if len(episode) < 2:
                continue
                
            # 랜덤 시작 위치 선택 (마지막 K 스텝 전까지)
            max_start = max(0, len(episode) - K)
            start_idx = np.random.randint(0, max_start + 1)
            
            # 초기 관측값
            observation = episode.states[start_idx].unsqueeze(0).to(device)
            
            # 초기 추론
            value, _, policy_logits, hidden_state = network.initial_inference(observation)
            
            # 첫 스텝의 손실 계산
            # Value loss
            target_value_scalar = calculate_target_value(episode, start_idx, config.discount)
            target_value_idx = scalar_to_support(
                torch.tensor([target_value_scalar], device=device),
                config.support_size
            )
            value_loss += torch.nn.functional.cross_entropy(value, target_value_idx)
            
            # Policy loss
            target_policy = torch.tensor(episode.policies[start_idx], device=device).unsqueeze(0)
            # KL divergence loss (target_policy는 확률 분포)
            log_policy = torch.nn.functional.log_softmax(policy_logits, dim=1)
            policy_loss += torch.nn.functional.kl_div(log_policy, target_policy, reduction='batchmean')
            
            # K 스텝 unroll
            for k in range(1, K + 1):
                if start_idx + k >= len(episode):
                    break
                    
                # 액션
                action = torch.tensor([episode.actions[start_idx + k - 1]], device=device)
                
                # Recurrent inference
                value, reward, policy_logits, hidden_state = network.recurrent_inference(
                    hidden_state, action
                )
                
                # Reward loss
                target_reward_scalar = episode.rewards[start_idx + k - 1]
                target_reward_idx = scalar_to_support(
                    torch.tensor([target_reward_scalar], device=device),
                    config.support_size
                )
                reward_loss += torch.nn.functional.cross_entropy(reward, target_reward_idx)
                
                # Value loss
                target_value_scalar = calculate_target_value(episode, start_idx + k, config.discount)
                target_value_idx = scalar_to_support(
                    torch.tensor([target_value_scalar], device=device),
                    config.support_size
                )
                value_loss += torch.nn.functional.cross_entropy(value, target_value_idx)
                
                # Policy loss
                if start_idx + k < len(episode):
                    target_policy = torch.tensor(episode.policies[start_idx + k], device=device).unsqueeze(0)
                    # KL divergence loss (target_policy는 확률 분포)
                    log_policy = torch.nn.functional.log_softmax(policy_logits, dim=1)
                    policy_loss += torch.nn.functional.kl_div(log_policy, target_policy, reduction='batchmean')
        
        # 평균 손실 계산
        num_samples = len(batch)
        value_loss = value_loss / (num_samples * (K + 1))
        reward_loss = reward_loss / (num_samples * K)
        policy_loss = policy_loss / (num_samples * (K + 1))
        
        # 총 손실
        total_loss = value_loss + reward_loss + policy_loss
        
        # 역전파
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=5.0)
        
        # 가중치 업데이트
        optimizer.step()
        
        # 로깅
        if step % 10 == 0:
            print(f"Step {step}: Loss={total_loss:.4f} "
                  f"(V={value_loss:.4f}, R={reward_loss:.4f}, P={policy_loss:.4f})")
            
            # TensorBoard 로깅
            writer.add_scalar('Loss/Total', total_loss.item(), step)
            writer.add_scalar('Loss/Value', value_loss.item(), step)
            writer.add_scalar('Loss/Reward', reward_loss.item(), step)
            writer.add_scalar('Loss/Policy', policy_loss.item(), step)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], step)
        
        # 학습률 업데이트
        if (step + 1) % config.lr_decay_steps == 0:
            lr_scheduler.step()
    
    writer.close()
    print("\n학습 완료!")
    
    # 모델 저장
    if config.save_model:
        model_path = config.results_path / "model.pth"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(network.state_dict(), model_path)
        print(f"모델 저장: {model_path}")


def calculate_target_value(episode, start_idx, discount):
    """n-step return 계산"""
    if start_idx >= len(episode):
        return 0.0
        
    # Bootstrap from value if available
    if start_idx < len(episode) and episode.values[start_idx] is not None:
        target_value = episode.values[start_idx]
    else:
        # Calculate discounted return
        target_value = 0.0
        for i in range(start_idx, len(episode)):
            target_value += (discount ** (i - start_idx)) * episode.rewards[i]
    
    return target_value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MuZero Self-play")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with visualization")
    args = parser.parse_args()
    
    main(debug_mode=args.debug)
