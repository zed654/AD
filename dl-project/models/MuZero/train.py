"""
MuZero 학습 모듈
신경망 학습 및 관련 유틸리티 함수들
"""
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from .utils import build_observation_tensor
from .models import MuZeroFullyConnectedNetwork
from .self_play import enrich_episode_with_mcts_parallel_worker

# 학습 중 PNG 시각화 관련
import matplotlib
matplotlib.use('Agg')  # non-interactive 백엔드 사용 (Figure 창이 뜨지 않음)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

# 학습 중 PNG 시각화 관련 2 (rosetta.py에서 color/grid 함수 import)
import sys
sys.path.append('data/db3_parser')
from rosetta import (get_grid_value_info, get_ldm_segmentation_name, get_ldm_segmentation_color, 
                     GRID_VALUE_EMPTY, GRID_VALUE_PEDESTRIAN, GRID_VALUE_VEHICLE, GRID_VALUE_OTHER_OBJECT, 
                     GRID_VALUE_CLIFF, GRID_VALUE_STATIC_OBSTACLE, GRID_VALUE_MISSING_POINTS)

# 학습 중 PNG 시각화 관련 3
def get_grid_color(value: float, channel: int = None) -> str:
    """Grid 값과 채널에 따른 색상을 반환합니다.
    
    Args:
        value: Grid 값
        channel: 채널 번호 (0: obstacles, 1: static_environment, 2: ground_class_id)
    """
    # 부동소수점 비교를 위한 작은 오차 허용
    epsilon = 0.01
    
    # 빈 공간
    if abs(value - GRID_VALUE_EMPTY) < epsilon:  # 0.0
        return 'white'
    
    # Channel 0: 동적 객체
    if channel == 0:
        if abs(value - GRID_VALUE_VEHICLE) < epsilon:  # 0.6: 차량
            return 'red'
        elif abs(value - GRID_VALUE_OTHER_OBJECT) < epsilon:  # 0.4: 기타 객체
            return 'orange'
        elif abs(value - GRID_VALUE_PEDESTRIAN) < epsilon:  # 0.3: 보행자
            return 'magenta'
    
    # Channel 1: 정적 환경
    elif channel == 1:
        if abs(value - GRID_VALUE_MISSING_POINTS) < epsilon:  # 0.9: Missing points
            return 'purple'
        elif abs(value - GRID_VALUE_STATIC_OBSTACLE) < epsilon:  # 0.6: Static obstacles
            return 'yellow'
        elif abs(value - GRID_VALUE_CLIFF) < epsilon:  # 0.3: Cliff
            return 'cyan'
    
    # Channel 2: 지면 semantic class ID
    elif channel == 2:
        class_id = int(value)
        if class_id == 0:  # Empty/Background
            return 'lightgray'
        else:
            # LDM segmentation 색상 가져오기
            color_hex = get_ldm_segmentation_color(class_id)
            if color_hex:
                return color_hex
            else:
                return 'gray'
    
    # 기본값
    return 'gray'



def h_transform(x):
    """
    MuZero 논문의 transform 함수
    h(x) = sign(x) * (sqrt(|x| + 1) - 1) + ε*x
    큰 값을 압축하여 학습 안정성 향상
    """
    epsilon = 0.001
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + epsilon * x

def h_inverse(x):
    """
    Transform의 역함수
    """
    epsilon = 0.001
    return torch.sign(x) * (
        ((torch.sqrt(1 + 4 * epsilon * (torch.abs(x) + 1 + epsilon)) - 1) / (2 * epsilon)) ** 2 - 1
    )

def support_to_scalar(logits, config):
    """
    Support distribution을 스칼라 값으로 변환 (with transform)
    config의 categorical_distribution_min/max 사용
    """
    # Logits를 확률로 변환
    probabilities = torch.softmax(logits, dim=1)
    # Support 범위 생성 (config 기반)
    if hasattr(config, 'categorical_distribution_min'):
        num_categorical_distribution_atoms = config.num_categorical_distribution_atoms if hasattr(config, 'num_categorical_distribution_atoms') else logits.shape[1]
        support = torch.linspace(config.categorical_distribution_min, config.categorical_distribution_max, num_categorical_distribution_atoms,
                                dtype=torch.float32, device=logits.device)
    else:
        # Fallback
        support_size = (logits.shape[1] - 1) // 2
        support = torch.arange(-support_size, support_size + 1,
                              dtype=torch.float32, device=logits.device)
    support = support.expand(logits.shape[0], -1)
    # 기대값 계산 (transformed space에서)
    transformed_value = torch.sum(support * probabilities, dim=1)
    # 역변환하여 실제 값 반환
    return h_inverse(transformed_value)


def scalar_to_support(scalars, config):
    """
    스칼라 값을 support distribution으로 변환 (Two-hot encoding)
    config의 categorical_distribution_min/max 사용
    """
    batch_size = scalars.shape[0]
    device = scalars.device

    # 1. Transform 적용
    transformed = h_transform(scalars)

    # 2. 범위 제한 (transformed space에서)
    categorical_distribution_min = config.categorical_distribution_min if hasattr(config, 'categorical_distribution_min') else -config.support_size
    categorical_distribution_max = config.categorical_distribution_max if hasattr(config, 'categorical_distribution_max') else config.support_size
    transformed = torch.clamp(transformed, categorical_distribution_min, categorical_distribution_max)

    # 3. Support 범위와 인덱스 계산
    num_categorical_distribution_atoms = config.num_categorical_distribution_atoms if hasattr(config, 'num_categorical_distribution_atoms') else (2 * config.support_size + 1)

    # 연속값을 이산 인덱스로 매핑
    # [categorical_distribution_min, categorical_distribution_max] → [0, num_categorical_distribution_atoms-1]
    normalized = (transformed - categorical_distribution_min) * (num_categorical_distribution_atoms - 1) / (categorical_distribution_max - categorical_distribution_min)

    # 4. Two-hot encoding (가장 가까운 두 원자에 확률 분배)
    lower = torch.floor(normalized).long()
    upper = lower + 1

    # 경계 처리
    lower = torch.clamp(lower, 0, num_categorical_distribution_atoms - 1)
    upper = torch.clamp(upper, 0, num_categorical_distribution_atoms - 1)

    # 가중치 계산 (float32로 명시적 변환)
    upper_weight = (normalized - lower.float()).float()
    lower_weight = (1 - upper_weight).float()

    # 5. Categorical distribution 생성 (dtype 명시)
    targets = torch.zeros(batch_size, num_categorical_distribution_atoms, device=device, dtype=torch.float32)

    # 배치 인덱스
    batch_idx = torch.arange(batch_size, device=device)

    # Two-hot assignment (squeeze 제거 - 이미 1D)
    targets[batch_idx, lower.squeeze()] = lower_weight
    targets[batch_idx, upper.squeeze()] = upper_weight

    return targets

def train_network(network, replay_buffer, config, device, logger, iteration_num=1, base_step=0, loss_history=None, writer=None):
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
    

    # Loss history 초기화 (전달되지 않은 경우만)
    if loss_history is None:
        loss_history = {
            'steps': [],
            'total_loss': [],
            'value_loss': [],
            'policy_loss': [],
            'reward_loss': [],
            'latest_loss': None
        }

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
    
    # Mixed Precision Training (AMP) 설정
    # 이론적 배경:
    # - 혼합 정밀도 학습: FP32(32비트)와 FP16(16비트) 부동소수점을 혼합 사용
    # - GPU Tensor Cores 활용: V100/A100 등에서 FP16 행렬 곱셈이 2-4배 빠름
    # - 메모리 절약: FP16 사용으로 GPU 메모리 사용량 약 50% 감소
    # - 수치적 안정성: GradScaler로 gradient underflow 방지 (loss scaling)
    use_amp = getattr(config, 'use_mixed_precision', False) and torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None  # FP16 gradient scaling으로 underflow 방지

    # Support 벡터 미리 생성 (재사용을 위해)
    support_vector = None
    if hasattr(config, 'categorical_distribution_min'):
        support_vector = torch.linspace(
            config.categorical_distribution_min,
            config.categorical_distribution_max,
            21,
            dtype=torch.float32,
            device=device
        )
    else:
        support_vector = torch.arange(-10, 11, dtype=torch.float32, device=device)
    
    # Gradient Accumulation 설정
    # 이론적 배경:
    # - 효과적 배치 크기 증가: 메모리 제한 하에서 큰 배치 효과 시뮬레이션
    # - 안정적인 학습: 더 큰 유효 배치로 gradient 분산 감소
    # - 메모리 효율성: 실제 메모리는 작은 배치만큼만 사용하면서 큰 배치 효과
    gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
    
    # 학습 설정
    K = config.unroll_steps  # Unroll 스텝 수 (config에서 가져옴)
    training_steps = config.training_steps  # K-Step Unrolling 하면서 Weight 업데이트하는 횟수
    
    # Replay buffer가 비어있으면 학습 스킵
    if replay_buffer.size() == 0:
        print("경고: Replay buffer가 비어있어 학습을 건너뜁니다.")
        return
    
    batch_size = min(config.batch_size, replay_buffer.size())  # 배치 크기 조정
    
    print(f"학습 설정:")
    print(f"  - Optimizer: {config.optimizer}")
    print(f"  - 학습률: {config.lr_init}")
    print(f"  - 배치 크기: {batch_size}")
    print(f"  - 학습 스텝: {training_steps}")
    print(f"  - Unroll 스텝: {K}")
    print(f"  - Mixed Precision: {'활성화' if use_amp else '비활성화'}")
    print(f"  - Gradient Accumulation: {gradient_accumulation_steps}")
    
    # TensorBoard 설정 (전달되지 않은 경우만 생성)
    if writer is None:
        writer = SummaryWriter(log_dir=config.results_path / "logs")
        close_writer = True  # 이 함수에서 생성한 경우 닫아야 함
    else:
        close_writer = False  # 외부에서 전달된 경우 닫지 않음
    
    # 학습 루프
    print("\n=== MuZero 네트워크 학습 시작 ===")
    for step in range(training_steps):
        # 배치 샘플링 ( n Batch = Episode n 개)
        batch = replay_buffer.get_batch(batch_size) # Replay Buffer 에서 batch_size 만큼 랜덤하게 Episode 를 뽑아냄
        if batch is None:
            print(f"Step {step}: 배치 샘플링 실패 (데이터 부족)")
            continue

        # Episode 통계 수집 (TensorBoard용) - no_grad로 메모리 절약
        with torch.no_grad():
            episode_rewards = []
            episode_values = []
            episode_target_values = []

        def to_python_float(value):
            """모든 타입을 Python float로 안전하게 변환

            Categorical distribution (21-element Tensor)을 scalar로 변환:
            - Softmax로 확률 분포 계산
            - Support와 내적하여 기댓값 계산
            """
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()

                # 1-element tensor: 바로 scalar 반환
                if value.numel() == 1:
                    return float(value.item())

                # (1, 21) 또는 (21,) shape의 categorical distribution
                if value.numel() == 21:
                    # Flatten to 1D
                    value_1d = value.flatten()

                    # Support 생성 (config의 설정 사용)
                    if hasattr(config, 'categorical_distribution_min'):
                        support = torch.linspace(
                            config.categorical_distribution_min,
                            config.categorical_distribution_max,
                            21,
                            dtype=torch.float32
                        )
                    else:
                        support = torch.arange(-10, 11, dtype=torch.float32)

                    # Logits → Probabilities → Expected Value
                    probs = torch.softmax(value_1d, dim=0)
                    scalar_value = torch.sum(probs * support).item()
                    return float(scalar_value)
                else:
                    # 그 외 다차원 Tensor는 평균 사용
                    print(f"Warning: Unexpected tensor size ({value.shape}), using mean")
                    return float(value.mean().item())
            elif isinstance(value, np.ndarray):
                if value.size == 1:
                    return float(value.item())
                else:
                    print(f"Warning: Multi-element array (size={value.shape}), using mean")
                    return float(np.mean(value))
            else:
                return float(value)

        with torch.no_grad():
            for episode in batch:
                if hasattr(episode, 'rewards') and episode.rewards:
                    for r in episode.rewards:
                        episode_rewards.append(to_python_float(r))
                if hasattr(episode, 'values') and episode.values:
                    for v in episode.values:
                        episode_values.append(to_python_float(v))
                if hasattr(episode, 'target_values') and episode.target_values:
                    for tv in episode.target_values:
                        episode_target_values.append(to_python_float(tv))

        # 손실 누적 변수
        total_loss = 0
        value_loss = 0
        reward_loss = 0
        policy_loss = 0
        
        
        # 1단계: 유효한 에피소드 및 시작점 수집 (배치 준비)
        valid_episodes = []
        start_indices = []
        for episode in batch:
            if len(episode) >= 2:  # 유효한 에피소드만
                max_start = max(0, len(episode) - K)
                start_idx = np.random.randint(0, max_start + 1)
                valid_episodes.append(episode)
                start_indices.append(start_idx)
        
        if not valid_episodes:
            continue
        
        # 2단계: 초기 관측값들을 배치로 처리
        initial_states = []
        for episode, start_idx in zip(valid_episodes, start_indices):
            state = episode.states[start_idx]
            if isinstance(state, dict):
                initial_states.append(state)
            else:
                print(f"경고: state가 dict가 아님: {type(state)}")
                continue
        
        if not initial_states:
            continue
        
        # 배치 단위 observation 생성 (최적화 핵심!)
        from .utils import build_observation_batch
        batch_observations = build_observation_batch(
            initial_states, config.observation_shape, device, config.use_obstacle_channels
        )
        
        # 3단계: 배치 단위 초기 추론 (Mixed Precision + 배치 최적화!)
        # 이론적 배경: 
        # - autocast Context Manager: 순전파에서 자동으로 FP16/FP32 정밀도 선택
        # - Tensor Cores 활용: 행렬 곱셈, 합성곱 등이 하드웨어 가속으로 2-4배 빨라짐
        # - 배치 처리: 개별 처리 대신 전체 배치를 한번에 GPU로 전송하여 처리량 최대화
        with autocast(enabled=use_amp):
            batch_values, _, batch_policy_logits, batch_hidden_states = network.initial_inference(batch_observations)
        

        reward_count = 0 
        value_count = 0 
        policy_count = 0
        # 4단계: 배치의 각 항목에 대해 손실 계산
        actual_batch_size = len(valid_episodes)

        # 모든 타겟 값들을 CPU에서 먼저 수집 (GPU 할당 최소화)
        all_target_values = []
        all_target_policies = []
        all_legal_masks = []

        for i in range(actual_batch_size):
            episode = valid_episodes[i]
            start_idx = start_indices[i]

            # Target value 수집
            if hasattr(episode, 'target_values') and episode.target_values is not None:
                target_value_scalar = episode.target_values[start_idx] if start_idx < len(episode.target_values) else 0.0
            else:
                target_value_scalar = episode.values[start_idx] if episode.values[start_idx] is not None else 0.0
            all_target_values.append(target_value_scalar)

            # Target policy 수집
            if episode.pi[start_idx] is not None:
                all_target_policies.append(episode.pi[start_idx])
            else:
                all_target_policies.append(None)

            # Legal mask 수집
            if hasattr(episode, 'legal_mask') and episode.legal_mask[start_idx] is not None:
                all_legal_masks.append(episode.legal_mask[start_idx])
            else:
                all_legal_masks.append(None)

        # CPU에서 수집한 데이터를 한 번에 GPU로 전송
        target_values_tensor = torch.tensor(all_target_values, dtype=torch.float32).to(device, non_blocking=True)

        # 배치 전체 처리
        for i in range(actual_batch_size):
            episode = valid_episodes[i]
            start_idx = start_indices[i]
            value = batch_values[i:i+1]  # 개별 항목 추출 (배치 차원 유지)
            policy_logits = batch_policy_logits[i:i+1]
            hidden_state = batch_hidden_states[i:i+1] if batch_hidden_states is not None else None

            # 첫 스텝의 손실 계산 (k=0)
            # Value loss - 미리 전송된 텐서 사용
            target_value_dist = scalar_to_support(
                target_values_tensor[i:i+1],
                config
            )
            # Categorical cross-entropy loss
            value_loss += -(target_value_dist * torch.nn.functional.log_softmax(value, dim=1)).sum(1).mean()
            value_count += 1

            # Policy loss
            # [참고] MuZero 논문: MCTS pi가 이미 합법 액션만 포함하므로 추가 masking 불필요
            if all_target_policies[i] is not None:
                # CPU에서 GPU로 한 번에 전송
                target_policy = torch.tensor(all_target_policies[i], dtype=torch.float32).to(device, non_blocking=True).unsqueeze(0)

                # KL divergence loss (target_policy는 확률 분포)
                log_policy = torch.nn.functional.log_softmax(policy_logits, dim=1)
                policy_loss += torch.nn.functional.kl_div(log_policy, target_policy, reduction='batchmean')
                policy_count += 1
            else:
                # GT 데이터의 경우 policy가 없으므로 합법 액션에 대해 uniform distribution 사용
                num_actions = policy_logits.shape[1]
                # CPU에서 생성 후 GPU로 전송
                target_policy = torch.ones(1, num_actions, dtype=torch.float32) / num_actions
                target_policy = target_policy.to(device, non_blocking=True)

                # Legal mask 적용 (GT는 MCTS를 거치지 않으므로 여기서만 필요)
                if all_legal_masks[i] is not None:
                    legal_mask = torch.tensor(all_legal_masks[i], dtype=torch.float32).to(device, non_blocking=True).unsqueeze(0)
                    target_policy = target_policy * legal_mask
                    policy_sum = target_policy.sum(dim=1, keepdim=True)
                    if policy_sum > 0:
                        target_policy = target_policy / policy_sum

                log_policy = torch.nn.functional.log_softmax(policy_logits, dim=1)
                policy_loss += torch.nn.functional.kl_div(log_policy, target_policy, reduction='batchmean')
                policy_count += 1
            

            # K 스텝 unroll
            for k in range(0, K):
                if start_idx + k + 1 >= len(episode):
                    break

                # 액션 (k번째 액션으로 dynamics 수행) - CPU에서 생성 후 GPU로
                action = torch.tensor([episode.actions[start_idx + k]], dtype=torch.long).to(device, non_blocking=True)
                
                # Recurrent inference (dynamics + prediction) - Mixed Precision 적용
                # reward_k, latent_{k+1} = dynamics(latent_k, action_k)
                # policy_{k+1}, value_{k+1} = prediction(latent_{k+1})
                # IMPORTANT: hidden_state.detach()로 computational graph 끊기 (메모리 누수 방지)
                with autocast(enabled=use_amp):
                    value, reward, policy_logits, hidden_state = network.recurrent_inference(
                        hidden_state.detach() if hidden_state is not None else hidden_state, action
                    )
                
                # Reward loss (k번째 액션의 reward)
                target_reward_scalar = episode.rewards[start_idx + k]
                # Two-hot categorical distribution - CPU에서 생성 후 GPU로
                target_reward_dist = scalar_to_support(
                    torch.tensor([target_reward_scalar], dtype=torch.float32).to(device, non_blocking=True),
                    config
                )
                # Categorical cross-entropy loss
                reward_loss += -(target_reward_dist * torch.nn.functional.log_softmax(reward, dim=1)).sum(1).mean()
                reward_count += 1

                # Value loss (k+1번째 상태의 value)
                # target_values 사용 (n-step bootstrap으로 계산된 학습 타깃)
                if hasattr(episode, 'target_values') and episode.target_values is not None:
                    if start_idx + k + 1 < len(episode.target_values):
                        target_value_scalar = episode.target_values[start_idx + k + 1]
                else:
                    # Fallback: target_values가 없으면 values 사용
                    if start_idx + k + 1 < len(episode.values):
                        target_value_scalar = episode.values[start_idx + k + 1] if episode.values[start_idx + k + 1] is not None else target_value_scalar
                # Two-hot categorical distribution - CPU에서 생성 후 GPU로
                target_value_dist = scalar_to_support(
                    torch.tensor([target_value_scalar], dtype=torch.float32).to(device, non_blocking=True),
                    config
                )
                # Categorical cross-entropy loss
                value_loss += -(target_value_dist * torch.nn.functional.log_softmax(value, dim=1)).sum(1).mean()
                value_count += 1

                # Policy loss (k+1번째 상태의 policy)
                if start_idx + k + 1 < len(episode):
                    if episode.pi[start_idx + k + 1] is not None:
                        target_policy = torch.tensor(episode.pi[start_idx + k + 1], dtype=torch.float32).to(device, non_blocking=True).unsqueeze(0)

                        # KL divergence loss (MCTS pi가 이미 합법 액션만 포함)
                        log_policy = torch.nn.functional.log_softmax(policy_logits, dim=1)
                        policy_loss += torch.nn.functional.kl_div(log_policy, target_policy, reduction='batchmean')
                        policy_count += 1
                    else:
                        # GT 데이터의 경우 policy가 없으므로 합법 액션에 대해 uniform distribution 사용
                        num_actions = policy_logits.shape[1]
                        target_policy = torch.ones(1, num_actions, dtype=torch.float32) / num_actions
                        target_policy = target_policy.to(device, non_blocking=True)

                        # Legal mask 적용 (GT는 MCTS를 거치지 않으므로 여기서만 필요)
                        if hasattr(episode, 'legal_mask') and start_idx + k + 1 < len(episode.legal_mask) and episode.legal_mask[start_idx + k + 1] is not None:
                            legal_mask = torch.tensor(episode.legal_mask[start_idx + k + 1], dtype=torch.float32).to(device, non_blocking=True).unsqueeze(0)
                            target_policy = target_policy * legal_mask
                            policy_sum = target_policy.sum(dim=1, keepdim=True)
                            if policy_sum > 0:
                                target_policy = target_policy / policy_sum

                        log_policy = torch.nn.functional.log_softmax(policy_logits, dim=1)
                        policy_loss += torch.nn.functional.kl_div(log_policy, target_policy, reduction='batchmean')
                        policy_count += 1
        
        # 평균 손실 계산
        value_loss = value_loss / max(1, value_count) * config.value_loss_weight
        reward_loss = reward_loss / max(1, reward_count) * config.reward_loss_weight
        policy_loss = policy_loss / max(1, policy_count) * config.policy_loss_weight
        
        # 총 손실 (Gradient Accumulation 정규화 적용)
        # MuZero 이론적 배경:
        # - Value Loss: 상태 가치 함수 V(s) 학습 → representation과 prediction 네트워크 동시 업데이트
        # - Reward Loss: 즉시 보상 R(s,a) 학습 → dynamics 네트워크 업데이트  
        # - Policy Loss: 정책 π(s) 학습 → prediction 네트워크의 정책 헤드 업데이트
        # - 백프로파게이션으로 모든 네트워크 컴포넌트가 end-to-end로 학습됨
        # 
        # Gradient Accumulation 이론:
        # - 작은 배치들의 gradient를 누적하여 큰 배치의 효과 시뮬레이션
        # - gradient_accumulation_steps로 나누어 평균 gradient 계산 (bias 없는 추정)
        total_loss = (value_loss + reward_loss + policy_loss) / gradient_accumulation_steps
        
        # Mixed Precision 역전파 처리
        # 이론적 배경:
        # - FP16 연산에서 gradient underflow 방지를 위해 loss scaling 적용
        # - scaler.scale(): 손실값을 큰 수로 곱해서 작은 gradient도 FP16 범위에서 표현 가능
        # - 역전파 후 gradient를 다시 원래 스케일로 복원
        if use_amp:
            scaler.scale(total_loss).backward()  # 스케일된 손실로 역전파
        else:
            total_loss.backward()  # 표준 FP32 역전파
        
        # Gradient Accumulation: 설정된 스텝마다만 가중치 업데이트
        # 이론: 여러 forward pass의 gradient를 누적한 후 한번에 optimizer step
        # 효과적 배치 크기 = batch_size × gradient_accumulation_steps
        if (step + 1) % gradient_accumulation_steps == 0:
            if use_amp:
                # Mixed Precision Gradient Processing
                # 1. unscale_(): gradient를 원래 스케일로 복원 (clipping 전 필수)
                # 2. clip_grad_norm_(): gradient exploding 방지 (norm > 5.0이면 clipping)
                # 3. step(): 스케일링된 gradient가 유효하면 가중치 업데이트
                # 4. update(): 다음 iteration을 위한 scaler 상태 업데이트
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)  # 메모리 효율적인 gradient 초기화
            else:
                # Standard Precision Gradient Processing
                # gradient clipping으로 훈련 안정성 확보 후 가중치 업데이트
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)  # 메모리 효율적인 gradient 초기화
        
        # 로깅
        if step % 10 == 0:
            print(f"Step {step}: Loss={total_loss:.4f} "
                  f"(V={value_loss:.4f}, R={reward_loss:.4f}, P={policy_loss:.4f})")

            # TensorBoard 로깅 (전역 스텝 사용)
            global_step = base_step + step
            writer.add_scalar('Loss/Total', total_loss.item(), global_step)
            writer.add_scalar('Loss/Value', value_loss.item(), global_step)
            writer.add_scalar('Loss/Reward', reward_loss.item(), global_step)
            writer.add_scalar('Loss/Policy', policy_loss.item(), global_step)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
            writer.add_scalar('Training/Iteration', iteration_num, global_step)

            # Episode 통계 TensorBoard 로깅
            if len(episode_rewards) > 0:
                mean_reward = sum(episode_rewards) / len(episode_rewards)
                sum_reward = sum(episode_rewards)
                writer.add_scalar('Episode/Mean_Reward', mean_reward, global_step)
                writer.add_scalar('Episode/Sum_Reward', sum_reward, global_step)
                writer.add_scalar('Episode/Std_Reward', float(np.std(episode_rewards)), global_step)

            if len(episode_values) > 0:
                mean_value = sum(episode_values) / len(episode_values)
                writer.add_scalar('Episode/Mean_Value', mean_value, global_step)
                writer.add_scalar('Episode/Std_Value', float(np.std(episode_values)), global_step)

            if len(episode_target_values) > 0:
                mean_tv = sum(episode_target_values) / len(episode_target_values)
                writer.add_scalar('Episode/Mean_Target_Value', mean_tv, global_step)
                writer.add_scalar('Episode/Std_Target_Value', float(np.std(episode_target_values)), global_step)

            # Value-Target 상관계수 계산
            if len(episode_values) > 0 and len(episode_target_values) > 0 and len(episode_values) == len(episode_target_values):
                correlation = np.corrcoef(episode_values, episode_target_values)[0, 1]
                writer.add_scalar('Metrics/Value_Target_Correlation', float(correlation), global_step)
        
        # 학습률 업데이트
        if (step + 1) % config.lr_decay_steps == 0:
            lr_scheduler.step()

        
        # Loss history 업데이트
        loss_dict = {
            'total_loss': total_loss.item(),
            'value_loss': value_loss.item(),
            'reward_loss': reward_loss.item(),
            'policy_loss': policy_loss.item()
        }
        if loss_dict:
            # 전역 스텝으로 loss_history 업데이트
            global_step = base_step + step + 1
            loss_history['steps'].append(global_step)
            loss_history['total_loss'].append(loss_dict['total_loss'])
            loss_history['value_loss'].append(loss_dict['value_loss'])
            loss_history['policy_loss'].append(loss_dict['policy_loss'])
            loss_history['reward_loss'].append(loss_dict['reward_loss'])
            loss_history['latest_loss'] = loss_dict
            
            # print(f"  📊 Loss: Total={loss_dict['total_loss']:.4f}, "
            #     f"Value={loss_dict['value_loss']:.4f}, "
            #     f"Policy={loss_dict['policy_loss']:.4f}, "
            #     f"Reward={loss_dict['reward_loss']:.4f}")
        
        # 2.5 학습 진행 상황 시각화 (선택적) ... iteration_num
        # if (step + 1) % config.visualization_interval_step == 0 and config.enable_training_visualization:
        if (step + 1) % config.visualization_interval_step == 0 and (iteration_num + 1) % config.visualization_interval_iteration == 0 and config.enable_training_visualization:
            try:
                visualize_training_progress(network, replay_buffer, step + 1, config, device, loss_history, iteration_num)
                # visualize_training_progress(network, step + 1, config, device, loss_history, iteration_num,candidate_episodes=valid_episodes)
            except Exception as e:
                print(f"    ⚠️ 시각화 중 오류 발생: {e}")
                logger.info(f"Visualization error at step {step+1}: {e}")
        
        # 3. 주기적으로 체크포인트 저장
        if (iteration_num + 1) % config.pth_checkpoint_interval_iteration == 0:
            checkpoint_dir = config.results_path / config.checkpoint_save_dir
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            save_path = checkpoint_dir / f"iter{iteration_num}.pth"
            torch.save({
                'iteration': iteration_num + 1,
                'network_state_dict': network.state_dict(),
                'config': config.__dict__
            }, save_path)
            print(f"  💾 체크포인트 저장: {save_path}")
            logger.checkpoint_saved(save_path)
    
    # Writer 종료 (이 함수에서 생성한 경우만)
    if close_writer:
        writer.close()

    # GPU 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print("\n학습 완료!")

    # 마지막 loss 값들 반환
    return {
        'total_loss': total_loss.item() if total_loss != 0 else 0.0,
        'value_loss': value_loss.item() if value_loss != 0 else 0.0,
        'reward_loss': reward_loss.item() if reward_loss != 0 else 0.0,
        'policy_loss': policy_loss.item() if policy_loss != 0 else 0.0
    } 



# def visualize_training_progress(network, candidate_episodes, step, config, device, loss_history=None, iteration_num=1):
def visualize_training_progress(network, replay_buffer, step, config, device, loss_history=None, iteration_num=1):
    """학습 중 모델의 예측 성능과 Loss를 시각화
    
    1. replay_buffer에서 Episode를 config 에 선언된 visualization num episodes 개수만큼 샘플링
    2. training 으로 학습된 network 를 사용하여 MCTS enrichment 수행(enrich_episodes_with_mcts())
    3. 시각화 수행 (Predicted Path + Loss)
    
    """
    print(f"\n  📊 step {step} 학습 진행 상황 시각화...")
    
    # 시각화할 Episode 샘플링 (config에서 설정)
    import copy
    # episodes = candidate_episodes[:min(config.visualization_num_episodes, len(candidate_episodes))]
    episodes = copy.deepcopy(replay_buffer.sample_episodes(min(config.visualization_num_episodes, len(replay_buffer))))
    
    if not episodes:
        print("    시각화할 Episode가 없습니다.")
        return
    
    # Self-play 매니저 생성 (MCTS enrichment를 위해)
    
    # Mock environment와 provider (시각화용이므로 실제 환경 불필요)
    class MockObservationProvider:
        def get_observation(self):
            return None
    
    # 1. Network state_dict 추출 (pickle 가능)
    network_state_dict = network.state_dict()
    
    # 각 Episode에 대해 시각화
    for ep_idx, episode in enumerate(episodes):
        # if len(episode) < config.visualization_min_episode_length:  # 너무 짧은 Episode는 스킵
        #     continue
            
        # 전체 Episode 시각화
        visualization_steps = len(episode)
        
        # Episode 전체를 복사 (수정하지 않기 위해)
        from models.MuZero.replay_buffer import Episode
        full_episode = Episode()
        for i in range(visualization_steps):
            full_episode.states.append(episode.states[i])
            full_episode.actions.append(0)  # 임시 action
            full_episode.rewards.append(0.0)  # 임시 reward
            full_episode.values.append(0.0)  # 임시 value
            full_episode.pi.append(None)  # 임시 π
            full_episode.legal_mask.append(None)  # 임시 legal_mask (MCTS에서 계산됨)
            full_episode.dones.append(False)  # 임시 done

        # expert_ego_pos는 states보다 길 수 있으므로 별도로 모든 위치 복사
        full_episode.expert_ego_pos = episode.expert_ego_pos.copy()

        # 학습 전 Episode의 Predicted Path 추출 (보라색 경로용)
        old_predicted_rows = []
        old_predicted_cols = []
        for i in range(len(episode.states)):
            state = episode.states[i]
            if isinstance(state, dict) and 'simulated_ego_pos' in state:
                pos = state['simulated_ego_pos']
                old_predicted_rows.append(pos[0])
                old_predicted_cols.append(pos[1])
            else:
                # Fallback: 중앙 위치
                old_predicted_rows.append(49)
                old_predicted_cols.append(49)

        if old_predicted_rows:
            print(f"    📍 Old Predicted path: start({old_predicted_rows[0]}, {old_predicted_cols[0]}) -> end({old_predicted_rows[-1]}, {old_predicted_cols[-1]})")

        # 현재 네트워크로 MCTS enrichment 수행
        print(f"    Episode {ep_idx+1}: {visualization_steps} 스텝 처리 중...")
        # enriched_episodes = self_play_manager.enrich_episodes_with_mcts([full_episode])
        # enriched_episode = enriched_episodes[0]
        args = (ep_idx, network_state_dict, config, device, episode)
        enriched_episode = enrich_episode_with_mcts_parallel_worker(args)
        
        # 시각화 - Loss 정보가 있으면 2개 subplot, 없으면 1개
        if loss_history and len(loss_history['steps']) > 0:
            fig, (ax, ax_loss) = plt.subplots(1, 2, figsize=(config.visualization_figsize[0]*2, config.visualization_figsize[1]))
        else:
            fig, ax = plt.subplots(1, 1, figsize=config.visualization_figsize)
        
        # 그리드 설정 (99x99) - visualize_muzero_episode.py와 동일한 방식
        ax.set_xlim(0, 99)
        ax.set_ylim(0, 99)
        ax.set_aspect('equal')
        # invert_yaxis() 제거 - 자연스러운 수학 좌표계 (아래가 0, 위로 갈수록 증가)
        ax.grid(True, alpha=0.3)
        # 제목에 출처 정보 포함
        title = f'step {step} - Episode {ep_idx+1} (All {visualization_steps} steps)'
        if hasattr(episode, 'source_file') and episode.source_file:
            title += f'\nSource: {episode.source_file}'
            if hasattr(episode, 'start_frame') and hasattr(episode, 'end_frame'):
                title += f' (Frames: {episode.start_frame}-{episode.end_frame})'
        ax.set_title(title)
        
        # 첫 프레임의 환경 정보 시각화 (배경으로)
        # Shift 적용으로 각 스텝의 observation이 다르므로, 원본 좌표계인 첫 스텝 사용
        if len(enriched_episode.states) > 0:
            last_state = enriched_episode.states[0]  # 첫 프레임 사용 -> 디버깅 목적이면, 마지막인 -1 로 설정
            
            # Channel 0: Dynamic objects (빨간색 계열)
            if 'obstacles' in last_state and last_state['obstacles'] is not None:
                obstacles = last_state['obstacles']
                obstacle_mask = obstacles > 0
                obstacle_points = np.where(obstacle_mask)
                ax.scatter(obstacle_points[1], obstacle_points[0], 
                          c='red', s=20, alpha=0.3, marker='s', label='Dynamic Objects')
            
            # Channel 1: Static environment (노란색 계열) - 더 진한 색상
            if 'static_environment' in last_state and last_state['static_environment'] is not None:
                static_env = last_state['static_environment']
                static_mask = static_env > 0
                static_points = np.where(static_mask)
                ax.scatter(static_points[1], static_points[0], 
                          c='gold', s=20, alpha=0.7, marker='^', label='Static Environment')
            
            # Channel 2: Ground class_id (LDM segmentation 색상) - 모든 클래스 그리기
            if 'ground_class_id' in last_state and last_state['ground_class_id'] is not None:
                class_id_grid = last_state['ground_class_id']
                
                # 각 클래스별 픽셀 수 계산
                unique_class_ids, counts = np.unique(class_id_grid, return_counts=True)
                sorted_indices = np.argsort(counts)[::-1]  # 카운트 순 정렬
                
                # 모든 클래스 그리기 (범례 제한 없이)
                legend_count = 0
                for idx in sorted_indices:
                    class_id = unique_class_ids[idx]
                    count = counts[idx]
                    
                    # Background(0)이거나 너무 적은 픽셀은 제외
                    if class_id == 0 or count < 10:
                        continue
                    
                    class_mask = class_id_grid == class_id
                    class_points = np.where(class_mask)
                    color = get_grid_color(class_id, channel=2)
                    class_name = get_ldm_segmentation_name(int(class_id))
                    
                    # 범례는 상위 5개만, 나머지는 범례 없이 그리기
                    if legend_count < 5:
                        label = f'{class_name}' if class_name else f'Class {int(class_id)}'
                        legend_count += 1
                    else:
                        label = None  # 범례에 표시하지 않음
                    
                    ax.scatter(class_points[1], class_points[0], 
                              c=color, s=15, alpha=0.4, marker='o', label=label)
            
            # Direction vectors (화살표로 표시)
            center_row, center_col = 49, 49  # 그리드 중앙
            
            # Near destination direction (초록색 화살표)
            if 'near_destination_direction' in last_state and last_state['near_destination_direction']:
                near_dir = last_state['near_destination_direction']
                if near_dir and 'x' in near_dir and 'y' in near_dir:
                    # x는 전방(양수가 앞), y는 좌우(양수가 오른쪽)
                    dx = near_dir['x'] * 10  # x는 전방 (row 방향 상방 +)
                    dy = near_dir['y'] * 10  # y는 우측방 (col 방향 우측 +)
                    # matplotlib 좌표계는 x축 좌우(우가 +), y축 상하(상이 +) 임. 그래서 거꾸로 넣어줘야함
                    ax.arrow(center_col, center_row, dy, dx,
                            head_width=2, head_length=1, fc='green', ec='green',
                            alpha=0.7, label='Near Direction')
            
            # Final destination direction (파란색 화살표)
            if 'final_destination_direction' in last_state and last_state['final_destination_direction']:
                final_dir = last_state['final_destination_direction']
                if final_dir and 'x' in final_dir and 'y' in final_dir:
                    # x는 전방(양수가 앞), y는 좌우(양수가 오른쪽)
                    dx = final_dir['x'] * 15  # x는 전방 (row 방향 상방 +), 더 길게
                    dy = final_dir['y'] * 15  # y는 우측방 (col 방향 우측 +), 더 길게
                    # matplotlib 좌표계는 x축 좌우(우가 +), y축 상하(상이 +) 임. 그래서 거꾸로 넣어줘야함
                    ax.arrow(center_col, center_row, dy, dx,
                            head_width=2, head_length=1, fc='darkblue', ec='darkblue',
                            alpha=0.7, label='Final Direction')
        
        # Expert 경로 (파란색 그라데이션: 연한색 -> 진한색)
        expert_rows = [pos[0] for pos in enriched_episode.expert_ego_pos]
        expert_cols = [pos[1] for pos in enriched_episode.expert_ego_pos]
        
        # Expert Path 그라데이션 그리기
        if len(expert_rows) > 1:
            for i in range(len(expert_rows) - 1):
                # 진행도에 따른 알파값 (0.3 -> 1.0)
                alpha = 0.3 + 0.7 * (i / (len(expert_rows) - 1))
                # 진행도에 따른 색상 강도 (연한 파란색 -> 진한 파란색)
                color_intensity = 0.4 + 0.6 * (i / (len(expert_rows) - 1))
                color = (0, 0, color_intensity)
                
                # 첫 번째 선분에만 label 추가 (범례용)
                label = 'Expert Path' if i == 0 else None
                ax.plot([expert_cols[i], expert_cols[i+1]], 
                       [expert_rows[i], expert_rows[i+1]], 
                       color=color, linewidth=2, alpha=alpha, label=label)
                ax.scatter(expert_cols[i], expert_rows[i], 
                          c=[color], s=30 + 20 * (i / max(1, len(expert_rows) - 1)), 
                          alpha=alpha, zorder=5)
            # 마지막 점
            ax.scatter(expert_cols[-1], expert_rows[-1], 
                      c='darkblue', s=50, alpha=1.0, zorder=5)
        else:
            ax.scatter(expert_cols, expert_rows, c='blue', s=50, zorder=5, label='Expert Path')

        # Old Predicted Path (보라색 그라데이션: 학습 전 경로)
        if old_predicted_rows and len(old_predicted_rows) > 1 and len(set(zip(old_predicted_rows, old_predicted_cols))) > 1:
            for i in range(len(old_predicted_rows) - 1):
                # 진행도에 따른 알파값 (0.3 -> 1.0)
                alpha = 0.3 + 0.7 * (i / (len(old_predicted_rows) - 1))
                # 진행도에 따른 색상 강도 (연한 보라색 -> 진한 보라색)
                color_intensity = 0.4 + 0.6 * (i / (len(old_predicted_rows) - 1))
                color = (color_intensity * 0.5, 0, color_intensity)  # 보라색 (R과 B 혼합)

                # 첫 번째 선분에만 label 추가 (범례용)
                label = 'Old Predicted (Before Training)' if i == 0 else None
                ax.plot([old_predicted_cols[i], old_predicted_cols[i+1]],
                       [old_predicted_rows[i], old_predicted_rows[i+1]],
                       color=color, linewidth=2, alpha=alpha, linestyle=':', label=label)  # dotted 라인
                ax.scatter(old_predicted_cols[i], old_predicted_rows[i],
                          c=[color], s=25 + 15 * (i / max(1, len(old_predicted_rows) - 1)),
                          alpha=alpha, zorder=4)
            # 마지막 점
            ax.scatter(old_predicted_cols[-1], old_predicted_rows[-1],
                      c='purple', s=45, alpha=1.0, zorder=4)
        elif old_predicted_rows and len(old_predicted_rows) > 0:
            # 모든 점이 같은 위치인 경우 점으로만 표시
            ax.scatter(old_predicted_cols[0], old_predicted_rows[0], c='purple', s=80, marker='x',
                      label='Old Predicted (No Movement)', zorder=4, alpha=0.7)

        # MCTS로 계산된 simulated 경로 (빨간색: 학습 후 경로)
        # enrich_episodes_with_mcts에서 이미 simulated_ego_pos가 계산되어 states에 저장됨
        sim_rows = []
        sim_cols = []
        
        # 디버그: Episode 정보 출력
        unique_actions = set(enriched_episode.actions)
        print(f"🔍 Episode {ep_idx+1} - States: {len(enriched_episode.states)}, Expert pos: {len(enriched_episode.expert_ego_pos)}")
        print(f"🔍 Actions: {unique_actions}, Total: {len(enriched_episode.actions)}")

        # Action별 Target Reward 평균 계산
        action_rewards = {}
        for action, reward in zip(enriched_episode.actions, enriched_episode.rewards):
            if action not in action_rewards:
                action_rewards[action] = []
            action_rewards[action].append(reward)

        action_avg_str = ", ".join([f"{action}:{np.mean(rewards):.3f}" for action, rewards in sorted(action_rewards.items())])
        print(f"🎯 Action별 Target Reward: {action_avg_str}")
        
        # Episode states에서 직접 simulated_ego_pos 추출
        fallback_count = 0
        for i in range(len(enriched_episode.states)):
            state = enriched_episode.states[i]
            if isinstance(state, dict) and 'simulated_ego_pos' in state:
                sim_pos = state['simulated_ego_pos']
                sim_rows.append(sim_pos[0])
                sim_cols.append(sim_pos[1])
            else:
                # fallback: 중앙 위치
                sim_rows.append(49)
                sim_cols.append(49)
                fallback_count += 1

        if fallback_count > 0:
            print(f"⚠️ {fallback_count}/{len(enriched_episode.states)} 스텝에서 simulated_ego_pos 누락으로 fallback 사용")
        
        print(f"🔍 New Predicted path: start({sim_rows[0]}, {sim_cols[0]}) -> end({sim_rows[-1]}, {sim_cols[-1]})")

        # New Predicted Path (빨간색 그라데이션: 학습 후 경로)
        if len(set(zip(sim_rows, sim_cols))) > 1:  # 서로 다른 위치가 2개 이상 있을 때만
            # New Predicted Path 그라데이션 그리기
            for i in range(len(sim_rows) - 1):
                # 진행도에 따른 알파값 (0.3 -> 1.0)
                alpha = 0.3 + 0.7 * (i / (len(sim_rows) - 1))
                # 진행도에 따른 색상 강도 (연한 빨간색 -> 진한 빨간색)
                color_intensity = 0.4 + 0.6 * (i / (len(sim_rows) - 1))
                color = (color_intensity, 0, 0)

                # 첫 번째 선분에만 label 추가 (범례용)
                label = 'New Predicted (After Training)' if i == 0 else None
                ax.plot([sim_cols[i], sim_cols[i+1]], 
                       [sim_rows[i], sim_rows[i+1]], 
                       color=color, linewidth=2, alpha=alpha, linestyle='--', label=label)
                ax.scatter(sim_cols[i], sim_rows[i], 
                          c=[color], s=30 + 20 * (i / max(1, len(sim_rows) - 1)), 
                          alpha=alpha, zorder=5)
            # 마지막 점
            ax.scatter(sim_cols[-1], sim_rows[-1], 
                      c='darkred', s=50, alpha=1.0, zorder=5)
        else:
            # 모든 점이 같은 위치인 경우 점으로만 표시
            ax.scatter(sim_cols[0], sim_rows[0], c='red', s=100, marker='x', label='New Predicted (No Movement)', zorder=5)
        
        # 시작점과 끝점 표시
        ax.scatter(expert_cols[0], expert_rows[0], c='green', s=100, marker='o', label='Start', zorder=6)
        if len(expert_cols) > 1:
            ax.scatter(expert_cols[-1], expert_rows[-1], c='orange', s=100, marker='s', label='End', zorder=6)
        
        # Target Goal 위치 표시 (reward 계산에 사용되는 Episode 끝 위치)
        if hasattr(enriched_episode, 'expert_ego_pos') and len(enriched_episode.expert_ego_pos) > 0:
            goal_pos = enriched_episode.expert_ego_pos[-1]  # Episode의 마지막 expert 위치
            ax.scatter(goal_pos[1], goal_pos[0], c='purple', s=120, marker='*', 
                      label='Target Goal', zorder=7, edgecolors='black', linewidth=1)
            
            # Target Goal 주변에 원뿔 형태의 포텐셜 필드 표시 (참고용)
            circle = plt.Circle((goal_pos[1], goal_pos[0]), radius=5, 
                              fill=False, color='purple', linestyle=':', alpha=0.5, zorder=3)
            ax.add_patch(circle)
        
        # 범례 (Channel 2 추가로 항목 증가, 더 많은 컬럼으로 조정)
        ax.legend(loc='upper right', fontsize=7, ncol=3, bbox_to_anchor=(1.0, 1.0))
        
        # Loss 정보 텍스트 박스 추가 (최신 Loss)
        if loss_history and 'latest_loss' in loss_history:
            latest = loss_history['latest_loss']
            loss_text = f"Latest Loss (step {step}):\n"
            loss_text += f"Total: {latest['total_loss']:.4f}\n"
            loss_text += f"Value: {latest['value_loss']:.4f}\n"
            loss_text += f"Policy: {latest['policy_loss']:.4f}\n"
            loss_text += f"Reward: {latest['reward_loss']:.4f}"
            ax.text(0.02, 0.98, loss_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # Episode 통계 및 Target Values 정보 추가
        stats_text = f"Episode {ep_idx+1} Stats:\n"
        stats_text += f"Steps: {len(enriched_episode.states)}\n"
        
        
        # Target Values 정보
        if hasattr(enriched_episode, 'target_values') and enriched_episode.target_values:
            stats_text += f"\n[Target Values]\n"
            stats_text += f"Avg: {np.mean(enriched_episode.target_values):.3f}\n"
            stats_text += f"Min/Max: {np.min(enriched_episode.target_values):.3f}/{np.max(enriched_episode.target_values):.3f}\n"
        
        # MCTS Predictions (for comparison)
        if enriched_episode.values:
            stats_text += f"\n[MCTS Values]\n"
            stats_text += f"Avg: {np.mean(enriched_episode.values):.3f}\n"
        
        # Target Policy 정보 (가장 많이 선택된 action)
        if enriched_episode.pi and any(p is not None for p in enriched_episode.pi):
            from models.MuZero.action import Action
            # 각 action의 평균 확률 계산
            num_actions = len(enriched_episode.pi[0]) if enriched_episode.pi[0] is not None else 9
            avg_pi = np.zeros(num_actions)
            count = 0
            for pi in enriched_episode.pi:
                if pi is not None:
                    avg_pi += pi
                    count += 1
            if count > 0:
                avg_pi /= count
                top_action_idx = np.argmax(avg_pi)
                top_action = Action.from_index(top_action_idx)
                stats_text += f"\n[Target π (Pi)]\n"
                stats_text += f"Top Action: {top_action.position.value} ({avg_pi[top_action_idx]:.2%})"
        
        # 방향 벡터 정보 추가 (첫 번째 step의 방향 벡터 사용)
        if enriched_episode.states and len(enriched_episode.states) > 0:
            first_state = enriched_episode.states[0]
            if isinstance(first_state, dict):
                near_dir = first_state.get('near_destination_direction')
                dest_dir = first_state.get('final_destination_direction')
                
                if near_dir or dest_dir:
                    stats_text += f"\n[Direction Vectors]\n"
                    
                if near_dir and isinstance(near_dir, dict) and 'x' in near_dir and 'y' in near_dir:
                    stats_text += f"Near: x={near_dir['x']:.3f}, y={near_dir['y']:.3f}\n"
                
                if dest_dir and isinstance(dest_dir, dict) and 'x' in dest_dir and 'y' in dest_dir:
                    stats_text += f"Final: x={dest_dir['x']:.3f}, y={dest_dir['y']:.3f}"
        
        # Target Goal 위치 정보 추가
        if hasattr(enriched_episode, 'expert_ego_pos') and len(enriched_episode.expert_ego_pos) > 0:
            goal_pos = enriched_episode.expert_ego_pos[-1]
            stats_text += f"\n[Target Goal]\n"
            stats_text += f"Position: ({goal_pos[0]}, {goal_pos[1]})\n"
            
            # 현재 simulated 위치에서 목표까지의 거리
            if sim_rows and sim_cols:
                current_pos = (sim_rows[-1], sim_cols[-1])
                distance_to_goal = np.sqrt((current_pos[0] - goal_pos[0])**2 + (current_pos[1] - goal_pos[1])**2)
                stats_text += f"Distance: {distance_to_goal:.2f} cells"
        
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=8,
               verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        # Action별 Target Reward 통계 (우측 하단에 표시)
        if enriched_episode.actions and enriched_episode.rewards:
            from models.MuZero.action import Action
            from collections import defaultdict
            
            # Action별 reward 수집
            action_rewards = defaultdict(list)
            for action_idx, reward in zip(enriched_episode.actions, enriched_episode.rewards):
                action_rewards[action_idx].append(reward)
            
            # 텍스트 생성 시작
            action_stats_text = "[Action-Reward Details]\n"
            
            # 1. Step별 Action-Reward (처음 30스텝까지)
            action_stats_text += "First 30 Steps:\n"
            max_display_steps = min(30, len(enriched_episode.actions))
            for ep_step in range(max_display_steps):
                if ep_step < len(enriched_episode.actions) and ep_step < len(enriched_episode.rewards):
                    action_obj = Action.from_index(enriched_episode.actions[ep_step])
                    reward = enriched_episode.rewards[ep_step]

                    # 해당 스텝의 simulated_ego_pos 가져오기
                    pos_info = ""
                    if (ep_step < len(enriched_episode.states) and
                        isinstance(enriched_episode.states[ep_step], dict) and
                        'simulated_ego_pos' in enriched_episode.states[ep_step]):
                        pos = enriched_episode.states[ep_step]['simulated_ego_pos']
                        pos_info = f" @{pos}"

                    # Target Value도 함께 표시
                    value_info = ""
                    if hasattr(enriched_episode, 'target_values') and ep_step < len(enriched_episode.target_values):
                        target_value = enriched_episode.target_values[ep_step]
                        value_info = f" | Val: {target_value:.3f}"
                    elif ep_step < len(enriched_episode.values):
                        value = enriched_episode.values[ep_step]
                        value_info = f" | Val: {value:.3f}"

                    action_stats_text += f"  {ep_step+1}: {action_obj.position.value}{pos_info} → {reward:.3f}{value_info}\n"
            if len(enriched_episode.actions) > max_display_steps:
                action_stats_text += f"  ... (+{len(enriched_episode.actions) - max_display_steps} more)\n"
            
            # 2. Action별 평균 reward
            action_stats_text += "\nAction Averages:\n"
            for action_idx in sorted(action_rewards.keys()):
                action_obj = Action.from_index(action_idx)
                rewards = action_rewards[action_idx]
                avg_reward = np.mean(rewards)
                count = len(rewards)
                
                # Action의 이동 벡터 정보 추가
                dx, dy = action_obj.get_movement_vector()
                vec_info = f"({dx:+d},{dy:+d})"
                
                action_stats_text += f"{action_obj.position.value} {vec_info}: {avg_reward:.3f} (×{count})\n"
            
            # 3. 전체 Target Rewards 통계
            action_stats_text += f"\nTotal Rewards:\n"
            action_stats_text += f"Sum: {sum(enriched_episode.rewards):.2f}\n"
            action_stats_text += f"Avg: {np.mean(enriched_episode.rewards):.3f}\n"
            action_stats_text += f"Range: {np.min(enriched_episode.rewards):.3f}~{np.max(enriched_episode.rewards):.3f}\n"
            action_stats_text += f"Non-zero: {np.count_nonzero(enriched_episode.rewards)}/{len(enriched_episode.rewards)}"
            
            # 우측 하단에 표시
            ax.text(0.98, 0.02, action_stats_text, transform=ax.transAxes, fontsize=6,
                   verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        # Loss 그래프 그리기 (있는 경우)
        if loss_history and len(loss_history['steps']) > 0 and 'ax_loss' in locals():
            steps = loss_history['steps']
            ax_loss.plot(steps, loss_history['total_loss'], 'b-', label='Total Loss', linewidth=2)
            ax_loss.plot(steps, loss_history['value_loss'], 'g--', label='Value Loss', alpha=0.7)
            ax_loss.plot(steps, loss_history['policy_loss'], 'r--', label='Policy Loss', alpha=0.7)
            ax_loss.plot(steps, loss_history['reward_loss'], 'y--', label='Reward Loss', alpha=0.7)
            
            ax_loss.set_xlabel('step', fontsize=12)
            ax_loss.set_ylabel('Loss', fontsize=12)
            ax_loss.set_title(f'Training Loss History (up to step {step})', fontsize=14)
            ax_loss.legend(loc='upper right')
            ax_loss.grid(True, alpha=0.3)
            
            # 현재 step 표시
            if step in steps:
                idx = steps.index(step)
                ax_loss.scatter([step], [loss_history['total_loss'][idx]], 
                              c='red', s=100, zorder=5, marker='o')
                ax_loss.annotate(f'Current\n{loss_history["total_loss"][idx]:.4f}',
                                xy=(step, loss_history['total_loss'][idx]),
                                xytext=(10, 10), textcoords='offset points',
                                fontsize=9, ha='left',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # 저장
        save_dir = config.results_path / "visualizations"
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / f"iter{iteration_num}_step_{step}_episode_{ep_idx+1}.png"
        plt.savefig(save_path, dpi=config.visualization_dpi, bbox_inches='tight')
        plt.close()
        print(f"    💾 시각화 저장: {save_path}")

    # 시각화 완료 후 GPU 메모리 정리
    # enrich_episodes_with_mcts()에서 생성한 GPU Tensor들 해제
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"    🧹 시각화 메모리 정리 완료")


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
            config.num_categorical_distribution_atoms,
            # 새로운 파라미터들
            use_coordconv=config.use_coordconv,
            use_goal_map=config.use_goal_map,
            use_film=config.use_film,
            alpha_init=config.alpha_init,
            use_resnet_dynamics=config.use_resnet_dynamics,
        ).to(device)
    else:
        raise NotImplementedError(
            '지원되는 네트워크 설정은 "fullyconnected"입니다.'
        )
