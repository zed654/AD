"""
MuZero 학습 모듈
신경망 학습 및 관련 유틸리티 함수들
"""
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from .utils import build_observation_tensor
from .models import MuZeroNetwork
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

        # 손실 누적 변수 (리스트로 변경 - PyTorch 표준 방법)
        value_losses = []
        reward_losses = []
        policy_losses = []
        
        
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
            value_losses.append(-(target_value_dist * torch.nn.functional.log_softmax(value, dim=1)).sum(1).mean())

            # Policy loss
            # [참고] MuZero 논문: MCTS pi가 이미 합법 액션만 포함하므로 추가 masking 불필요
            if all_target_policies[i] is not None:
                # CPU에서 GPU로 한 번에 전송
                target_policy = torch.tensor(all_target_policies[i], dtype=torch.float32).to(device, non_blocking=True).unsqueeze(0)

                # KL divergence loss (target_policy는 확률 분포)
                log_policy = torch.nn.functional.log_softmax(policy_logits, dim=1)
                policy_losses.append(torch.nn.functional.kl_div(log_policy, target_policy, reduction='batchmean'))
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
                policy_losses.append(torch.nn.functional.kl_div(log_policy, target_policy, reduction='batchmean'))

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
                    detached_hidden = hidden_state.detach() if hidden_state is not None else None
                    value, reward, policy_logits, hidden_state = network.recurrent_inference(
                        detached_hidden, action
                    )
                
                # Reward loss (k번째 액션의 reward)
                target_reward_scalar = episode.rewards[start_idx + k]
                # Two-hot categorical distribution - CPU에서 생성 후 GPU로
                target_reward_dist = scalar_to_support(
                    torch.tensor([target_reward_scalar], dtype=torch.float32).to(device, non_blocking=True),
                    config
                )
                # Categorical cross-entropy loss
                reward_losses.append(-(target_reward_dist * torch.nn.functional.log_softmax(reward, dim=1)).sum(1).mean())

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
                value_losses.append(-(target_value_dist * torch.nn.functional.log_softmax(value, dim=1)).sum(1).mean())

                # Policy loss (k+1번째 상태의 policy)
                if start_idx + k + 1 < len(episode):
                    if episode.pi[start_idx + k + 1] is not None:
                        target_policy = torch.tensor(episode.pi[start_idx + k + 1], dtype=torch.float32).to(device, non_blocking=True).unsqueeze(0)

                        # KL divergence loss (MCTS pi가 이미 합법 액션만 포함)
                        log_policy = torch.nn.functional.log_softmax(policy_logits, dim=1)
                        policy_losses.append(torch.nn.functional.kl_div(log_policy, target_policy, reduction='batchmean'))
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
                        policy_losses.append(torch.nn.functional.kl_div(log_policy, target_policy, reduction='batchmean'))

        # 평균 손실 계산 (PyTorch 표준 방법: 리스트의 텐서들을 stack하여 평균)
        value_loss = torch.stack(value_losses).mean() * config.value_loss_weight if value_losses else torch.tensor(0.0, device=device)
        reward_loss = torch.stack(reward_losses).mean() * config.reward_loss_weight if reward_losses else torch.tensor(0.0, device=device)
        policy_loss = torch.stack(policy_losses).mean() * config.policy_loss_weight if policy_losses else torch.tensor(0.0, device=device)

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

        
        # Loss history 업데이트 (tensor를 item()으로 변환)
        loss_dict = {
            'total_loss': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
            'value_loss': value_loss.item() if isinstance(value_loss, torch.Tensor) else value_loss,
            'reward_loss': reward_loss.item() if isinstance(reward_loss, torch.Tensor) else reward_loss,
            'policy_loss': policy_loss.item() if isinstance(policy_loss, torch.Tensor) else policy_loss
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

def get_network(config, device):
    """설정에 따라 적절한 MuZero 네트워크를 생성합니다."""
    if config.network == "fullyconnected":
        return MuZeroNetwork(
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
            use_resnet=config.use_resnet,
        ).to(device)
    else:
        raise NotImplementedError(
            '지원되는 네트워크 설정은 "fullyconnected"입니다.'
        )
