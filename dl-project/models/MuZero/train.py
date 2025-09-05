"""
MuZero 학습 모듈
신경망 학습 및 관련 유틸리티 함수들
"""
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from .utils import build_observation_tensor


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
    # ex) support_size = 20
    
    # Clamp scalars to the valid support range
    # 1. 범위 제한: -20 ~ +20 (support_size 만큼 범위 제한)
    scalars = torch.clamp(scalars, -support_size, support_size)
    
    # 2. 인덱스로 변환
    # This is a simplified version - MuZero paper uses a more sophisticated method
    # -20.3 → -1, -19 → 1, ..., 0 → 20, ..., 19.4 → 39
    floor_idx = torch.floor(scalars + support_size).long()
    
    # 3. 범위 제한: 0 ~ 40 (2 * support_size)
    # 최대치 제한: 양수의 최고치인 support_size * 2 를 넘어가면 범위 제한
    # 최소치 제한: 음수를 제외하기위해 0으로 제한
    floor_idx = torch.clamp(floor_idx, 0, 2 * support_size)
    
    return floor_idx


# def calculate_target_value(episode, start_idx, discount, n_steps=5):
#     """n-step return 계산
    
#     MuZero는 n-step return + bootstrap value를 사용합니다.
#     target_value = sum(discount^k * reward_k) + discount^n * value_n
#     bootstrap value 는 n_steps 만큼 적용됨 (n_steps 만큼 미래의 가치를 현재에 대입하는 것)
    
#     * Bellman equation 원리임 - Baysian Theory 의 반대 개념으로, 미래의 가치를 현재에 대입하는 것
#     """
#     if start_idx >= len(episode):
#         return 0.0
    
#     target_value = 0.0
    
#     # n-step return 계산
#     for i in range(n_steps):
#         if start_idx + i >= len(episode):
#             break
#         target_value += (discount ** i) * episode.rewards[start_idx + i]
    
#     # Bootstrap value 추가 (n스텝 후의 value)
#     bootstrap_idx = start_idx + n_steps
#     if bootstrap_idx < len(episode) and episode.values[bootstrap_idx] is not None:
#         target_value += (discount ** n_steps) * episode.values[bootstrap_idx]
    
#     return target_value


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
    
    # Mixed Precision Training (AMP) 설정
    # 이론적 배경:
    # - 혼합 정밀도 학습: FP32(32비트)와 FP16(16비트) 부동소수점을 혼합 사용
    # - GPU Tensor Cores 활용: V100/A100 등에서 FP16 행렬 곱셈이 2-4배 빠름
    # - 메모리 절약: FP16 사용으로 GPU 메모리 사용량 약 50% 감소
    # - 수치적 안정성: GradScaler로 gradient underflow 방지 (loss scaling)
    use_amp = getattr(config, 'use_mixed_precision', False) and device != 'cpu'
    scaler = GradScaler() if use_amp else None  # FP16 gradient scaling으로 underflow 방지
    
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
    
    # TensorBoard 설정
    writer = SummaryWriter(log_dir=config.results_path / "logs")
    
    # 학습 루프
    print("\n=== MuZero 네트워크 학습 시작 ===")
    for step in range(training_steps):
        # 배치 샘플링
        batch = replay_buffer.get_batch(batch_size) # Replay Buffer 에서 batch_size 만큼 랜덤하게 Episode 를 뽑아냄 
        if batch is None:
            print(f"Step {step}: 배치 샘플링 실패 (데이터 부족)")
            continue
        
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
        
        # 4단계: 배치의 각 항목에 대해 손실 계산
        actual_batch_size = len(valid_episodes)
        for i in range(actual_batch_size):
            episode = valid_episodes[i]
            start_idx = start_indices[i]
            value = batch_values[i:i+1]  # 개별 항목 추출 (배치 차원 유지)
            policy_logits = batch_policy_logits[i:i+1]
            hidden_state = batch_hidden_states[i:i+1] if batch_hidden_states is not None else None
            
            # 첫 스텝의 손실 계산 (k=0)
            # Value loss
            # target_values 사용 (n-step bootstrap으로 계산된 학습 타깃)
            if hasattr(episode, 'target_values') and episode.target_values is not None:
                target_value_scalar = episode.target_values[start_idx] if start_idx < len(episode.target_values) else 0.0
            else:
                # Fallback: target_values가 없으면 values 사용
                target_value_scalar = episode.values[start_idx] if episode.values[start_idx] is not None else 0.0
            target_value_idx = scalar_to_support(
                torch.tensor([target_value_scalar], device=device),
                config.support_size
            )
            value_loss += torch.nn.functional.cross_entropy(value, target_value_idx)
            
            # Policy loss
            # [참고] MuZero 논문에서는 CrossEntropy 사용했음
            if episode.policies[start_idx] is not None:
                target_policy = torch.tensor(episode.policies[start_idx], device=device).unsqueeze(0)
                # KL divergence loss (target_policy는 확률 분포)
                log_policy = torch.nn.functional.log_softmax(policy_logits, dim=1)
                policy_loss += torch.nn.functional.kl_div(log_policy, target_policy, reduction='batchmean')
            else:
                # GT 데이터의 경우 policy가 없으므로 uniform distribution 사용
                num_actions = policy_logits.shape[1]
                target_policy = torch.ones(1, num_actions, device=device) / num_actions
                log_policy = torch.nn.functional.log_softmax(policy_logits, dim=1)
                policy_loss += torch.nn.functional.kl_div(log_policy, target_policy, reduction='batchmean')
            
            # K 스텝 unroll
            for k in range(0, K):
                if start_idx + k + 1 >= len(episode):
                    break
                    
                # 액션 (k번째 액션으로 dynamics 수행)
                action = torch.tensor([episode.actions[start_idx + k]], device=device)
                
                # Recurrent inference (dynamics + prediction) - Mixed Precision 적용
                # reward_k, latent_{k+1} = dynamics(latent_k, action_k)
                # policy_{k+1}, value_{k+1} = prediction(latent_{k+1})
                with autocast(enabled=use_amp):
                    value, reward, policy_logits, hidden_state = network.recurrent_inference(
                        hidden_state, action
                    )
                
                # Reward loss (k번째 액션의 reward)
                target_reward_scalar = episode.rewards[start_idx + k]
                target_reward_idx = scalar_to_support(
                    torch.tensor([target_reward_scalar], device=device),
                    config.support_size
                )
                reward_loss += torch.nn.functional.cross_entropy(reward, target_reward_idx)
                
                # Value loss (k+1번째 상태의 value)
                # target_values 사용 (n-step bootstrap으로 계산된 학습 타깃)
                if hasattr(episode, 'target_values') and episode.target_values is not None:
                    if start_idx + k + 1 < len(episode.target_values):
                        target_value_scalar = episode.target_values[start_idx + k + 1]
                else:
                    # Fallback: target_values가 없으면 values 사용
                    if start_idx + k + 1 < len(episode.values):
                        target_value_scalar = episode.values[start_idx + k + 1] if episode.values[start_idx + k + 1] is not None else target_value_scalar
                target_value_idx = scalar_to_support(
                    torch.tensor([target_value_scalar], device=device),
                    config.support_size
                )
                value_loss += torch.nn.functional.cross_entropy(value, target_value_idx)
                
                # Policy loss (k+1번째 상태의 policy)
                if start_idx + k + 1 < len(episode):
                    if episode.policies[start_idx + k + 1] is not None:
                        target_policy = torch.tensor(episode.policies[start_idx + k + 1], device=device).unsqueeze(0)
                        # KL divergence loss (target_policy는 확률 분포)
                        log_policy = torch.nn.functional.log_softmax(policy_logits, dim=1)
                        policy_loss += torch.nn.functional.kl_div(log_policy, target_policy, reduction='batchmean')
                    else:
                        # GT 데이터의 경우 policy가 없으므로 uniform distribution 사용
                        num_actions = policy_logits.shape[1]
                        target_policy = torch.ones(1, num_actions, device=device) / num_actions
                        log_policy = torch.nn.functional.log_softmax(policy_logits, dim=1)
                        policy_loss += torch.nn.functional.kl_div(log_policy, target_policy, reduction='batchmean')
        
        # 평균 손실 계산
        num_samples = len(batch)
        value_loss = value_loss / (num_samples * (K + 1))
        reward_loss = reward_loss / (num_samples * K)
        policy_loss = policy_loss / (num_samples * (K + 1))
        
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
                optimizer.zero_grad()
            else:
                # Standard Precision Gradient Processing  
                # gradient clipping으로 훈련 안정성 확보 후 가중치 업데이트
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad()
        
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
    
    # 마지막 loss 값들 반환
    return {
        'total_loss': total_loss.item() if total_loss != 0 else 0.0,
        'value_loss': value_loss.item() if value_loss != 0 else 0.0,
        'reward_loss': reward_loss.item() if reward_loss != 0 else 0.0,
        'policy_loss': policy_loss.item() if policy_loss != 0 else 0.0
    } 