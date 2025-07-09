"""
MuZero 학습 모듈
신경망 학습 및 관련 유틸리티 함수들
"""
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter


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


def calculate_target_value(episode, start_idx, discount, n_steps=5):
    """n-step return 계산
    
    MuZero는 n-step return + bootstrap value를 사용합니다.
    target_value = sum(discount^k * reward_k) + discount^n * value_n
    bootstrap value 는 n_steps 만큼 적용됨 (n_steps 만큼 미래의 가치를 현재에 대입하는 것)
    
    * Bellman equation 원리임 - Baysian Theory 의 반대 개념으로, 미래의 가치를 현재에 대입하는 것
    """
    if start_idx >= len(episode):
        return 0.0
    
    target_value = 0.0
    
    # n-step return 계산
    for i in range(n_steps):
        if start_idx + i >= len(episode):
            break
        target_value += (discount ** i) * episode.rewards[start_idx + i]
    
    # Bootstrap value 추가 (n스텝 후의 value)
    bootstrap_idx = start_idx + n_steps
    if bootstrap_idx < len(episode) and episode.values[bootstrap_idx] is not None:
        target_value += (discount ** n_steps) * episode.values[bootstrap_idx]
    
    return target_value


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
    print("\n=== MuZero 네트워크 학습 시작 ===")
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
            
            # 초기 관측값 ... PyTorch 모델 배치형태 맞추려고 unsqueeze 함 ([B, C, H, W] -> [1, C, H, W])
            observation = episode.states[start_idx].unsqueeze(0).to(device)
            
            # 초기 추론 (representation + prediction)
            # latent_0 = representation(observation)
            # policy_0, value_0 = prediction(latent_0)
            value, _, policy_logits, hidden_state = network.initial_inference(observation)
            
            # 첫 스텝의 손실 계산 (k=0)
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
            for k in range(0, K):
                if start_idx + k + 1 >= len(episode):
                    break
                    
                # 액션 (k번째 액션으로 dynamics 수행)
                action = torch.tensor([episode.actions[start_idx + k]], device=device)
                
                # Recurrent inference (dynamics + prediction)
                # reward_k, latent_{k+1} = dynamics(latent_k, action_k)
                # policy_{k+1}, value_{k+1} = prediction(latent_{k+1})
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
                target_value_scalar = calculate_target_value(episode, start_idx + k + 1, config.discount)
                target_value_idx = scalar_to_support(
                    torch.tensor([target_value_scalar], device=device),
                    config.support_size
                )
                value_loss += torch.nn.functional.cross_entropy(value, target_value_idx)
                
                # Policy loss (k+1번째 상태의 policy)
                if start_idx + k + 1 < len(episode):
                    target_policy = torch.tensor(episode.policies[start_idx + k + 1], device=device).unsqueeze(0)
                    # KL divergence loss (target_policy는 확률 분포)
                    log_policy = torch.nn.functional.log_softmax(policy_logits, dim=1)
                    policy_loss += torch.nn.functional.kl_div(log_policy, target_policy, reduction='batchmean')
        
        # 평균 손실 계산
        num_samples = len(batch)
        value_loss = value_loss / (num_samples * (K + 1))
        reward_loss = reward_loss / (num_samples * K)
        policy_loss = policy_loss / (num_samples * (K + 1))
        
        # 총 손실
        # value, reward, policy 손실을 합하지만, 내부적으로는 계산 히스토리가 남아있어서 문제없음
        #   즉, value, policy 의 손실은 prediction(latent) 을 업데이트하고, reward 의 손실은 dynamics(latent, action) 를 업데이트함
        #   또한, representation() 도 간접적으로 업데이트가 됨
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