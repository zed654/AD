import torch
import numpy as np
import copy
from pathlib import Path
import datetime
from torch.utils.tensorboard import SummaryWriter
from models.MuZero.muzero_model import MuZeroFullyConnectedNetwork
from models.MuZero.config import MuZeroConfig

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

def main():
    """MuZero 네트워크와 설정을 생성하고 사용하는 예제 함수입니다."""
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 설정 생성
    config = MuZeroConfig()
    
    # 네트워크 생성
    network = get_network(config).to(device) # 안써도 되긴 함. get_network 에서 DataParallel 을 사용하기 때문에
    print(f"network.device: {next(network.parameters()).device}")
    
    # 랜덤 관찰 생성
    observation = torch.rand(1, *config.observation_shape).to(device)
    print(f"observation.device: {observation.device}")
    
    # 네트워크 추론 실행
    value, reward, policy_logits, latent = network.initial_inference(observation)
    print(f"latent.device: {latent.device}")
    print(f"value.device: {value.device}")
    print(f"reward.device: {reward.device}")
    print(f"policy_logits.device: {policy_logits.device}")
    
    print(f"Value: {value}")
    print(f"Reward: {reward}")
    print(f"Policy logits: {policy_logits}")
    print(f"Latent shape: {latent.shape}")
    
    # 랜덤 액션 생성
    action = torch.tensor([[0]]).to(device)
    print(f"action.device: {action.device}")
    
    # 재귀적 추론 실행
    next_value, next_reward, next_policy_logits, next_latent = network.recurrent_inference(latent, action)
    
    print(f"Next value: {next_value}")
    print(f"Next reward: {next_reward}")
    print(f"Next policy logits: {next_policy_logits}")
    print(f"Next latent shape: {next_latent.shape}")

if __name__ == "__main__":
    main()
