import torch
import numpy as np
import copy
from pathlib import Path
import datetime
from torch.utils.tensorboard import SummaryWriter
from models.MuZero.muzero_model import MuZeroFullyConnectedNetwork, MuZeroResidualNetwork
from config.config import MuZeroConfig

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
    # 설정 생성
    config = MuZeroConfig()
    
    # 네트워크 생성
    network = get_network(config)
    
    # 랜덤 관찰 생성
    observation = torch.rand(1, *config.observation_shape)
    
    # 네트워크 추론 실행
    value, reward, policy_logits, encoded_state = network.initial_inference(observation)
    
    print(f"Value: {value}")
    print(f"Reward: {reward}")
    print(f"Policy logits: {policy_logits}")
    print(f"Encoded state shape: {encoded_state.shape}")
    
    # 랜덤 액션 생성
    action = torch.tensor([[0]])
    
    # 재귀적 추론 실행
    next_value, next_reward, next_policy_logits, next_encoded_state = network.recurrent_inference(encoded_state, action)
    
    print(f"Next value: {next_value}")
    print(f"Next reward: {next_reward}")
    print(f"Next policy logits: {next_policy_logits}")
    print(f"Next encoded state shape: {next_encoded_state.shape}")

if __name__ == "__main__":
    main()
