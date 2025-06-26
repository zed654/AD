import torch
import numpy as np
from pathlib import Path
import datetime
from torch.utils.tensorboard import SummaryWriter

# MuZero 관련 import
from models.MuZero.models import MuZeroFullyConnectedNetwork
from models.MuZero.config import MuZeroConfig
from models.MuZero.mcts import MCTS
from models.MuZero.action import Action

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
    """입력 / GT 데이터 가져오기 """
    # data/db3_parser/results 에서 가져오기
    
    
    """MuZero 네트워크와 설정을 생성하고 사용하는 예제 함수입니다."""
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 설정 생성
    config = MuZeroConfig()
    
    # 네트워크 생성
    network = get_network(config).to(device) # 안써도 되긴 함. get_network 에서 DataParallel 을 사용하기 때문에
    
    mcts = MCTS(network, len(config.action_space), config.num_simulations) # MCTS 생성
    
    # 랜덤 관찰 생성
    observation = torch.rand(1, *config.observation_shape).to(device) # 추후 실제 관측값으로 입력 필요
    
    result = mcts.run(observation)
    selected_action_index = result['action']
    
    # Action 의 Index to String 작업 수행
    selected_action = Action.from_index(selected_action_index)
    
    print(f"Selected action index: {selected_action_index}")
    print(f"Selected action: {selected_action}")
    print(f"  Position: {selected_action.position.value}")
    print(f"  StopAndGo: {selected_action.stopgo.value}")
    print(f"Root value: {result['root_value']:.4f}")
    
    # 정책 타겟을 보기 좋게 출력
    print("\nPolicy distribution:")
    policy = result['policy_targets']
    top_actions = np.argsort(policy)[-5:][::-1]  # 상위 5개 액션
    for action_idx in top_actions:
        if policy[action_idx] > 0.01:  # 1% 이상인 액션만 표시
            action = Action.from_index(action_idx)
            print(f"  {action}: {policy[action_idx]:.3f}")

if __name__ == "__main__":
    main()
