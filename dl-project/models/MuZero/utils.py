"""MuZero 공통 유틸리티 함수"""
import numpy as np
import torch
from typing import List, Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .self_play import GameHistory


def calculate_game_statistics(games: List['GameHistory']) -> Dict[str, float]:
    """게임 통계 계산
    
    Args:
        games: GameHistory 리스트
        
    Returns:
        통계 딕셔너리
    """
    if not games:
        return {
            'num_games': 0,
            'avg_length': 0,
            'std_length': 0,
            'avg_reward': 0,
            'std_reward': 0,
            'max_reward': 0,
            'min_reward': 0
        }
    
    game_lengths = [game.get_length() for game in games]
    total_rewards = [sum(game.rewards) for game in games]
    
    stats = {
        'num_games': len(games),
        'avg_length': np.mean(game_lengths),
        'std_length': np.std(game_lengths),
        'avg_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'max_reward': np.max(total_rewards),
        'min_reward': np.min(total_rewards)
    }
    
    return stats


def print_game_statistics(stats: Dict[str, float], title: str = "게임 통계"):
    """통계 출력
    
    Args:
        stats: calculate_game_statistics로 생성된 통계
        title: 출력 제목
    """
    print(f"\n{title} ({stats['num_games']}게임):")
    print(f"- 평균 길이: {stats['avg_length']:.1f} (±{stats['std_length']:.1f})")
    print(f"- 평균 보상: {stats['avg_reward']:.2f} (±{stats['std_reward']:.2f})")
    print(f"- 최대 보상: {stats['max_reward']:.2f}")
    print(f"- 최소 보상: {stats['min_reward']:.2f}")
    
    
def print_episode_summary(episode_idx: int, game: 'GameHistory', show_actions: bool = True):
    """에피소드 요약 출력
    
    Args:
        episode_idx: 에피소드 번호 (0-based)
        game: GameHistory 객체
        show_actions: 액션 분포 출력 여부
    """
    from .action import Action
    import json
    
    print(f"\n📊 Episode {episode_idx + 1} 결과:")
    print(f"   - 총 스텝: {game.get_length()}")
    print(f"   - 총 보상: {sum(game.rewards):.2f}")
    
    if show_actions:
        # 액션 통계
        action_counts = {}
        for action_idx in game.actions:
            action = Action.from_index(action_idx)
            key = str(action)
            action_counts[key] = action_counts.get(key, 0) + 1
        print(f"   - 액션 분포: {json.dumps(action_counts, indent=6)}")


def build_observation_tensor(state_dict: Dict, observation_shape: Tuple[int, int, int], device: str = 'cpu', use_obstacle_channels: bool = None) -> torch.Tensor:
    """
    state_dict로부터 6채널 observation tensor를 생성하는 공용 함수
    
    Args:
        state_dict: Episode의 state dictionary
        observation_shape: (channels, height, width) 형태의 관측값 shape
        device: 텐서를 생성할 디바이스
        
    Returns:
        6채널 observation tensor (shape: observation_shape)
    """
    # 6채널 observation 생성
    observation_tensor = torch.zeros(observation_shape, device=device)
    
    # Channel 0~2: 장애물 채널 사용 여부
    if use_obstacle_channels is None:
        # 파라미터가 없으면 환경 변수에서 읽기 (fallback)
        import os
        USE_OBSTACLE_CHANNELS = os.getenv('USE_OBSTACLE_CHANNELS', 'False').lower() == 'true'
    else:
        USE_OBSTACLE_CHANNELS = use_obstacle_channels
    
    if USE_OBSTACLE_CHANNELS:
        # Channel 0: Dynamic obstacles
        if 'obstacles' in state_dict and state_dict['obstacles'] is not None:
            data = state_dict['obstacles']
            if isinstance(data, torch.Tensor):
                observation_tensor[0] = data.to(device)
            else:
                observation_tensor[0] = torch.from_numpy(data).to(device)
        
        # Channel 1: Static environment
        if 'static_environment' in state_dict and state_dict['static_environment'] is not None:
            data = state_dict['static_environment']
            if isinstance(data, torch.Tensor):
                observation_tensor[1] = data.to(device)
            else:
                observation_tensor[1] = torch.from_numpy(data).to(device)
        
        # Channel 2: Ground class_id
        if 'ground_class_id' in state_dict and state_dict['ground_class_id'] is not None:
            data = state_dict['ground_class_id']
            if isinstance(data, torch.Tensor):
                observation_tensor[2] = data.to(device)
            else:
                observation_tensor[2] = torch.from_numpy(data).to(device)
    else:
        # 명시적으로 0으로 유지 (데이터 로드하지 않음)
        # observation_tensor[0:3]는 이미 zeros()로 초기화되어 있음
        pass
    
    # Channel 3: Near destination direction
    if 'near_destination_direction' in state_dict:
        near_dir = state_dict['near_destination_direction']
        if near_dir:
            observation_tensor[3, 0, 0] = near_dir.get('x', 0.0)
            observation_tensor[3, 0, 1] = near_dir.get('y', 0.0)
    
    # Channel 4: Final destination direction
    if 'final_destination_direction' in state_dict:
        final_dir = state_dict['final_destination_direction']
        if final_dir:
            observation_tensor[4, 0, 0] = final_dir.get('x', 0.0)
            observation_tensor[4, 0, 1] = final_dir.get('y', 0.0)
    
    # Channel 5: Simulated ego position
    if 'simulated_ego_pos' in state_dict:
        ego_pos = state_dict['simulated_ego_pos']
        if ego_pos is not None:
            # ego_pos는 (x, y) 튜플 또는 (row, col) 튜플
            if isinstance(ego_pos, tuple) and len(ego_pos) == 2:
                # expert_data_loader에서는 (row, col) 형태로 전달
                # environment에서는 (x, y) 형태로 전달
                if isinstance(ego_pos[0], int):  # (row, col) 형태
                    row, col = ego_pos
                    center = observation_shape[1] // 2  # 49
                    ego_x = float(row - center)  # 전방이 양수
                    ego_y = float(center - col)  # 오른쪽이 양수
                else:  # (x, y) 형태
                    ego_x, ego_y = ego_pos
            else:
                print("Warning: Invalid simulated ego position format, using default (0.0, 0.0)")
                ego_x, ego_y = 0.0, 0.0
            
            # Channel 5의 (0, 0)에 x, (0, 1)에 y 저장
            observation_tensor[5, 0, 0] = float(ego_x)
            observation_tensor[5, 0, 1] = float(ego_y)
    
    return observation_tensor