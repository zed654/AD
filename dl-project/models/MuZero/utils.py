"""MuZero 공통 유틸리티 함수"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .self_play import GameHistory

from rosetta import (get_grid_value_info, get_ldm_segmentation_name, get_ldm_segmentation_color, 
                     GRID_VALUE_EMPTY, GRID_VALUE_PEDESTRIAN, GRID_VALUE_VEHICLE, GRID_VALUE_OTHER_OBJECT, 
                     GRID_VALUE_CLIFF, GRID_VALUE_STATIC_OBSTACLE, GRID_VALUE_MISSING_POINTS)
import sys
def calculate_memory(obj, seen=None):
    """Recursively calculate memory usage of objects including Tensors and Numpy arrays"""
    if seen is None:
        seen = set()
    
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    
    size = sys.getsizeof(obj)
    
    if torch.is_tensor(obj):
        size += obj.element_size() * obj.nelement()
    elif isinstance(obj, np.ndarray):
        size += obj.nbytes
    elif isinstance(obj, dict):
        size += sum(calculate_memory(k, seen) + calculate_memory(v, seen) for k, v in obj.items())
    elif hasattr(obj, '__dict__'):
        size += calculate_memory(obj.__dict__, seen)
    elif isinstance(obj, (list, tuple, set)):
        size += sum(calculate_memory(i, seen) for i in obj)
        
    return size

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
    
    

# 글로벌 텐서 풀 (메모리 풀링 기법을 통한 성능 최적화)
# 이론적 배경: 
# - 딥러닝에서 메모리 할당/해제는 GPU 성능 병목의 주요 원인 중 하나
# - 텐서 풀링(Tensor Pooling)은 미리 할당된 메모리를 재사용하여 GPU 메모리 할당 오버헤드 제거
# - CUDA 커널 런치 비용도 감소시켜 전체 처리량(throughput) 향상
_tensor_pool = {}


def get_pooled_tensor(shape: Tuple[int, int, int], device: str) -> torch.Tensor:
    """
    메모리 풀에서 재사용 가능한 텐서 획득 (Zero-Copy 메모리 관리 최적화)
    
    이론적 배경:
    - Memory Pool Pattern: 미리 할당된 메모리 블록을 재사용하여 동적 할당 비용 제거
    - Pin Memory: CPU-GPU 데이터 전송 시 페이징 없는 메모리 사용으로 DMA 전송 가속화
    - Zero Initialization: 기존 텐서를 0으로 초기화하여 새 할당 없이 재사용
    
    Args:
        shape: 텐서 크기 (channels, height, width)
        device: 계산 디바이스 ('cuda' 또는 'cpu')
        
    Returns:
        초기화된 재사용 텐서 (메모리 할당 오버헤드 없음)
    """
    key = (shape, str(device))
    if key not in _tensor_pool:
        # 최초 생성: pin_memory=True로 CPU-GPU 전송 최적화
        # pin_memory는 페이지 고정 메모리로 DMA(Direct Memory Access) 가능
        _tensor_pool[key] = torch.zeros(shape, device=device, pin_memory=(device == 'cpu'))
    else:
        # 기존 텐서를 0으로 초기화 (in-place 연산으로 재할당 없음)
        # zero_() 메서드는 메모리 주소 변경 없이 값만 0으로 설정
        _tensor_pool[key].zero_()
    return _tensor_pool[key]


def build_observation_tensor(state_dict: Dict, observation_shape: Tuple[int, int, int], device: str = 'cpu', use_obstacle_channels: bool = None) -> torch.Tensor:
    """
    state_dict로부터 6채널 observation tensor를 생성하는 최적화된 공용 함수
    
    이론적 배경:
    - MuZero의 observation은 다차원 상태 표현으로 환경의 모든 관측 정보를 인코딩
    - Multi-Channel Representation: 각 채널이 서로 다른 의미적 정보를 담아 CNN의 feature extraction 최적화
    - Memory Layout Optimization: 연속적인 메모리 배치로 CUDA 텐서 연산 효율성 향상
    - Asynchronous Transfer: non_blocking=True로 CPU-GPU 데이터 전송과 연산 병렬화
    
    채널 구성:
    - Channel 0: Dynamic objects (차량, 보행자)
    - Channel 1: Static environment (절벽, 정적 장애물, 누락 포인트)
    - Channel 2: Ground class_id (지면 의미론적 정보)
    - Channel 3: Near destination direction (근거리 목적지 방향)
    - Channel 4: Final destination direction (최종 목적지 방향)
    - Channel 5: Simulated ego position (시뮬레이션된 자아 위치)
    
    Args:
        state_dict: Episode의 state dictionary (원시 센서 데이터)
        observation_shape: (channels, height, width) 형태의 관측값 shape
        device: 텐서를 생성할 디바이스 ('cuda' 권장)
        use_obstacle_channels: 장애물 채널 사용 여부 (메모리 최적화용)
        
    Returns:
        6채널 observation tensor (MuZero 네트워크 입력 형식)
    """
    # 텐서 풀에서 재사용 텐서 획득 (동적 메모리 할당 비용 제거)
    # 이론: 매번 torch.zeros() 호출 시 GPU 메모리 할당/해제 오버헤드 발생
    # 풀링으로 58,806개 요소(6×99×99) 텐서 재사용하여 성능 향상
    observation_tensor = get_pooled_tensor(observation_shape, device)
    
    # Channel 0~2: 장애물 채널 사용 여부 (한번만 체크)
    if use_obstacle_channels is None:
        import os
        USE_OBSTACLE_CHANNELS = os.getenv('USE_OBSTACLE_CHANNELS', 'False').lower() == 'true'
    else:
        USE_OBSTACLE_CHANNELS = use_obstacle_channels
    
    if USE_OBSTACLE_CHANNELS:
        # Channel 0: Dynamic obstacles - Zero-Copy GPU 메모리 직접 복사
        # 이론적 배경: copy_() 메서드는 메모리 재할당 없이 데이터만 복사하는 in-place 연산
        # non_blocking=True로 CPU-GPU 전송을 비동기화하여 연산과 전송 병렬화
        if 'obstacles' in state_dict and state_dict['obstacles'] is not None:
            data = state_dict['obstacles']
            if isinstance(data, torch.Tensor):
                if data.device != device:
                    # 디바이스 간 데이터 전송 (비동기 처리)
                    # non_blocking=True: CPU 코드 실행을 차단하지 않고 CUDA 스트림에서 처리
                    observation_tensor[0].copy_(data.to(device, non_blocking=True))
                else:
                    # 같은 디바이스 내 복사 (메모리 대역폭 최적화)
                    observation_tensor[0].copy_(data)
            else:
                # NumPy → PyTorch 변환 최적화
                # 이론: torch.from_numpy()는 메모리 공유(zero-copy)로 변환하여 속도 향상
                temp_tensor = torch.from_numpy(data)
                if device != 'cpu':
                    # CPU 텐서를 GPU로 비동기 전송
                    observation_tensor[0].copy_(temp_tensor.to(device, non_blocking=True))
                else:
                    # CPU 내 복사 (시스템 메모리 대역폭 활용)
                    observation_tensor[0].copy_(temp_tensor)
        
        # Channel 1: Static environment - 동일한 최적화 적용
        if 'static_environment' in state_dict and state_dict['static_environment'] is not None:
            data = state_dict['static_environment']
            if isinstance(data, torch.Tensor):
                if data.device != device:
                    observation_tensor[1].copy_(data.to(device, non_blocking=True))
                else:
                    observation_tensor[1].copy_(data)
            else:
                temp_tensor = torch.from_numpy(data)
                if device != 'cpu':
                    observation_tensor[1].copy_(temp_tensor.to(device, non_blocking=True))
                else:
                    observation_tensor[1].copy_(temp_tensor)
        
        # Channel 2: Ground class_id - 동일한 최적화 적용
        if 'ground_class_id' in state_dict and state_dict['ground_class_id'] is not None:
            data = state_dict['ground_class_id']
            if isinstance(data, torch.Tensor):
                if data.device != device:
                    observation_tensor[2].copy_(data.to(device, non_blocking=True))
                else:
                    observation_tensor[2].copy_(data)
            else:
                temp_tensor = torch.from_numpy(data)
                if device != 'cpu':
                    observation_tensor[2].copy_(temp_tensor.to(device, non_blocking=True))
                else:
                    observation_tensor[2].copy_(temp_tensor)
    
    # Channel 3: Near destination direction - 직접 할당으로 최적화
    near_dir = state_dict.get('near_destination_direction')
    if near_dir:
        observation_tensor[3, 0, 0] = near_dir.get('x', 0.0)
        observation_tensor[3, 0, 1] = near_dir.get('y', 0.0)
    
    # Channel 4: Final destination direction - 직접 할당으로 최적화
    final_dir = state_dict.get('final_destination_direction')
    if final_dir:
        observation_tensor[4, 0, 0] = final_dir.get('x', 0.0)
        observation_tensor[4, 0, 1] = final_dir.get('y', 0.0)
    
    # Channel 5: Simulated ego position - 최적화된 처리
    # ego_pos = state_dict.get('simulated_ego_pos')
    # if ego_pos is not None and isinstance(ego_pos, tuple) and len(ego_pos) == 2:
    #     if isinstance(ego_pos[0], int):  # (row, col) 형태
    #         row, col = ego_pos
    #         center = observation_shape[1] // 2  # 49
    #         ego_x = float(row - center)
    #         ego_y = float(center - col)
    #     else:  # (x, y) 형태
    #         ego_x, ego_y = ego_pos
        
    observation_tensor[5, 0, 0] = 49
    observation_tensor[5, 0, 1] = 49
    
    return observation_tensor


def build_observation_batch(state_dicts: List[Dict], observation_shape: Tuple[int, int, int], device: str = 'cpu', use_obstacle_channels: bool = None) -> torch.Tensor:
    """
    여러 state_dict를 배치 단위로 처리하는 최적화된 함수
    
    Args:
        state_dicts: Episode state dictionary 리스트
        observation_shape: (channels, height, width) 형태의 관측값 shape
        device: 텐서를 생성할 디바이스
        
    Returns:
        배치 observation tensor (shape: [batch_size, channels, height, width])
    """
    batch_size = len(state_dicts)
    batch_tensor = torch.zeros((batch_size,) + observation_shape, device=device, pin_memory=(device == 'cpu'))
    
    # 배치의 각 항목을 병렬로 처리 (메모리 효율성)
    for i, state_dict in enumerate(state_dicts):
        batch_tensor[i] = build_observation_tensor(state_dict, observation_shape, device, use_obstacle_channels)
    
    return batch_tensor




# def visualize_training_progress(network, candidate_episodes, step, config, device, loss_history=None, iteration_num=1):
def visualize_training_progress(network, episodes, config, device, loss_history=None, iteration_num=1):
    """학습 중 모델의 예측 성능과 Loss를 시각화
    
    1. replay_buffer에서 Episode를 config 에 선언된 visualization num episodes 개수만큼 샘플링
    2. training 으로 학습된 network 를 사용하여 MCTS enrichment 수행(enrich_episodes_with_mcts())
    3. 시각화 수행 (Predicted Path + Loss)
    
    """
    print(f"\n  📊 Iteration {iteration_num} 학습 진행 상황 시각화...")
    
    # 시각화할 Episode 샘플링 (config에서 설정)
    # import copy
    # episodes = candidate_episodes[:min(config.visualization_num_episodes, len(candidate_episodes))]
    # episodes = copy.deepcopy(replay_buffer.sample_episodes(min(config.visualization_num_episodes, len(replay_buffer))))
    
    if not episodes:
        print("    시각화할 Episode가 없습니다.")
        return
    
    # Self-play 매니저 생성 (MCTS enrichment를 위해)
    
    # Mock environment와 provider (시각화용이므로 실제 환경 불필요)
    class MockObservationProvider:
        def get_observation(self):
            return None
    
    # 1. Network state_dict 추출 (pickle 가능)
    # network_state_dict = network.state_dict()
    
    # 각 Episode에 대해 시각화
    for ep_idx, episode in enumerate(episodes):
        # if len(episode) < config.visualization_min_episode_length:  # 너무 짧은 Episode는 스킵
        #     continue
            
        # 전체 Episode 시각화
        visualization_steps = len(episode)
        # Episode 전체를 복사 (수정하지 않기 위해)
        # from models.MuZero.replay_buffer import Episode
        # full_episode = Episode()
        # for i in range(visualization_steps):
        #     full_episode.states.append(episode.states[i])
        #     full_episode.actions.append(0)  # 임시 action
        #     full_episode.rewards.append(0.0)  # 임시 reward
        #     full_episode.values.append(0.0)  # 임시 value
        #     full_episode.pi.append(None)  # 임시 π
        #     full_episode.legal_mask.append(None)  # 임시 legal_mask (MCTS에서 계산됨)
        #     full_episode.dones.append(False)  # 임시 done

        # # expert_ego_pos는 states보다 길 수 있으므로 별도로 모든 위치 복사
        # full_episode.expert_ego_pos = episode.expert_ego_pos.copy()

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

        # if old_predicted_rows:
            # print(f"    📍 Old Predicted path: start({old_predicted_rows[0]}, {old_predicted_cols[0]}) -> end({old_predicted_rows[-1]}, {old_predicted_cols[-1]})")

        # 현재 네트워크로 MCTS enrichment 수행
        # print(f"    Episode {ep_idx+1}: {visualization_steps} 스텝 처리 중...")
        # enriched_episodes = self_play_manager.enrich_episodes_with_mcts([full_episode])
        # enriched_episode = enriched_episodes[0]
        # args = (ep_idx, network_state_dict, config, device, episode)
        # enriched_episode = enrich_episode_with_mcts_parallel_worker(args)
        
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
        title = f'Iter {iteration_num} - Episode {ep_idx+1} (All {visualization_steps} steps)'
        if hasattr(episode, 'source_file') and episode.source_file:
            title += f'\nSource: {episode.source_file}'
            if hasattr(episode, 'start_frame') and hasattr(episode, 'end_frame'):
                title += f' (Frames: {episode.start_frame}-{episode.end_frame})'
        ax.set_title(title)
        
        # 첫 프레임의 환경 정보 시각화 (배경으로)
        # Shift 적용으로 각 스텝의 observation이 다르므로, 원본 좌표계인 첫 스텝 사용
        if len(episode.states) > 0:
            last_state = episode.states[0]  # 첫 프레임 사용 -> TODO: 디버깅 목적이면, 마지막인 -1 로 설정
            
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
        expert_rows = [pos[0] for pos in episode.expert_ego_pos]
        expert_cols = [pos[1] for pos in episode.expert_ego_pos]
        
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
        unique_actions = set(episode.actions)
        # print(f"🔍 Episode {ep_idx+1} - States: {len(episode.states)}, Expert pos: {len(episode.expert_ego_pos)}")
        # print(f"🔍 Actions: {unique_actions}, Total: {len(episode.actions)}")

        # Action별 Target Reward 평균 계산
        action_rewards = {}
        for action, reward in zip(episode.actions, episode.rewards):
            if action not in action_rewards:
                action_rewards[action] = []
            action_rewards[action].append(reward)

        action_avg_str = ", ".join([f"{action}:{np.mean(rewards):.3f}" for action, rewards in sorted(action_rewards.items())])
        # print(f"🎯 Action별 Target Reward: {action_avg_str}")
        
        # Episode states에서 직접 simulated_ego_pos 추출
        fallback_count = 0
        for i in range(len(episode.states)):
            state = episode.states[i]
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
            print(f"⚠️ {fallback_count}/{len(episode.states)} 스텝에서 simulated_ego_pos 누락으로 fallback 사용")
        
        # print(f"🔍 simulated_ego_pos: start({sim_rows[0]}, {sim_cols[0]}) -> end({sim_rows[-1]}, {sim_cols[-1]})")

        # New Predicted Path (빨간색 그라데이션: 학습 후 경로) (simulated_ego_pos)
        if len(set(zip(sim_rows, sim_cols))) > 1:  # 서로 다른 위치가 2개 이상 있을 때만
            # New Predicted Path 그라데이션 그리기 (simulated_ego_pos)
            for i in range(len(sim_rows) - 1):
                # 진행도에 따른 알파값 (0.3 -> 1.0)
                alpha = 0.3 + 0.7 * (i / (len(sim_rows) - 1))
                # 진행도에 따른 색상 강도 (연한 빨간색 -> 진한 빨간색)
                color_intensity = 0.4 + 0.6 * (i / (len(sim_rows) - 1))
                color = (color_intensity, 0, 0)

                # 첫 번째 선분에만 label 추가 (범례용)
                label = 'simulated_ego_pos' if i == 0 else None
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
            ax.scatter(sim_cols[0], sim_rows[0], c='red', s=100, marker='x', label='simulated_ego_pos', zorder=5)
        
        # 시작점과 끝점 표시
        ax.scatter(expert_cols[0], expert_rows[0], c='green', s=100, marker='o', label='Start', zorder=6)
        if len(expert_cols) > 1:
            ax.scatter(expert_cols[-1], expert_rows[-1], c='orange', s=100, marker='s', label='End', zorder=6)
        
        # Target Goal 위치 표시 (reward 계산에 사용되는 Episode 끝 위치)
        if hasattr(episode, 'expert_ego_pos') and len(episode.expert_ego_pos) > 0:
            goal_pos = episode.expert_ego_pos[-1]  # Episode의 마지막 expert 위치
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
            # loss_text = f"Latest Loss (step {step}):\n"
            loss_text = f"Total: {latest['total_loss']:.4f}\n"
            loss_text += f"Value: {latest['value_loss']:.4f}\n"
            loss_text += f"Policy: {latest['policy_loss']:.4f}\n"
            loss_text += f"Reward: {latest['reward_loss']:.4f}"
            ax.text(0.02, 0.98, loss_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # Episode 통계 및 Target Values 정보 추가
        stats_text = f"Episode {ep_idx+1} Stats:\n"
        stats_text += f"Steps: {len(episode.states)}\n"
        
        
        # Target Values 정보
        if hasattr(episode, 'target_values') and episode.target_values:
            stats_text += f"\n[Target Values]\n"
            stats_text += f"Avg: {np.mean(episode.target_values):.3f}\n"
            stats_text += f"Min/Max: {np.min(episode.target_values):.3f}/{np.max(episode.target_values):.3f}\n"
        
        # MCTS Predictions (for comparison)
        if episode.values:
            stats_text += f"\n[MCTS Values]\n"
            stats_text += f"Avg: {np.mean(episode.values):.3f}\n"
        
        # Target Policy 정보 (가장 많이 선택된 action)
        if episode.pi and any(p is not None for p in episode.pi):
            from models.MuZero.action import Action
            # 각 action의 평균 확률 계산
            num_actions = len(episode.pi[0]) if episode.pi[0] is not None else 9
            avg_pi = np.zeros(num_actions)
            count = 0
            for pi in episode.pi:
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
        if episode.states and len(episode.states) > 0:
            first_state = episode.states[0]
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
        if hasattr(episode, 'expert_ego_pos') and len(episode.expert_ego_pos) > 0:
            goal_pos = episode.expert_ego_pos[-1]
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
        if episode.actions and episode.target_rewards:
            from models.MuZero.action import Action
            from collections import defaultdict
            
            # Action별 reward 수집
            action_rewards = defaultdict(list)
            for action_idx, reward in zip(episode.actions, episode.target_rewards):
                action_rewards[action_idx].append(reward)
            
            # 텍스트 생성 시작
            action_stats_text = "[Action-Reward Details]\n"
            
            # 1. Step별 Action-Reward (처음 30스텝까지)
            action_stats_text += "First 30 Steps:\n"
            max_display_steps = min(30, len(episode.actions))
            for ep_step in range(max_display_steps):
                if ep_step < len(episode.actions) and ep_step < len(episode.target_rewards):
                    action_obj = Action.from_index(episode.actions[ep_step])
                    reward = episode.target_rewards[ep_step]

                    # 해당 스텝의 simulated_ego_pos 가져오기
                    pos_info = ""
                    if (ep_step < len(episode.states) and
                        isinstance(episode.states[ep_step], dict) and
                        'simulated_ego_pos' in episode.states[ep_step]):
                        pos = episode.states[ep_step]['simulated_ego_pos']
                        pos_info = f" @{pos}"

                    # Target Value + Predicted Value (MCTS root value) 함께 표시
                    value_info = ""
                    if hasattr(episode, 'target_values') and ep_step < len(episode.target_values):
                        target_value = episode.target_values[ep_step]
                        # Predicted Value (MCTS root value) 추가
                        pred_v_info = ""
                        if ep_step < len(episode.values) and episode.values[ep_step] is not None:
                            pred_v_info = f" pred: {episode.values[ep_step]:.3f})"
                        value_info = f" | V: (tar:{target_value:.3f}/{pred_v_info}"
                    elif ep_step < len(episode.values):
                        value = episode.values[ep_step]
                        value_info = f" | V: (tar:{value:.3f})"

                    # Predicted Reward 추가
                    pred_r_info = ""
                    if hasattr(episode, 'rewards') and ep_step < len(episode.rewards) and episode.rewards[ep_step] != 0.0:
                        pred_r_info = f" pred: {episode.rewards[ep_step]:.3f}"
                    action_stats_text += f"  {ep_step+1}: {action_obj.position.value}{pos_info} → R: (tar:{reward:.3f}/{pred_r_info}{value_info})\n"
            if len(episode.actions) > max_display_steps:
                action_stats_text += f"  ... (+{len(episode.actions) - max_display_steps} more)\n"
            
            # 2. Action별 평균 reward
            action_stats_text += "\nAction Averages:\n"
            for action_idx in sorted(action_rewards.keys()):
                action_obj = Action.from_index(action_idx)
                target_rewards = action_rewards[action_idx]
                avg_reward = np.mean(target_rewards)
                count = len(target_rewards)
                
                # Action의 이동 벡터 정보 추가
                dx, dy = action_obj.get_movement_vector()
                vec_info = f"({dx:+d},{dy:+d})"
                
                action_stats_text += f"{action_obj.position.value} {vec_info}: {avg_reward:.3f} (×{count})\n"
            
            # 3. 전체 Target Rewards 통계
            action_stats_text += f"\nTotal Rewards:\n"
            action_stats_text += f"Sum: {sum(episode.target_rewards):.2f}\n"
            action_stats_text += f"Avg: {np.mean(episode.target_rewards):.3f}\n"
            action_stats_text += f"Range: {np.min(episode.target_rewards):.3f}~{np.max(episode.target_rewards):.3f}\n"
            action_stats_text += f"Non-zero: {np.count_nonzero(episode.target_rewards)}/{len(episode.target_rewards)}"
            
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
            
            ax_loss.set_xlabel('training_step', fontsize=12)
            ax_loss.set_ylabel('Loss', fontsize=12)
            ax_loss.set_title(f'Training Loss History', fontsize=14)
            ax_loss.legend(loc='upper right')
            ax_loss.grid(True, alpha=0.3)
            
        # 저장
        if ep_idx < config.visualization_num_episodes:
            save_dir = config.results_path / "visualizations"
            save_dir.mkdir(exist_ok=True)
            save_path = save_dir / f"iter{iteration_num}_episode_{ep_idx+1}.png"
            plt.savefig(save_path, dpi=config.visualization_dpi, bbox_inches='tight')
            plt.close()
            # print(f"    💾 시각화 저장: {save_path}")

    # 시각화 완료 후 GPU 메모리 정리
    # enrich_episodes_with_mcts()에서 생성한 GPU Tensor들 해제
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # print(f"    🧹 시각화 메모리 정리 완료")


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


def reward_sum_of_valid(P, Q, grid_size = 5, reward_max=0.5, reward_scale_factor=0.1, gt_threshold=0.2, count_threshold=0.3):
    """
    P: GT 분포 (Raw Count 권장)
    Q: Robot 분포 (반드시 Raw Count여야 함! 예: [2, 0, 1...])
    
    [기능 1] Density Gate: 로봇이 예측한 픽셀 수가 너무 적으면(예: 30% 미만) 0점 처리 (찍기 방지)
    [기능 2] Valid Sum: GT가 인정한 땅(20% 이상)에 대한 로봇의 점유율 합산
    [기능 3] Saturation: 합산 점유율이 일정 수준(50%)을 넘으면 만점 처리 (자유도 부여)
    
    Args:
        grid_total: 전체 그리드 칸 수 (예: 5x5 = 25)
        max_reward: 최종 만점 점수 (예: 0.5)
        gt_threshold: GT 유효성 기준 (예: 0.2 -> GT 20% 미만은 노이즈 취급)
        count_threshold: 로봇 최소 개수 기준 (예: 0.3 -> 전체 그리드의 30%는 채워야 함)
        saturation_point: 점수 포화 기준 (예: 0.5 -> 유효 확률 합이 50%만 넘으면 만점)
    """
    P = np.array(P)
    Q = np.array(Q)
    
    # 정규화 (합이 1이 되도록 안전장치)
    sum_p = np.sum(P)
    sum_q = np.sum(Q)
    
    if sum_p > 0: P = P / sum_p
    if sum_q > 0: Q = Q / sum_q

    # 입력 값의 개수가 5x5 Grid 의 총 개수 25개의 count_threshold 비율보다 작으면, 입력값 조건 미충족으로 0 리턴
    if sum_q < grid_size * grid_size * count_threshold:
        return -reward_max*reward_scale_factor

    # 1. GT에서 유효한(Valid) 클래스 찾기 (예: 20% 이상인 땅들)
    # 경계선에서는 [True, True, False...] 처럼 두 개가 True가 됨
    valid_mask = P >= gt_threshold
    if not np.any(valid_mask):
        print("GT 지면 Class 에 정상 데이터가 없어요")

    # 2. 로봇이 '유효한 땅'에 둔 확률들을 전부 합침 (Sum)
    # 로봇이 인도 0.5, 횡단보도 0.5를 줬다면 -> 합은 1.0
    valid_prob_sum = np.sum(Q[valid_mask])
    
    # 3. 클리핑 (최대 1.0) 및 스케일링
    # 합이 1.0을 넘을 순 없겠지만(확률이니까), 안전하게 min 처리
    score = min(1.0, valid_prob_sum)
    score = ((reward_max if score > reward_max else score) - reward_max) * reward_scale_factor  # TODO: 정리필요. 범위를 -0.5 ~ 0 으로 만들기 위함 (0~0.5 만든 후, -0.5 ~ 0 만들고 0.1 곱함. 수식 비효율적. 수정필요)
    
    return score

def max_relative_scailing(P, Q, reward_scale=0.05):
    """
    P: GT (Ground Truth) 분포 [리스트]
    Q: Robot 행동 분포 [리스트]
    reward_scale: 리워드 최대값 (기본 0.05)
    정답지 1등에 대해서 입력값의 1등이 얼마나 비중있는지를 점수화
    """
    P = np.array(P, dtype=np.float64)
    Q = np.array(Q, dtype=np.float64)

    # 정규화 (합이 1이 되도록 안전장치)
    P = P / np.sum(P)
    Q = Q / np.sum(Q)

    # --- 2. Max-Relative Scaling 방식 (제안 방식) ---
    # 로봇이 선택한 행동(Class) 찾기
    # Q에서 가장 확률이 높은 인덱스를 로봇이 밟은 땅으로 간주
    robot_choice_idx = np.argmax(Q)

    # GT에서의 해당 클래스 확률 / GT의 최대 확률
    p_at_robot_choice = P[robot_choice_idx]
    p_max_gt = np.max(P)

    if p_max_gt == 0:
        score_ratio = 0
    else:
        score_ratio = p_at_robot_choice / p_max_gt

    reward_max_rel = reward_scale * score_ratio

    return {
        "Reward_MaxRel": reward_max_rel,
        "Robot_Choice": robot_choice_idx
    }

def js_divergence(p, q):
    """
    Jensen-Shannon Divergence 계산
    
    Args:
        p: 확률 분포 P (list or numpy array)
        q: 확률 분포 Q (list or numpy array)
    
    Returns:
        float: JS Divergence 값
    """
    # numpy array로 변환
    p = np.array(p, dtype=np.float64)
    q = np.array(q, dtype=np.float64)
    
    # 정규화 (합이 1이 되도록)
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # 평균 분포 M
    m = 0.5 * (p + q)
    
    # KL divergence 계산 (0*log(0) = 0 처리)
    def kl_div(x, y):
        # x > 0인 경우만 계산
        mask = x > 0
        result = np.sum(x[mask] * np.log2(x[mask] / y[mask]))
        return result
    
    # JS divergence = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    js = 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)
    
    return js


# # 사용 예시
# from js_divergence import js_divergence

# p = [0.5, 0.3, 0.2]
# q = [0.4, 0.4, 0.2]
# result = js_divergence(p, q)
# print(f"JS Divergence: {result}")
# if __name__ == "__main__":
#     # 예시 1: 같은 분포
#     p1 = [0.5, 0.3, 0.2]
#     q1 = [0.5, 0.3, 0.2]
#     print(f"같은 분포: JS = {js_divergence(p1, q1):.6f}")
    
#     # 예시 2: 다른 분포
#     p2 = [0.5, 0.3, 0.2]
#     q2 = [0.4, 0.4, 0.2]
#     print(f"다른 분포: JS = {js_divergence(p2, q2):.6f}")
    
#     # 예시 3: 정규화 필요한 경우
#     p3 = [1, 2, 3]
#     q3 = [3, 2, 1]
#     print(f"정규화 필요: JS = {js_divergence(p3, q3):.6f}")
    
#     # 예시 4: 완전히 다른 분포
#     p4 = [1, 0, 0]
#     q4 = [0, 0, 1]
#     print(f"완전히 다른 분포: JS = {js_divergence(p4, q4):.6f}")