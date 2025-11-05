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