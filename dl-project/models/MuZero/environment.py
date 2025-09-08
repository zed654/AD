"""
자율주행 환경 클래스
상태 전이, 보상, 종료 조건 등을 관리
"""
import torch
import numpy as np
from typing import Tuple, Dict, Optional
from abc import ABC, abstractmethod
from .action import Action
from .observation_provider import ObservationProvider


class BaseEnvironment(ABC):
    """환경 기본 클래스"""
    
    @abstractmethod
    def reset(self) -> torch.Tensor:
        """환경을 초기화하고 초기 관측값 반환"""
        pass
    
    @abstractmethod
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        """액션을 수행하고 다음 상태, 보상, 종료 여부, 정보 반환"""
        pass
    
    @abstractmethod
    def is_terminal(self) -> bool:
        """현재 상태가 종료 상태인지 확인"""
        pass


class DrivingEnvironment(BaseEnvironment):
    """자율주행 환경"""
    
    def __init__(self, config=None, observation_provider: Optional[ObservationProvider] = None):
        # config가 없으면 기본값 사용
        if config is None:
            self.max_steps = 100
            self.collision_threshold = 0.5
            self.goal_threshold = 1.0
            self.observation_shape = (1, 1, 4)
        else:
            # config에서 환경 설정 가져오기
            self.max_steps = config.max_episode_step_length
            self.collision_threshold = config.collision_threshold
            self.goal_threshold = config.goal_threshold
            self.observation_shape = config.observation_shape
            self.config = config
        
        # ObservationProvider 설정
        if observation_provider is not None:
            self.observation_provider = observation_provider
        else:
            # 기본값: Mock provider 사용
            from .observation_provider import MockObservationProvider
            self.observation_provider = MockObservationProvider(config or self._create_default_config())
        
        # 상태 변수들
        self.current_step = 0
        self.ego_position = None
        self.goal_position = None
        self.obstacles = []
        self.game_done = False  # 게임 종료 여부
        self.done_reason = None
        self.last_action = None
        self.simulated_ego_pos = (0.0, 0.0)  # 시뮬레이션 ego 위치 (x, y) - 그리드 중심 기준 상대 좌표
        
    def reset(self) -> torch.Tensor:
        """환경 초기화"""
        self.current_step = 0
        self.game_done = False
        self.done_reason = None
        self.simulated_ego_pos = (0.0, 0.0)  # 원점에서 시작
        
        # ObservationProvider로부터 초기 관측값 가져오기
        observation_dict = self.observation_provider.reset()
        
        # GT 데이터에서 ego 위치 초기화 (충돌 체크용)
        ego_info = None
        objects = []
        ldm_points = []
        trajectory = []
        near_destination_direction = None
        final_destination_direction = None
        
        if hasattr(self.observation_provider, 'get_current_frame_data'):
            frame_data = self.observation_provider.get_current_frame_data()
            if frame_data and hasattr(self.observation_provider, 'parser'):
                ego_info = self.observation_provider.parser.parse_ego_info(frame_data)
                objects = self.observation_provider.parser.parse_objects(frame_data)
                ldm_points = self.observation_provider.parser.parse_ldm_points(frame_data)
                trajectory = self.observation_provider.parser.parse_trajectory(frame_data)
                near_destination_direction, final_destination_direction = self.observation_provider.parser.get_direction_vectors(frame_data)
                
                if ego_info:
                    self.ego_position = np.array([ego_info.x, ego_info.y])
                    self.ego_yaw = ego_info.yaw
                    
                    # 목표 위치 설정 (trajectory의 5번째 포인트)
                    if trajectory and len(trajectory) > 5:
                        self.goal_position = np.array([trajectory[5].x, trajectory[5].y])
        
        # 임시: ego_position 초기화 (추후 observation에서 추출)
        if self.ego_position is None:
            self.ego_position = np.array([0.0, 0.0])
        if self.goal_position is None:
            self.goal_position = np.array([10.0, 10.0])
        
        # dict에서 grid를 torch tensor로 변환
        observation_np = observation_dict['grid']
        observation_tensor = torch.from_numpy(observation_np).float()
        
        # 차원 추가 (배치 차원)
        if len(observation_tensor.shape) == len(self.observation_shape):
            observation_tensor = observation_tensor.unsqueeze(0)
        
        # 전체 state를 dictionary로 반환
        state = {
            'obstacles': observation_tensor,  # grid를 obstacles로 이름 변경
            'ldm': ldm_points,
            'objects': objects,
            'near_destination_direction': near_destination_direction,
            'final_destination_direction': final_destination_direction,
            'ego_pos': (self.ego_position[0], self.ego_position[1]) if self.ego_position is not None else None,
            'simulated_ego_pos': self.simulated_ego_pos  # (x, y) 형태의 상대 좌표
        }
        
        return state
    
    def step(self, action_index: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        """액션 수행"""
        self.current_step += 1
        
        # 액션 저장
        self.last_action = action_index
        
        # Action 디코딩
        action = Action.from_index(action_index)
        
        # ObservationProvider로부터 다음 상태 가져오기
        observation_dict = self.observation_provider.get_observation()
        
        # GT 데이터 추가 정보 가져오기
        ego_info = None
        objects = []
        ldm_points = []
        trajectory = []
        near_destination_direction = None
        final_destination_direction = None
        
        if hasattr(self.observation_provider, 'get_current_frame_data'):
            frame_data = self.observation_provider.get_current_frame_data()
            if frame_data and hasattr(self.observation_provider, 'parser'):
                ego_info = self.observation_provider.parser.parse_ego_info(frame_data)
                objects = self.observation_provider.parser.parse_objects(frame_data)
                ldm_points = self.observation_provider.parser.parse_ldm_points(frame_data)
                trajectory = self.observation_provider.parser.parse_trajectory(frame_data)
                near_destination_direction, final_destination_direction = self.observation_provider.parser.get_direction_vectors(frame_data)
        
        # dict에서 grid를 torch tensor로 변환
        observation_np = observation_dict['grid']
        observation_tensor = torch.from_numpy(observation_np).float()
        if len(observation_tensor.shape) == len(self.observation_shape):
            observation_tensor = observation_tensor.unsqueeze(0)
        
        # 상태 업데이트
        self._update_state(action)
        
        # 게임 로직 처리
        reward = self._calculate_reward(action)
        self.game_done = self._check_termination()
        
        # GT 데이터 종료 확인
        if hasattr(self.observation_provider, 'is_data_ended') and self.observation_provider.is_data_ended():
            self.game_done = True
            self.done_reason = "data_ended"
        
        # 추가 정보
        info = {
            'done_reason': self.done_reason,
            'ego_position': self.ego_position.copy() if self.ego_position is not None else None,
            'steps': self.current_step
        }
        
        # GT 데이터 정보 추가
        if hasattr(self.observation_provider, 'current_frame_idx'):
            info['frame_idx'] = self.observation_provider.current_frame_idx
        
        # 전체 state를 dictionary로 반환
        state = {
            'obstacles': observation_tensor,  # grid를 obstacles로 이름 변경
            'ldm': ldm_points,
            'objects': objects,
            'near_destination_direction': near_destination_direction,
            'final_destination_direction': final_destination_direction,
            'ego_pos': (self.ego_position[0], self.ego_position[1]) if self.ego_position is not None else None,
            'simulated_ego_pos': self.simulated_ego_pos  # (x, y) 형태의 상대 좌표
        }
        
        # state, reward, 게임종료여부, 추가정보 반환
        return state, reward, self.game_done, info
    
    def is_terminal(self) -> bool:
        """현재 게임이 종료 상태인지 확인
        
        Returns:
            bool: 게임이 종료되었으면 True, 계속 진행중이면 False
        """
        return self.game_done
    
    def _check_termination(self) -> bool:
        """종료 조건 확인"""
        # 1. 최대 스텝 초과
        if self.current_step >= self.max_steps:
            self.done_reason = "max_steps"
            return True
        
        # 2. 목적지 도착
        distance_to_goal = np.linalg.norm(self.ego_position - self.goal_position)
        if distance_to_goal < self.goal_threshold:
            self.done_reason = "goal_reached"
            return True
        
        # 3. 충돌 검사 (간단한 예시)
        if self._check_collision():
            self.done_reason = "collision"
            return True
        
        # 4. 도로 이탈 (간단한 예시)
        if self._check_off_road():
            self.done_reason = "off_road"
            return True
        
        return False
    
    def _check_collision(self) -> bool:
        """충돌 검사"""
        # ego_position이 없으면 충돌 없음
        if self.ego_position is None:
            return False
        
        # 충돌 판정 거리 (미터 단위, 약 1.5m)
        collision_distance = 1.5
        
        # GT 데이터가 있는 경우
        if hasattr(self.observation_provider, 'get_current_frame_data'):
            frame_data = self.observation_provider.get_current_frame_data()
            if frame_data and hasattr(self.observation_provider, 'parser'):
                objects = self.observation_provider.parser.parse_objects(frame_data)
                ego_info = self.observation_provider.parser.parse_ego_info(frame_data)
                
                if ego_info:
                    # 각 객체와의 거리 확인
                    for obj in objects:
                        distance = np.sqrt(
                            (obj.global_x - ego_info.x)**2 + 
                            (obj.global_y - ego_info.y)**2
                        )
                        if distance < collision_distance:
                            return True
        
        # ROS2 데이터가 있는 경우
        elif hasattr(self.observation_provider, 'latest_obstacles') and self.observation_provider.latest_obstacles:
            # ROS2 메시지 형식에 따라 처리 (TODO: 실제 메시지 형식에 맞게 수정)
            pass
        
        # Mock provider인 경우 간단한 충돌 체크
        else:
            # 예시: 특정 위치에 장애물이 있다고 가정
            obstacle_positions = [
                np.array([5.0, 5.0]),
                np.array([8.0, 3.0]),
                np.array([3.0, 8.0])
            ]
            
            for obs_pos in obstacle_positions:
                distance = np.linalg.norm(self.ego_position - obs_pos)
                if distance < collision_distance:
                    return True
        
        return False
    
    def _check_off_road(self) -> bool:
        """도로 이탈 검사"""
        # TODO: 실제 도로 이탈 검사 로직
        # 예: 도로 경계를 벗어났는지 확인
        return False
    
    def _update_state(self, action: Action):
        """액션에 따른 상태 업데이트"""
        # 액션의 이동 벡터 가져오기 (그리드 단위)
        dx, dy = action.get_movement_vector()
        
        # Self-play용 ego 위치 업데이트 (x, y 좌표)
        # dx는 x축(상하), dy는 y축(좌우)
        # 로봇좌표계임 (x 축 전방 양수; row / y 축 우측 양수; col)
        current_x, current_y = self.simulated_ego_pos
        new_x = current_x + dx  # x 변화 (전방/후방)
        new_y = current_y + dy  # y 변화 (좌/우)
        
        # 그리드 범위 체크 (중심 기준 ±49)
        grid_half = self.config.observation_shape[1] // 2  # 49
        if -grid_half <= new_x <= grid_half and -grid_half <= new_y <= grid_half:
            self.simulated_ego_pos = (new_x, new_y)
        
        # GT 데이터가 있는 경우 프레임 데이터에서 업데이트
        if hasattr(self.observation_provider, 'get_current_frame_data'):
            frame_data = self.observation_provider.get_current_frame_data()
            if frame_data and hasattr(self.observation_provider, 'parser'):
                ego_info = self.observation_provider.parser.parse_ego_info(frame_data)
                if ego_info:
                    self.ego_position = np.array([ego_info.x, ego_info.y])
                    self.ego_yaw = ego_info.yaw
                    return
        
        # GT 데이터가 없는 경우 간단한 시뮬레이션
        # 실제 이동량 계산 (0.2m 셀 크기 * 그리드 이동량)
        cell_size = 0.2
        actual_dx = dx * cell_size
        actual_dy = dy * cell_size
        
        # 위치 업데이트
        if self.ego_position is not None:
            self.ego_position[0] += actual_dx
            self.ego_position[1] += actual_dy
    
    def _calculate_reward(self, action: Action) -> float:
        """보상 계산"""
        reward = 0.0
        
        # 기본 생존 보상 (작은 음수)
        reward -= 0.01
        
        # 이동에 대한 보상 (POS0이 아닌 경우)
        if action.position != Action.Position.POS0:
            reward += 0.1
        
        # 종료 상태에 따른 보상
        if self.done_reason == "goal_reached":
            reward += 100.0
        elif self.done_reason == "collision":
            reward = -10.0  # 충돌 시 큰 페널티
        elif self.done_reason == "off_road":
            reward -= 5.0
        
        # GT 데이터가 있는 경우 추가 보상
        if hasattr(self.observation_provider, 'get_current_frame_data'):
            frame_data = self.observation_provider.get_current_frame_data()
            if frame_data and hasattr(self.observation_provider, 'parser'):
                ego_info = self.observation_provider.parser.parse_ego_info(frame_data)
                objects = self.observation_provider.parser.parse_objects(frame_data)
                
                if ego_info:
                    # 속도 보상
                    if ego_info.velocity > 0.5:
                        reward += min(ego_info.velocity * 0.1, 0.5)
                    
                    # 너무 가까운 장애물에 대한 페널티
                    min_distance = float('inf')
                    for obj in objects:
                        distance = np.sqrt(
                            (obj.global_x - ego_info.x)**2 + 
                            (obj.global_y - ego_info.y)**2
                        )
                        min_distance = min(min_distance, distance)
                    
                    if min_distance < 3.0:  # 3m 이내
                        reward -= (3.0 - min_distance) * 0.5
        
        return reward
    
    def _get_observation(self) -> torch.Tensor:
        """현재 상태의 관측값 생성 (레거시 메서드 - ObservationProvider 사용 권장)"""
        # ObservationProvider를 사용하는 경우
        if hasattr(self, 'observation_provider'):
            observation_dict = self.observation_provider.get_observation()
            observation_np = observation_dict['grid']
            observation = torch.from_numpy(observation_np).float()
            if len(observation.shape) == len(self.observation_shape):
                observation = observation.unsqueeze(0)
            return observation
        
        # 기존 방식 (하위 호환성)
        return torch.rand(1, *self.observation_shape)
    
    def _create_default_config(self):
        """기본 설정 객체 생성"""
        class DefaultConfig:
            observation_shape = (10, 12)
            env_max_steps = 100
            collision_threshold = 0.5
            goal_threshold = 1.0
        return DefaultConfig()
    
    def close(self):
        """환경 종료 시 리소스 정리"""
        if hasattr(self, 'observation_provider'):
            self.observation_provider.close() 