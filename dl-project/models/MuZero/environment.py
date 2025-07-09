"""
자율주행 환경 클래스
상태 전이, 보상, 종료 조건 등을 관리
"""
import torch
import numpy as np
from typing import Tuple, Dict, Optional
from abc import ABC, abstractmethod
from .action import Action
from .observation_provider import ObservationProvider, GTObservationProvider, ROS2ObservationProvider, MockObservationProvider


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
            self.max_steps = config.env_max_steps
            self.collision_threshold = config.collision_threshold
            self.goal_threshold = config.goal_threshold
            self.observation_shape = config.observation_shape
            self.config = config
        
        # ObservationProvider 설정
        if observation_provider is not None:
            self.observation_provider = observation_provider
        else:
            # 기본값: Mock provider 사용
            self.observation_provider = MockObservationProvider(config or self._create_default_config())
        
        # 상태 변수들
        self.current_step = 0
        self.ego_position = None
        self.goal_position = None
        self.obstacles = []
        self.game_done = False  # 게임 종료 여부
        self.done_reason = None
        self.last_action = None
        
    def reset(self) -> torch.Tensor:
        """환경 초기화"""
        self.current_step = 0
        self.game_done = False
        self.done_reason = None
        
        # ObservationProvider로부터 초기 관측값 가져오기
        observation_np = self.observation_provider.reset()
        
        # numpy array를 torch tensor로 변환
        observation = torch.from_numpy(observation_np).float()
        
        # 차원 추가 (배치 차원)
        if len(observation.shape) == len(self.observation_shape):
            observation = observation.unsqueeze(0)
        
        # 임시: ego_position 초기화 (추후 observation에서 추출)
        self.ego_position = np.array([0.0, 0.0])
        self.goal_position = np.array([10.0, 10.0])
        
        return observation
    
    def step(self, action_index: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        """액션 수행"""
        self.current_step += 1
        
        # 액션 저장
        self.last_action = action_index
        
        # Action 디코딩
        action = Action.from_index(action_index)
        
        # ObservationProvider로부터 다음 상태 가져오기
        observation_np, provider_reward, provider_done, provider_info = self.observation_provider.get_observation(action_index)
        
        # numpy array를 torch tensor로 변환
        observation = torch.from_numpy(observation_np).float()
        if len(observation.shape) == len(self.observation_shape):
            observation = observation.unsqueeze(0)
        
        # 상태 업데이트 (provider가 Mock인 경우를 위해)
        if isinstance(self.observation_provider, MockObservationProvider):
            self._update_state(action)
            reward = self._calculate_reward(action)
            self.game_done = self._check_termination()
        else:
            # GT나 ROS2 provider인 경우 provider의 정보 사용
            reward = provider_reward
            self.game_done = provider_done
            if provider_done and 'episode_ended' in provider_info:
                self.done_reason = "episode_ended"
        
        # 추가 정보 병합
        info = {
            'done_reason': self.done_reason,
            'ego_position': self.ego_position.copy() if self.ego_position is not None else None,
            'steps': self.current_step,
            **provider_info  # provider의 추가 정보 포함
        }
        
        # observation, reward, 게임종료여부, 추가정보 반환
        return observation, reward, self.game_done, info
    
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
        # TODO: 실제 충돌 검사 로직
        return False
    
    def _check_off_road(self) -> bool:
        """도로 이탈 검사"""
        # TODO: 실제 도로 이탈 검사 로직
        # 예: 도로 경계를 벗어났는지 확인
        return False
    
    def _update_state(self, action: Action):
        """액션에 따른 상태 업데이트"""
        # 간단한 이동 로직
        if action.stopgo == Action.StopAndGo.GO:
            # Position에 따른 이동 방향 (8방향)
            direction_map = {
                0: np.array([0, 1]),     # 북
                1: np.array([1, 1]),     # 북동
                2: np.array([1, 0]),     # 동
                3: np.array([1, -1]),    # 남동
                4: np.array([0, -1]),    # 남
                5: np.array([-1, -1]),   # 남서
                6: np.array([-1, 0]),    # 서
                7: np.array([-1, 1]),    # 북서
            }
            
            pos_idx = int(action.position.name[3:])  # POS0 -> 0
            direction = direction_map.get(pos_idx, np.array([0, 0]))
            
            # 이동 (속도는 고정값으로 가정)
            self.ego_position += direction * 0.5
    
    def _calculate_reward(self, action: Action) -> float:
        """보상 계산"""
        reward = 0.0
        
        # 목표 지점에 가까워지면 양의 보상
        distance_to_goal = np.linalg.norm(self.ego_position - self.goal_position)
        reward += -0.1 * distance_to_goal
        
        # 정지하면 작은 페널티
        if action.stopgo == Action.StopAndGo.STOP:
            reward -= 0.1
        
        # 종료 상태에 따른 보상
        if self.done_reason == "goal_reached":
            reward += 100.0
        elif self.done_reason == "collision":
            reward -= 100.0
        elif self.done_reason == "off_road":
            reward -= 50.0
        
        return reward
    
    def _get_observation(self) -> torch.Tensor:
        """현재 상태의 관측값 생성 (레거시 메서드 - ObservationProvider 사용 권장)"""
        # ObservationProvider를 사용하는 경우
        if hasattr(self, 'observation_provider'):
            observation_np, _, _, _ = self.observation_provider.get_observation(self.last_action)
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