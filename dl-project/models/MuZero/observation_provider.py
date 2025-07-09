"""
MuZero Observation Provider 인터페이스 및 구현체
학습 시에는 GT 데이터에서, 실행 시에는 ROS2 Topic에서 관측값을 가져옴
"""

from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import json


class ObservationProvider(ABC):
    """관측값 제공자 인터페이스"""
    
    @abstractmethod
    def reset(self) -> np.ndarray:
        """환경을 리셋하고 초기 관측값을 반환"""
        pass
    
    @abstractmethod
    def get_observation(self, action: Optional[int] = None) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        현재 상태의 관측값을 반환
        
        Args:
            action: 이전에 수행한 액션 (첫 번째 호출 시 None)
            
        Returns:
            observation: 현재 관측값
            reward: 보상
            done: 에피소드 종료 여부
            info: 추가 정보
        """
        pass
    
    @abstractmethod
    def close(self):
        """리소스 정리"""
        pass


class GTObservationProvider(ObservationProvider):
    """Ground Truth 데이터에서 관측값을 제공하는 클래스 (학습용)"""
    
    def __init__(self, config, data_path: Optional[Path] = None):
        """
        Args:
            config: MuZero 설정
            data_path: GT 데이터 파일 경로 (JSONL 형식)
        """
        self.config = config
        self.data_path = data_path or Path("data/db3_parser/results/matched_data.jsonl")
        self.observation_shape = config.observation_shape
        
        # GT 데이터 로드
        self.episodes = []
        self.current_episode = None
        self.current_step = 0
        
        self._load_gt_data()
        
    def _load_gt_data(self):
        """GT 데이터를 로드"""
        if not self.data_path.exists():
            print(f"경고: GT 데이터 파일이 없습니다: {self.data_path}")
            return
            
        with open(self.data_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                self.episodes.append(data)
        
        if self.episodes:
            print(f"GT 데이터 로드 완료: {len(self.episodes)}개 에피소드")
        else:
            print("경고: GT 데이터가 비어있습니다")
    
    def reset(self) -> np.ndarray:
        """새로운 에피소드 시작"""
        if not self.episodes:
            # GT 데이터가 없으면 임시로 랜덤 데이터 반환
            print("경고: GT 데이터가 없어 랜덤 관측값을 사용합니다")
            return np.random.rand(*self.observation_shape).astype(np.float32)
        
        # 랜덤하게 에피소드 선택
        import random
        self.current_episode = random.choice(self.episodes)
        self.current_step = 0
        
        # 초기 관측값 생성
        observation = self._extract_observation(self.current_episode)
        return observation
    
    def get_observation(self, action: Optional[int] = None) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """GT 데이터에서 다음 관측값 가져오기"""
        if not self.current_episode:
            # 에피소드가 없으면 리셋
            observation = self.reset()
            return observation, 0.0, False, {}
        
        # 액션 수행 후 다음 상태로 전환
        self.current_step += 1
        
        # GT trajectory 길이 확인
        gt_trajectory = self.current_episode.get("gt_trajectory", [])
        if self.current_step >= len(gt_trajectory):
            # 에피소드 종료
            observation = self._extract_observation(self.current_episode)
            return observation, 0.0, True, {"episode_ended": True}
        
        # 관측값 추출
        observation = self._extract_observation(self.current_episode)
        
        # 보상 계산 (TODO: 실제 보상 함수 구현 필요)
        reward = self._calculate_reward(action, self.current_episode, self.current_step)
        
        # 종료 조건 확인
        done = self._check_termination(self.current_episode, self.current_step)
        
        info = {
            "step": self.current_step,
            "episode_id": self.current_episode.get("timestamp", "unknown")
        }
        
        return observation, reward, done, info
    
    def _extract_observation(self, episode_data: Dict) -> np.ndarray:
        """
        에피소드 데이터에서 관측값 추출
        
        TODO: 실제 관측값 추출 로직 구현
        - obstacle tracking 정보를 observation_shape에 맞게 변환
        - 자차 상태 정보 포함
        """
        # 임시 구현: 랜덤 데이터 반환
        return np.random.rand(*self.observation_shape).astype(np.float32)
    
    def _calculate_reward(self, action: Optional[int], episode_data: Dict, step: int) -> float:
        """
        보상 계산
        
        TODO: 실제 보상 함수 구현
        - 충돌 회피
        - 목표 지점 도달
        - 부드러운 주행
        """
        # 임시 구현
        return 0.0
    
    def _check_termination(self, episode_data: Dict, step: int) -> bool:
        """에피소드 종료 조건 확인"""
        gt_trajectory = episode_data.get("gt_trajectory", [])
        return step >= len(gt_trajectory)
    
    def close(self):
        """리소스 정리"""
        pass


class ROS2ObservationProvider(ObservationProvider):
    """ROS2 Topic에서 실시간으로 관측값을 제공하는 클래스 (추론용)"""
    
    def __init__(self, config):
        """
        Args:
            config: MuZero 설정
        """
        self.config = config
        self.observation_shape = config.observation_shape
        
        # ROS2 관련 초기화
        self.ros2_initialized = False
        self._init_ros2()
        
        # 최신 관측값 저장
        self.latest_obstacles = None
        self.latest_odometry = None
        self.latest_image = None
        
    def _init_ros2(self):
        """ROS2 노드 및 구독자 초기화"""
        try:
            import rclpy
            from rclpy.node import Node
            
            # TODO: ROS2 노드 초기화
            # rclpy.init()
            # self.node = Node('muzero_observation_node')
            
            # TODO: Topic 구독자 생성
            # - /apollo/agent_0/detected_object_array
            # - /carla/ego_vehicle/odometry
            # - /carla/ego_vehicle/rgb_front/image (필요시)
            
            self.ros2_initialized = True
            print("ROS2 ObservationProvider 초기화 완료")
            
        except ImportError:
            print("경고: ROS2가 설치되지 않았습니다. 시뮬레이션 모드로 동작합니다.")
            self.ros2_initialized = False
    
    def reset(self) -> np.ndarray:
        """환경 리셋 (ROS2에서는 현재 상태 반환)"""
        if not self.ros2_initialized:
            # ROS2가 없으면 랜덤 데이터
            return np.random.rand(*self.observation_shape).astype(np.float32)
        
        # 현재 ROS2 topic에서 받은 최신 관측값 반환
        observation = self._create_observation_from_ros2()
        return observation
    
    def get_observation(self, action: Optional[int] = None) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """ROS2에서 실시간 관측값 가져오기"""
        if not self.ros2_initialized:
            # ROS2가 없으면 시뮬레이션
            observation = np.random.rand(*self.observation_shape).astype(np.float32)
            return observation, 0.0, False, {"simulated": True}
        
        # ROS2 메시지 처리 (spin_once)
        # TODO: rclpy.spin_once(self.node, timeout_sec=0.1)
        
        # 최신 데이터로 관측값 생성
        observation = self._create_observation_from_ros2()
        
        # 실시간 시스템에서는 보상과 종료 조건을 외부에서 결정
        reward = 0.0
        done = False
        info = {
            "has_obstacles": self.latest_obstacles is not None,
            "has_odometry": self.latest_odometry is not None
        }
        
        return observation, reward, done, info
    
    def _create_observation_from_ros2(self) -> np.ndarray:
        """
        ROS2 메시지에서 관측값 생성
        
        TODO: 실제 변환 로직 구현
        - detected_object_array를 그리드맵으로 변환
        - 자차 속도, 위치 정보 포함
        """
        # 임시 구현
        return np.random.rand(*self.observation_shape).astype(np.float32)
    
    def _obstacle_callback(self, msg):
        """장애물 검출 메시지 콜백"""
        self.latest_obstacles = msg
        
    def _odometry_callback(self, msg):
        """주행 정보 메시지 콜백"""
        self.latest_odometry = msg
        
    def _image_callback(self, msg):
        """카메라 이미지 메시지 콜백 (옵션)"""
        self.latest_image = msg
    
    def close(self):
        """ROS2 리소스 정리"""
        if self.ros2_initialized:
            # TODO: ROS2 노드 종료
            # self.node.destroy_node()
            # rclpy.shutdown()
            pass


class MockObservationProvider(ObservationProvider):
    """테스트용 Mock 관측값 제공자"""
    
    def __init__(self, config):
        self.config = config
        self.observation_shape = config.observation_shape
        self.step_count = 0
        
    def reset(self) -> np.ndarray:
        self.step_count = 0
        return np.random.rand(*self.observation_shape).astype(np.float32)
    
    def get_observation(self, action: Optional[int] = None) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self.step_count += 1
        observation = np.random.rand(*self.observation_shape).astype(np.float32)
        
        # 간단한 보상 함수
        reward = -0.1  # 생존 페널티
        if action is not None and action < 8:  # 전진 액션
            reward += 0.5
        
        # 100 스텝 후 종료
        done = self.step_count >= 100
        
        info = {"step": self.step_count}
        return observation, reward, done, info
    
    def close(self):
        pass