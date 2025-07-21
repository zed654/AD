"""
MuZero Observation Provider 인터페이스 및 구현체
학습 시에는 GT 데이터에서, 실행 시에는 ROS2 Topic에서 관측값을 가져옴
"""

from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import Dict, Optional, List
from pathlib import Path
import json
import random

# 공통 데이터 파서 import
from .data_parser import GTDataParser, ObservationBuilder, EgoInfo, ObjectInfo, LDMPoint


class ObservationProvider(ABC):
    """관측값 제공자 인터페이스"""
    
    @abstractmethod
    def reset(self) -> Dict[str, np.ndarray]:
        """환경을 리셋하고 초기 관측값을 반환"""
        pass
    
    @abstractmethod
    def get_observation(self) -> Dict[str, np.ndarray]:
        """
        현재 상태의 관측값을 반환
        
        Args:
            action: 이전에 수행한 액션 (첫 번째 호출 시 None)
            
        Returns:
            observation: 현재 관측값 딕셔너리 (grid, vectors)
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
        self.data_path = data_path or config.gt_data_path
        self.observation_shape = config.observation_shape
        
        # 데이터 파서 및 observation 빌더 초기화
        try:
            self.parser = GTDataParser(self.data_path)
            self.obs_builder = ObservationBuilder(config)
            print(f"GT 데이터 로드 완료: {self.parser.get_frame_count()}개 프레임")
        except FileNotFoundError as e:
            print(f"경고: {e}")
            self.parser = None
            self.obs_builder = None
        
        # 에피소드 관리
        self.current_frame_idx = 0
        self.episode_start_idx = 0
        self.steps_in_episode = 0
        self.max_episode_length = getattr(config, 'max_episode_length', 50)
        
        # Episode 시작 위치 추적
        self.episode_start_ego = None
        
        # 이전 프레임 정보 (보상 계산용)
        self.prev_ego_info = None
        self.prev_distance_to_goal = None
        
    def reset(self) -> Dict[str, np.ndarray]:
        """새로운 에피소드 시작"""
        if not self.parser:
            # GT 데이터가 없으면 임시로 랜덤 데이터 반환
            print("경고: GT 데이터가 없어 랜덤 관측값을 사용합니다")
            return {
                'grid': np.random.rand(*self.observation_shape).astype(np.float32),
                'vectors': np.zeros(4, dtype=np.float32),
                'trajectory': np.zeros((5, 4), dtype=np.float32)
            }
        
        # 랜덤하게 시작 프레임 선택 (뒤에 충분한 프레임이 있는 곳에서)
        max_start = max(0, self.parser.get_frame_count() - self.max_episode_length - 10)
        self.episode_start_idx = random.randint(0, max_start)
        self.current_frame_idx = self.episode_start_idx
        self.steps_in_episode = 0
        
        # 초기 프레임 데이터 가져오기
        frame_data = self.parser.get_frame_data(self.current_frame_idx)
        if not frame_data:
            return {
                'grid': np.zeros(self.observation_shape, dtype=np.float32),
                'vectors': np.zeros(4, dtype=np.float32),
                'trajectory': np.zeros((0, 4), dtype=np.float32)
            }
        
        # Episode 시작 위치 설정
        self.episode_start_ego = self.parser.parse_ego_info(frame_data)
        
        # 관측값 생성
        observation = self._create_observation(frame_data)
        
        # 초기 상태 저장
        self.prev_ego_info = self.episode_start_ego
        trajectory = self.parser.parse_trajectory(frame_data)
        if trajectory and len(trajectory) > 5:
            goal = trajectory[5]  # 5번째 포인트를 목표로
            if self.prev_ego_info:
                self.prev_distance_to_goal = np.sqrt(
                    (goal.x - self.prev_ego_info.x)**2 + 
                    (goal.y - self.prev_ego_info.y)**2
                )
        
        return observation
    
    def get_observation(self) -> Dict[str, np.ndarray]:
        """GT 데이터에서 다음 관측값 가져오기"""
        if not self.parser:
            return {
                'grid': np.random.rand(*self.observation_shape).astype(np.float32),
                'vectors': np.zeros(4, dtype=np.float32),
                'trajectory': np.zeros((5, 4), dtype=np.float32)
            }
        
        # 다음 프레임으로 이동
        self.current_frame_idx += 1
        self.steps_in_episode += 1
        
        # 프레임 데이터 가져오기
        frame_data = self.parser.get_frame_data(self.current_frame_idx)
        if not frame_data:
            # 데이터 끝에 도달하면 리셋
            return self.reset()
        
        # 관측값 생성
        observation = self._create_observation(frame_data)
        
        # 현재 상태 저장 (나중에 사용 가능하도록)
        self.current_frame_data = frame_data
        ego_info = self.parser.parse_ego_info(frame_data)
        self.prev_ego_info = ego_info
        
        return observation
    
    def _create_observation(self, frame_data: Dict) -> Dict[str, np.ndarray]:
        """프레임 데이터에서 observation 생성"""
        # 데이터 파싱
        ego_info = self.parser.parse_ego_info(frame_data)
        objects = self.parser.parse_objects(frame_data)
        # ldm_points = self.parser.parse_ldm_points(frame_data)  # 추후 Transformer 모델에서 사용 예정
        ldm_points = []  # 임시로 빈 리스트
        trajectory = self.parser.parse_trajectory(frame_data)
        near_direction, destination_direction = self.parser.get_direction_vectors(frame_data)
        
        # 데이터가 없으면 빈 observation
        if not ego_info:
            return {
                'grid': np.zeros(self.observation_shape, dtype=np.float32),
                'vectors': np.zeros(4, dtype=np.float32),
                'trajectory': np.zeros((0, 4), dtype=np.float32)
            }
        
        # Observation 생성 (episode 시작 위치 기준)
        observation = self.obs_builder.build_observation(
            ego_info, objects, ldm_points, trajectory, near_direction, destination_direction,
            episode_start_ego=self.episode_start_ego
        )
        
        return observation
    
    def get_current_frame_data(self) -> Optional[Dict]:
        """현재 프레임 데이터 반환 (Environment에서 충돌 체크 등에 사용)"""
        if hasattr(self, 'current_frame_data'):
            return self.current_frame_data
        return None
    
    def is_data_ended(self) -> bool:
        """데이터가 끝났는지 확인"""
        if not self.parser:
            return False
        return self.current_frame_idx >= self.parser.get_frame_count() - 5
    
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
            # import rclpy
            # from rclpy.node import Node
            
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
    
    def reset(self) -> Dict[str, np.ndarray]:
        """환경 리셋 (ROS2에서는 현재 상태 반환)"""
        if not self.ros2_initialized:
            # ROS2가 없으면 랜덤 데이터
            return {
                'grid': np.random.rand(*self.observation_shape).astype(np.float32),
                'vectors': np.zeros(4, dtype=np.float32),
                'trajectory': np.zeros((5, 4), dtype=np.float32)
            }
        
        # 현재 ROS2 topic에서 받은 최신 관측값 반환
        observation = self._create_observation_from_ros2()
        return observation
    
    def get_observation(self) -> Dict[str, np.ndarray]:
        """ROS2에서 실시간 관측값 가져오기"""
        if not self.ros2_initialized:
            # ROS2가 없으면 랜덤 데이터
            return {
                'grid': np.random.rand(*self.observation_shape).astype(np.float32),
                'vectors': np.zeros(4, dtype=np.float32),
                'trajectory': np.zeros((5, 4), dtype=np.float32)
            }
        
        # ROS2 메시지 처리 (spin_once)
        # TODO: rclpy.spin_once(self.node, timeout_sec=0.1)
        
        # 최신 데이터로 관측값 생성
        observation = self._create_observation_from_ros2()
        
        return observation
    
    def _create_observation_from_ros2(self) -> Dict[str, np.ndarray]:
        """
        ROS2 메시지에서 관측값 생성
        
        TODO: 실제 변환 로직 구현
        - detected_object_array를 그리드맵으로 변환
        - 자차 속도, 위치 정보 포함
        """
        # 임시 구현
        return {
            'grid': np.random.rand(*self.observation_shape).astype(np.float32),
            'vectors': np.zeros(4, dtype=np.float32)
        }
    
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
        self.episode_start_ego = EgoInfo(x=0, y=0, yaw=0, velocity=0)
        
    def reset(self) -> Dict[str, np.ndarray]:
        self.step_count = 0
        return {
            'grid': np.random.rand(*self.observation_shape).astype(np.float32),
            'vectors': np.zeros(4, dtype=np.float32)
        }
    
    def get_observation(self) -> Dict[str, np.ndarray]:
        self.step_count += 1
        return {
            'grid': np.random.rand(*self.observation_shape).astype(np.float32),
            'vectors': np.zeros(4, dtype=np.float32)
        }
    
    def close(self):
        pass