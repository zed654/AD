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
    def get_frame_count(self) -> int:
        """사용 가능한 프레임 수 반환"""
        pass
    
    @abstractmethod
    def get_frame_data(self, frame_idx: int) -> Optional[Dict]:
        """특정 프레임의 원시 데이터 반환"""
        pass
    
    @abstractmethod
    def reset(self) -> Dict[str, np.ndarray]:
        """환경을 리셋하고 초기 관측값을 반환 (self-play용)"""
        pass
    
    @abstractmethod
    def get_observation(self) -> Dict[str, np.ndarray]:
        """현재 상태의 관측값을 반환 (self-play용)"""
        pass
    
    @abstractmethod
    def close(self):
        """리소스 정리"""
        pass


class GTObservationProvider(ObservationProvider):
    """Ground Truth 데이터에서 프레임 데이터를 제공하는 클래스"""
    
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
        
        # Self-play를 위한 상태 변수 (단순화)
        self.current_frame_idx = 0
        self.episode_start_ego = None
        
    def get_frame_count(self) -> int:
        """사용 가능한 프레임 수 반환"""
        if not self.parser:
            return 0
        return self.parser.get_frame_count()
    
    def get_frame_data(self, frame_idx: int) -> Optional[Dict]:
        """특정 프레임의 원시 데이터 반환"""
        if not self.parser:
            return None
        return self.parser.get_frame_data(frame_idx)
    
    def get_parser(self) -> Optional[GTDataParser]:
        """파서 객체 반환 (ExpertDataLoader에서 필요)"""
        return self.parser
    
    def reset(self) -> Dict[str, np.ndarray]:
        """환경을 리셋하고 초기 관측값을 반환 (self-play용)"""
        if not self.parser:
            return {
                'grid': np.random.rand(*self.observation_shape).astype(np.float32),
                'vectors': np.zeros(4, dtype=np.float32)
            }
        
        # 랜덤하게 시작 프레임 선택
        import random
        max_start = max(0, self.parser.get_frame_count() - 100)
        self.current_frame_idx = random.randint(0, max_start)
        
        # 초기 프레임 데이터
        frame_data = self.parser.get_frame_data(self.current_frame_idx)
        if not frame_data:
            return {
                'grid': np.zeros(self.observation_shape, dtype=np.float32),
                'vectors': np.zeros(4, dtype=np.float32)
            }
        
        # Episode 시작 위치 설정
        self.episode_start_ego = self.parser.parse_ego_info(frame_data)
        
        # Observation 생성
        return self._create_observation(frame_data)
    
    def get_observation(self) -> Dict[str, np.ndarray]:
        """현재 상태의 관측값을 반환 (self-play용)"""
        if not self.parser:
            return {
                'grid': np.random.rand(*self.observation_shape).astype(np.float32),
                'vectors': np.zeros(4, dtype=np.float32)
            }
        
        # 다음 프레임으로 이동
        self.current_frame_idx += 1
        
        # 프레임 데이터 가져오기
        frame_data = self.parser.get_frame_data(self.current_frame_idx)
        if not frame_data:
            # 데이터 끝에 도달하면 리셋
            return self.reset()
        
        # Observation 생성
        return self._create_observation(frame_data)
    
    def _create_observation(self, frame_data: Dict) -> Dict[str, np.ndarray]:
        """프레임 데이터에서 observation 생성"""
        # 데이터 파싱
        ego_info = self.parser.parse_ego_info(frame_data)
        objects = self.parser.parse_objects(frame_data)
        ldm_points = []  # 임시로 빈 리스트
        trajectory = self.parser.parse_trajectory(frame_data)
        near_direction, destination_direction = self.parser.get_direction_vectors(frame_data)
        
        # 데이터가 없으면 빈 observation
        if not ego_info:
            return {
                'grid': np.zeros(self.observation_shape, dtype=np.float32),
                'vectors': np.zeros(4, dtype=np.float32)
            }
        
        # Observation 생성 (episode 시작 위치 기준)
        observation = self.obs_builder.build_observation(
            ego_info, objects, ldm_points, trajectory, near_direction, destination_direction,
            episode_start_ego=self.episode_start_ego
        )
        
        return observation
    
    def get_current_frame_data(self) -> Optional[Dict]:
        """현재 프레임 데이터 반환"""
        if hasattr(self, 'current_frame_idx'):
            return self.get_frame_data(self.current_frame_idx)
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
    """ROS2 Topic에서 실시간으로 프레임 데이터를 제공하는 클래스 (추론용)"""
    
    def __init__(self, config):
        """
        Args:
            config: MuZero 설정
        """
        self.config = config
        
        # ROS2 관련 초기화
        self.ros2_initialized = False
        self._init_ros2()
        
        # 최신 데이터 저장
        self.latest_frame_data = None
        self.frame_buffer = []  # 프레임 버퍼
        
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
    
    def get_frame_count(self) -> int:
        """사용 가능한 프레임 수 반환 (ROS2는 실시간이므로 버퍼 크기)"""
        return len(self.frame_buffer)
    
    def get_frame_data(self, frame_idx: int) -> Optional[Dict]:
        """특정 프레임의 데이터 반환 (버퍼에서)"""
        if 0 <= frame_idx < len(self.frame_buffer):
            return self.frame_buffer[frame_idx]
        return self.latest_frame_data  # 최신 데이터 반환
    
    def reset(self) -> Dict[str, np.ndarray]:
        """환경을 리셋하고 초기 관측값을 반환 (ROS2에서는 현재 상태 반환)"""
        if not self.ros2_initialized:
            return {
                'grid': np.random.rand(*self.config.observation_shape).astype(np.float32),
                'vectors': np.zeros(4, dtype=np.float32)
            }
        
        # TODO: 현재 ROS2 topic에서 받은 최신 관측값 반환
        return self._create_observation_from_ros2()
    
    def get_observation(self) -> Dict[str, np.ndarray]:
        """현재 상태의 관측값을 반환 (ROS2에서 실시간 관측값 가져오기)"""
        if not self.ros2_initialized:
            return {
                'grid': np.random.rand(*self.config.observation_shape).astype(np.float32),
                'vectors': np.zeros(4, dtype=np.float32)
            }
        
        # TODO: ROS2 메시지 처리 (spin_once)
        # rclpy.spin_once(self.node, timeout_sec=0.1)
        
        # TODO: 최신 데이터로 관측값 생성
        return self._create_observation_from_ros2()
    
    def _create_observation_from_ros2(self) -> Dict[str, np.ndarray]:
        """
        ROS2 메시지에서 관측값 생성
        
        TODO: 실제 변환 로직 구현
        - detected_object_array를 그리드맵으로 변환
        - 자차 속도, 위치 정보 포함
        """
        # 임시 구현
        return {
            'grid': np.random.rand(*self.config.observation_shape).astype(np.float32),
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
    """테스트용 Mock 데이터 제공자"""
    
    def __init__(self, config):
        self.config = config
        self.observation_shape = config.observation_shape
        self.mock_frames = self._generate_mock_frames(1000)  # 1000개의 가짝 프레임 생성
        self.current_frame_idx = 0
        
    def get_frame_count(self) -> int:
        """사용 가능한 프레임 수 반환"""
        return len(self.mock_frames)
    
    def get_frame_data(self, frame_idx: int) -> Optional[Dict]:
        """특정 프레임의 데이터 반환"""
        if 0 <= frame_idx < len(self.mock_frames):
            return self.mock_frames[frame_idx]
        return None
    
    def reset(self) -> Dict[str, np.ndarray]:
        """환경을 리셋하고 초기 관측값을 반환"""
        self.current_frame_idx = 0
        return {
            'grid': np.random.rand(*self.observation_shape).astype(np.float32),
            'vectors': np.zeros(4, dtype=np.float32)
        }
    
    def get_observation(self) -> Dict[str, np.ndarray]:
        """현재 상태의 관측값을 반환"""
        self.current_frame_idx += 1
        return {
            'grid': np.random.rand(*self.observation_shape).astype(np.float32),
            'vectors': np.zeros(4, dtype=np.float32)
        }
    
    def _generate_mock_frames(self, count: int) -> List[Dict]:
        """테스트용 가짝 프레임 데이터 생성"""
        frames = []
        for i in range(count):
            frame = {
                'timestamp': i * 0.1,  # 0.1초 간격
                'ego': {
                    'x': i * 0.5,  # 전진
                    'y': np.sin(i * 0.1) * 2,  # 사인파 경로
                    'yaw': i * 0.01,
                    'velocity': 5.0
                },
                'trajectory': [{'x': (i+j)*0.5, 'y': np.sin((i+j)*0.1)*2} for j in range(10)]
            }
            frames.append(frame)
        return frames
    
    def close(self):
        pass