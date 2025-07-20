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
import random

# 공통 데이터 파서 import
from .data_parser import GTDataParser, ObservationBuilder, EgoInfo, ObjectInfo, LDMPoint


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
        
        # 이전 프레임 정보 (보상 계산용)
        self.prev_ego_info = None
        self.prev_distance_to_goal = None
        
    def reset(self) -> np.ndarray:
        """새로운 에피소드 시작"""
        if not self.parser:
            # GT 데이터가 없으면 임시로 랜덤 데이터 반환
            print("경고: GT 데이터가 없어 랜덤 관측값을 사용합니다")
            return np.random.rand(*self.observation_shape).astype(np.float32)
        
        # 랜덤하게 시작 프레임 선택 (뒤에 충분한 프레임이 있는 곳에서)
        max_start = max(0, self.parser.get_frame_count() - self.max_episode_length - 10)
        self.episode_start_idx = random.randint(0, max_start)
        self.current_frame_idx = self.episode_start_idx
        self.steps_in_episode = 0
        
        # 초기 프레임 데이터 가져오기
        frame_data = self.parser.get_frame_data(self.current_frame_idx)
        if not frame_data:
            return np.zeros(self.observation_shape, dtype=np.float32)
        
        # 관측값 생성
        observation = self._create_observation(frame_data)
        
        # 초기 상태 저장
        self.prev_ego_info = self.parser.parse_ego_info(frame_data)
        trajectory = self.parser.parse_trajectory(frame_data)
        if trajectory and len(trajectory) > 5:
            goal = trajectory[5]  # 5번째 포인트를 목표로
            if self.prev_ego_info:
                self.prev_distance_to_goal = np.sqrt(
                    (goal.x - self.prev_ego_info.x)**2 + 
                    (goal.y - self.prev_ego_info.y)**2
                )
        
        return observation
    
    def get_observation(self, action: Optional[int] = None) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """GT 데이터에서 다음 관측값 가져오기"""
        if not self.parser:
            observation = np.random.rand(*self.observation_shape).astype(np.float32)
            return observation, 0.0, False, {"simulated": True}
        
        # 다음 프레임으로 이동
        self.current_frame_idx += 1
        self.steps_in_episode += 1
        
        # 프레임 데이터 가져오기
        frame_data = self.parser.get_frame_data(self.current_frame_idx)
        if not frame_data:
            # 데이터 끝에 도달
            return self.reset(), 0.0, True, {"reason": "data_end"}
        
        # 관측값 생성
        observation = self._create_observation(frame_data)
        
        # 보상 계산
        reward = self._calculate_reward(action, frame_data)
        
        # 종료 조건 확인
        done = self._check_termination(frame_data)
        
        # 정보 딕셔너리
        ego_info = self.parser.parse_ego_info(frame_data)
        info = {
            "step": self.steps_in_episode,
            "frame_idx": self.current_frame_idx,
            "timestamp": self.parser.get_timestamp(frame_data),
            "ego_velocity": ego_info.velocity if ego_info else 0.0
        }
        
        # 현재 상태를 이전 상태로 저장
        self.prev_ego_info = ego_info
        
        return observation, reward, done, info
    
    def _create_observation(self, frame_data: Dict) -> np.ndarray:
        """프레임 데이터에서 observation 생성"""
        # 데이터 파싱
        ego_info = self.parser.parse_ego_info(frame_data)
        objects = self.parser.parse_objects(frame_data)
        ldm_points = self.parser.parse_ldm_points(frame_data)
        trajectory = self.parser.parse_trajectory(frame_data)
        
        # 데이터가 없으면 빈 observation
        if not ego_info:
            return np.zeros(self.observation_shape, dtype=np.float32)
        
        # Observation 생성
        observation = self.obs_builder.build_observation(
            ego_info, objects, ldm_points, trajectory
        )
        
        return observation
    
    def _calculate_reward(self, action: Optional[int], frame_data: Dict) -> float:
        """보상 계산"""
        reward = 0.0
        
        # 현재 상태 파싱
        ego_info = self.parser.parse_ego_info(frame_data)
        objects = self.parser.parse_objects(frame_data)
        trajectory = self.parser.parse_trajectory(frame_data)
        
        if not ego_info or not self.prev_ego_info:
            return 0.0
        
        # 1. 생존 보상 (매 스텝 작은 음수)
        reward -= 0.1
        
        # 2. 전진 보상 (속도에 비례)
        if ego_info.velocity > 0:
            reward += min(ego_info.velocity / 10.0, 0.5)  # 최대 0.5
        
        # 3. 충돌 체크 (가까운 객체와의 거리)
        min_distance = float('inf')
        for obj in objects:
            distance = np.sqrt(
                (obj.global_x - ego_info.x)**2 + 
                (obj.global_y - ego_info.y)**2
            )
            min_distance = min(min_distance, distance)
        
        if min_distance < 2.0:  # 2m 이내에 객체가 있으면 페널티
            reward -= (2.0 - min_distance) * 2.0
        
        # 4. 경로 추종 보상
        if trajectory and len(trajectory) > 5:
            # 5번째 trajectory 포인트를 목표로
            goal = trajectory[5]
            distance_to_goal = np.sqrt(
                (goal.x - ego_info.x)**2 + 
                (goal.y - ego_info.y)**2
            )
            
            # 목표에 가까워지면 보상
            if self.prev_distance_to_goal:
                progress = self.prev_distance_to_goal - distance_to_goal
                reward += progress * 0.5
            
            self.prev_distance_to_goal = distance_to_goal
            
            # 목표 도달
            if distance_to_goal < 3.0:
                reward += 10.0
        
        # 5. 액션에 따른 보상 조정
        if action is not None:
            # Action space: 0-7 (positions), 8-15 (positions with stop)
            if action >= 8:  # 정지 액션
                if min_distance < 5.0:  # 가까운 곳에 객체가 있을 때 정지하면 보상
                    reward += 0.2
                else:  # 불필요한 정지는 페널티
                    reward -= 0.3
        
        return reward
    
    def _check_termination(self, frame_data: Dict) -> bool:
        """에피소드 종료 조건 확인"""
        # 1. 최대 스텝 수 도달
        if self.steps_in_episode >= self.max_episode_length:
            return True
        
        # 2. 데이터 끝 근처 도달
        if self.current_frame_idx >= self.parser.get_frame_count() - 5:
            return True
        
        # 3. 충돌 감지 (매우 가까운 객체)
        ego_info = self.parser.parse_ego_info(frame_data)
        objects = self.parser.parse_objects(frame_data)
        
        if ego_info:
            for obj in objects:
                distance = np.sqrt(
                    (obj.global_x - ego_info.x)**2 + 
                    (obj.global_y - ego_info.y)**2
                )
                if distance < 1.0:  # 1m 이내 충돌로 간주
                    return True
        
        return False
    
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