"""
공통 데이터 파서 모듈
db3_extractor.py로 추출한 JSONL 데이터를 파싱하고 처리하는 공통 기능
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import math
from dataclasses import dataclass


@dataclass
class EgoInfo:
    """자차 정보"""
    x: float
    y: float
    yaw: float
    velocity: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'x': self.x,
            'y': self.y,
            'yaw': self.yaw,
            'velocity': self.velocity
        }


@dataclass
class ObjectInfo:
    """객체 정보"""
    id: str
    class_id: int
    global_x: float
    global_y: float
    global_yaw: float
    velocity: float
    width: float = 0.0
    height: float = 0.0
    global_points: List[Tuple[float, float]] = None
    
    def __post_init__(self):
        if self.global_points is None:
            self.global_points = []


@dataclass
class LDMPoint:
    """LDM 포인트 정보"""
    x: float
    y: float
    obstacles_type: int
    class_id: int


@dataclass
class TrajectoryPoint:
    """궤적 포인트"""
    x: float
    y: float
    yaw: float
    velocity: float = 0.0
    timestamp: int = 0


class GTDataParser:
    """GT(Ground Truth) JSONL 데이터 파서"""
    
    def __init__(self, filepath: Path):
        """
        Args:
            filepath: JSONL 파일 경로
        """
        self.filepath = Path(filepath)
        self.data = self._load_jsonl_data()
        self.current_index = 0
    
    def _load_jsonl_data(self) -> List[Dict]:
        """JSONL 파일에서 데이터를 로드합니다."""
        data = []
        if not self.filepath.exists():
            raise FileNotFoundError(f"JSONL 파일을 찾을 수 없습니다: {self.filepath}")
            
        with open(self.filepath, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def get_frame_count(self) -> int:
        """전체 프레임 수 반환"""
        return len(self.data)
    
    def get_frame_data(self, index: int) -> Optional[Dict]:
        """특정 인덱스의 프레임 데이터 반환"""
        if 0 <= index < len(self.data):
            return self.data[index]
        return None
    
    def parse_ego_info(self, frame_data: Dict) -> Optional[EgoInfo]:
        """프레임 데이터에서 자차 정보 추출"""
        ego_trajectory = frame_data.get('gt_trajectory', [])
        
        if not ego_trajectory:
            return None
            
        # 첫 번째 trajectory 포인트가 현재 자차 위치
        first_point = ego_trajectory[0]
        pose = first_point.get('_pose', {}).get('_pose', {})
        position = pose.get('_position', {})
        orientation = pose.get('_orientation', {})
        
        # 쿼터니언에서 yaw 계산
        x, y, z, w = (orientation.get('_x', 0), 
                      orientation.get('_y', 0), 
                      orientation.get('_z', 0), 
                      orientation.get('_w', 1))
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        
        # 속도 정보
        linear_vel = first_point.get('_linear_vel', 0.0)
        
        return EgoInfo(
            x=position.get('_x', 0),
            y=position.get('_y', 0),
            yaw=yaw,
            velocity=linear_vel
        )
    
    def parse_objects(self, frame_data: Dict) -> List[ObjectInfo]:
        """프레임 데이터에서 객체 정보 추출"""
        objects_data = frame_data.get('input_environment', {}).get('_objects', [])
        objects = []
        
        for obj_data in objects_data:
            global_center = obj_data.get('_global_center', {})
            
            # 객체 포인트들
            global_points = []
            for point in obj_data.get('_global_points', []):
                global_points.append((
                    point.get('_x', 0),
                    point.get('_y', 0)
                ))
            
            obj = ObjectInfo(
                id=obj_data.get('_id', ''),
                class_id=obj_data.get('_class_id', -1),
                global_x=global_center.get('_x', 0),
                global_y=global_center.get('_y', 0),
                global_yaw=obj_data.get('_global_yaw_heading', 0),
                velocity=obj_data.get('_global_velocity', 0),
                width=obj_data.get('_width', 0),
                height=obj_data.get('_height', 0),
                global_points=global_points
            )
            objects.append(obj)
            
        return objects
    
    def parse_ldm_points(self, frame_data: Dict) -> List[LDMPoint]:
        """프레임 데이터에서 LDM 포인트 추출"""
        ldm_data = frame_data.get('input_environment', {}).get('_ldm_points', [])
        ldm_points = []
        
        for point_data in ldm_data:
            point = LDMPoint(
                x=point_data.get('x', 0),
                y=point_data.get('y', 0),
                obstacles_type=point_data.get('obstacles_type', 0),
                class_id=point_data.get('class_id', 0)
            )
            ldm_points.append(point)
            
        return ldm_points
    
    def parse_trajectory(self, frame_data: Dict) -> List[TrajectoryPoint]:
        """프레임 데이터에서 GT trajectory 추출"""
        gt_trajectory = frame_data.get('gt_trajectory', [])
        trajectory_points = []
        
        for gt_point in gt_trajectory:
            pose = gt_point.get('_pose', {}).get('_pose', {})
            position = pose.get('_position', {})
            orientation = pose.get('_orientation', {})
            
            # 쿼터니언에서 yaw 계산
            x, y, z, w = (orientation.get('_x', 0), 
                          orientation.get('_y', 0), 
                          orientation.get('_z', 0), 
                          orientation.get('_w', 1))
            yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            
            point = TrajectoryPoint(
                x=position.get('_x', 0),
                y=position.get('_y', 0),
                yaw=yaw,
                velocity=gt_point.get('_linear_vel', 0.0),
                timestamp=gt_point.get('_timestamp_abs_rel', 0)
            )
            trajectory_points.append(point)
            
        return trajectory_points
    
    def get_direction_vectors(self, frame_data: Dict) -> Tuple[Optional[Dict], Optional[Dict]]:
        """방향 벡터 정보 추출"""
        input_env = frame_data.get('input_environment', {})
        near_direction = input_env.get('near_direction_vector')
        destination_direction = input_env.get('destination_direction_vector')
        return near_direction, destination_direction
    
    def get_timestamp(self, frame_data: Dict) -> int:
        """프레임의 타임스탬프 추출"""
        header = frame_data.get('input_environment', {}).get('_header', {})
        stamp = header.get('_stamp', {})
        if stamp:
            return stamp.get('_sec', 0) * 1_000_000_000 + stamp.get('_nanosec', 0)
        return 0


class CoordinateTransformer:
    """좌표 변환 유틸리티"""
    
    @staticmethod
    def global_to_ego(global_x: float, global_y: float, 
                      ego_x: float, ego_y: float, ego_yaw: float) -> Tuple[float, float]:
        """글로벌 좌표를 자차 중심 좌표로 변환"""
        # 자차 기준 상대 좌표
        dx = global_x - ego_x
        dy = global_y - ego_y
        
        # 자차 좌표계로 회전
        cos_yaw = np.cos(-ego_yaw)
        sin_yaw = np.sin(-ego_yaw)
        
        ego_x = dx * cos_yaw - dy * sin_yaw
        ego_y = dx * sin_yaw + dy * cos_yaw
        
        return ego_x, ego_y
    
    @staticmethod
    def ego_to_screen(ego_x: float, ego_y: float) -> Tuple[float, float]:
        """자차 좌표를 화면 좌표로 변환 (y축이 전방)"""
        screen_x = -ego_y
        screen_y = ego_x
        return screen_x, screen_y
    
    @staticmethod
    def global_to_screen(global_x: float, global_y: float,
                        ego_x: float, ego_y: float, ego_yaw: float) -> Tuple[float, float]:
        """글로벌 좌표를 화면 좌표로 직접 변환"""
        ego_x_coord, ego_y_coord = CoordinateTransformer.global_to_ego(
            global_x, global_y, ego_x, ego_y, ego_yaw
        )
        return CoordinateTransformer.ego_to_screen(ego_x_coord, ego_y_coord)


class ObservationBuilder:
    """MuZero 학습을 위한 Observation 생성기"""
    
    def __init__(self, config):
        """
        Args:
            config: MuZero 설정 (observation_shape, grid_size 등 포함)
        """
        self.config = config
        self.observation_shape = config.observation_shape
        
        # 그리드 설정 (예: 10x12 그리드, 각 셀은 3x3m)
        if len(self.observation_shape) == 3:
            self.channels = self.observation_shape[0]  # 1
            self.grid_rows = self.observation_shape[1]  # 10
            self.grid_cols = self.observation_shape[2]  # 12
        else:
            self.channels = 1
            self.grid_rows = self.observation_shape[0]  # 10
            self.grid_cols = self.observation_shape[1]  # 12
        self.cell_size = getattr(config, 'cell_size', 3.0)  # 각 셀의 크기 (m)
        
        # 자차 중심은 그리드의 하단 중앙
        self.ego_grid_row = self.grid_rows - 1
        self.ego_grid_col = self.grid_cols // 2
    
    def build_observation(self, ego_info: EgoInfo, objects: List[ObjectInfo], 
                         ldm_points: List[LDMPoint], 
                         trajectory: List[TrajectoryPoint]) -> np.ndarray:
        """
        프레임 데이터로부터 MuZero 학습용 observation 생성
        
        Returns:
            observation: shape이 config.observation_shape인 numpy 배열
        """
        # 빈 그리드 초기화
        if len(self.observation_shape) == 3:
            grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
        else:
            grid = np.zeros(self.observation_shape, dtype=np.float32)
        
        # 1. 자차 위치 표시
        grid[self.ego_grid_row, self.ego_grid_col] = 1.0
        
        # 2. 객체들을 그리드에 배치
        for obj in objects:
            # 자차 기준 상대 좌표 계산
            rel_x, rel_y = CoordinateTransformer.global_to_ego(
                obj.global_x, obj.global_y, ego_info.x, ego_info.y, ego_info.yaw
            )
            
            # 그리드 인덱스 계산
            grid_row = self.ego_grid_row - int(rel_x / self.cell_size)
            grid_col = self.ego_grid_col + int(rel_y / self.cell_size)
            
            # 그리드 범위 확인
            if 0 <= grid_row < self.grid_rows and 0 <= grid_col < self.grid_cols:
                # 클래스별로 다른 값 할당
                if obj.class_id == 100:  # 보행자
                    grid[grid_row, grid_col] = 0.3
                elif obj.class_id in [202, 204, 205]:  # 차량류
                    grid[grid_row, grid_col] = 0.6
                else:
                    grid[grid_row, grid_col] = 0.4
        
        # 3. LDM 포인트 추가 (장애물 정보)
        for ldm in ldm_points:
            rel_x, rel_y = CoordinateTransformer.global_to_ego(
                ldm.x, ldm.y, ego_info.x, ego_info.y, ego_info.yaw
            )
            
            grid_row = self.ego_grid_row - int(rel_x / self.cell_size)
            grid_col = self.ego_grid_col + int(rel_y / self.cell_size)
            
            if 0 <= grid_row < self.grid_rows and 0 <= grid_col < self.grid_cols:
                if ldm.obstacles_type >= 1:  # 장애물이 있는 경우
                    grid[grid_row, grid_col] = max(grid[grid_row, grid_col], 0.2)
        
        # 4. GT trajectory 표시 (목표 경로)
        for i, traj_point in enumerate(trajectory[:5]):  # 처음 5개 포인트만
            rel_x, rel_y = CoordinateTransformer.global_to_ego(
                traj_point.x, traj_point.y, ego_info.x, ego_info.y, ego_info.yaw
            )
            
            grid_row = self.ego_grid_row - int(rel_x / self.cell_size)
            grid_col = self.ego_grid_col + int(rel_y / self.cell_size)
            
            if 0 <= grid_row < self.grid_rows and 0 <= grid_col < self.grid_cols:
                # 경로는 음수 값으로 표시 (구분을 위해)
                grid[grid_row, grid_col] = -0.5
        
        # 3차원 shape이 필요한 경우 차원 추가
        if len(self.observation_shape) == 3:
            grid = grid[np.newaxis, :, :]  # (H, W) -> (1, H, W)
        
        return grid
    
    def decode_observation(self, observation: np.ndarray) -> Dict[str, Any]:
        """observation을 디버깅용으로 디코드"""
        info = {
            'ego_position': (self.ego_grid_row, self.ego_grid_col),
            'objects': [],
            'trajectory': [],
            'obstacles': []
        }
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                value = observation[row, col]
                if value > 0 and (row, col) != (self.ego_grid_row, self.ego_grid_col):
                    if value >= 0.6:
                        info['objects'].append(('vehicle', row, col))
                    elif value >= 0.3:
                        info['objects'].append(('pedestrian', row, col))
                    elif value >= 0.2:
                        info['obstacles'].append((row, col))
                elif value < 0:
                    info['trajectory'].append((row, col))
        
        return info