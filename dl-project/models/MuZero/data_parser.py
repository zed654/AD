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
    # width: float = 0.0
    # height: float = 0.0
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
                # width=obj_data.get('_width', 0),
                # height=obj_data.get('_height', 0),
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


def get_object_color(class_id: int) -> str:
    """클래스 ID에 따른 색상을 반환합니다."""
    color_map = {
        100: 'red',      # 보행자
        202: 'blue',     # 차량  
        203: 'orange',   # 자전거
        204: 'purple',   # 트럭
        205: 'green',    # 버스
        300: 'gray',     # 기타
        0: 'darkred',    # Unknown
        -1: 'black',     # 미분류
    }
    return color_map.get(class_id, 'black')


def get_ldm_color(obstacles_type: int) -> Optional[str]:
    """LDM obstacles_type에 따른 색상을 반환합니다."""
    color_map = {
        1: 'lime',       # cliff - 연두색
        2: 'yellow',     # static_obstacles - 노란색
        3: 'purple'      # missing_points - 보라색
    }
    if obstacles_type >= 1:
        return color_map.get(obstacles_type, 'gray')
    return None


def get_ldm_segmentation_color(class_id: int) -> str:
    """LDM class_id (segmentation)에 따른 색상을 반환합니다."""
    color_map = {
        0: '#000000',   # Background
        1: '#3D3DF5',   # Blocks
        2: '#CC3366',   # Colored Road
        3: '#F078F0',   # Cement Asphalt
        4: '#008000',   # Crosswalk
        5: '#733380',   # Speed Bumps
        6: '#FF6A4D',   # Manhole
        7: '#5986B3',   # Curbs
        8: '#00FF00',   # Grass
        9: '#CC9933',   # Soil/Gravel
        10: '#34D1B7',  # Grating
        11: '#FFCC33',  # Guide Blocks
        12: '#DBDB8D',  # Rock
        13: '#FF007C',  # Unidentified
        14: '#942701',  # Wood
        15: '#AEC7E8',  # Iron
        16: '#A7FFD8',  # Artificial Turf
    }
    return color_map.get(class_id, '#FFFFFF')


# MuZero 그리드 값 상수 정의
GRID_VALUE_EMPTY = 0.0
GRID_VALUE_PEDESTRIAN = 0.3
GRID_VALUE_OTHER_OBJECT = 0.4
GRID_VALUE_VEHICLE = 0.6
GRID_VALUE_EGO = 1.0


def get_grid_value_from_class_id(class_id: int) -> float:
    """
    객체 클래스 ID를 MuZero 그리드 값으로 변환합니다.
    
    Args:
        class_id: 객체 클래스 ID
        
    Returns:
        그리드 값 (0.0 ~ 1.0)
    """
    if class_id == 100:  # 보행자
        return GRID_VALUE_PEDESTRIAN
    elif class_id in [202, 204, 205]:  # 차량류 (차량, 트럭, 버스)
        return GRID_VALUE_VEHICLE
    else:  # 기타 객체
        return GRID_VALUE_OTHER_OBJECT


def get_grid_value_info(value: float) -> Tuple[str, str]:
    """
    MuZero 그리드 값에 대한 정보를 반환합니다.
    
    Returns:
        (객체 타입 이름, 색상 코드)
    """
    if value == GRID_VALUE_EMPTY:
        return "Empty", "black"
    elif value == GRID_VALUE_PEDESTRIAN:
        return "Pedestrian", get_object_color(100)
    elif value == GRID_VALUE_OTHER_OBJECT:
        return "Other Object", get_object_color(300)
    elif value == GRID_VALUE_VEHICLE:
        return "Vehicle", get_object_color(202)
    elif value == GRID_VALUE_EGO:
        return "Ego Vehicle", "white"
    else:
        return f"Unknown ({value})", "gray"


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
        
        # 그리드 설정 (예: 99x99 그리드, 각 셀은 0.2x0.2m)
        if len(self.observation_shape) == 3:
            self.channels = self.observation_shape[0]  # 1
            self.grid_rows = self.observation_shape[1]  # 99
            self.grid_cols = self.observation_shape[2]  # 99
        else:
            self.channels = 1
            self.grid_rows = self.observation_shape[0]  # 99
            self.grid_cols = self.observation_shape[1]  # 99
        self.cell_size = getattr(config, 'cell_size', 0.2)  # 각 셀의 크기 (m)
        
        # 자차 중심은 그리드의 정중앙
        self.ego_grid_row = self.grid_rows // 2
        self.ego_grid_col = self.grid_cols // 2
    
    def build_observation(self, ego_info: EgoInfo, objects: List[ObjectInfo], 
                         ldm_points: List[LDMPoint], 
                         trajectory: List[TrajectoryPoint],
                         near_direction: Optional[Dict] = None,
                         destination_direction: Optional[Dict] = None,
                         episode_start_ego: Optional[EgoInfo] = None) -> Dict[str, np.ndarray]:
        """
        프레임 데이터로부터 MuZero 학습용 observation 생성
        
        Args:
            episode_start_ego: Episode 시작 시점의 ego 정보 (None이면 현재 ego를 기준으로 함)
            
        Returns:
            observation_dict: 'grid', 'vectors', 'trajectory' 키를 가진 딕셔너리
                - grid: shape이 config.observation_shape인 numpy 배열 (장애물 정보만)
                - vectors: 방향 벡터들 (near_direction, destination_direction)
                - trajectory: GT trajectory 정보 (별도 관리)
        """
        # 빈 그리드 초기화
        if len(self.observation_shape) == 3:
            grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
        else:
            grid = np.zeros(self.observation_shape, dtype=np.float32)
        
        # Episode 시작점 기준 좌표계 설정
        if episode_start_ego is None:
            episode_start_ego = ego_info
            
        # 1. 자차 위치 표시 (episode 시작점 기준)
        ego_rel_x, ego_rel_y = CoordinateTransformer.global_to_ego(
            ego_info.x, ego_info.y, episode_start_ego.x, episode_start_ego.y, episode_start_ego.yaw
        )
        ego_grid_row = self.ego_grid_row - int(ego_rel_x / self.cell_size)
        ego_grid_col = self.ego_grid_col + int(ego_rel_y / self.cell_size)
        
        if 0 <= ego_grid_row < self.grid_rows and 0 <= ego_grid_col < self.grid_cols:
            grid[ego_grid_row, ego_grid_col] = GRID_VALUE_EGO
        
        # 2. 객체들을 그리드에 배치 (episode 시작점 기준)
        for obj in objects:
            # 객체의 중심점과 모든 꼭짓점을 그리드에 표시
            grid_value = get_grid_value_from_class_id(obj.class_id)
            
            # 중심점 표시
            rel_x, rel_y = CoordinateTransformer.global_to_ego(
                obj.global_x, obj.global_y, episode_start_ego.x, episode_start_ego.y, episode_start_ego.yaw
            )
            grid_row = self.ego_grid_row - int(rel_x / self.cell_size)
            grid_col = self.ego_grid_col + int(rel_y / self.cell_size)
            
            if 0 <= grid_row < self.grid_rows and 0 <= grid_col < self.grid_cols:
                grid[grid_row, grid_col] = grid_value
            
            # 모든 꼭짓점(points) 표시
            for point_x, point_y in obj.global_points:
                rel_x, rel_y = CoordinateTransformer.global_to_ego(
                    point_x, point_y, episode_start_ego.x, episode_start_ego.y, episode_start_ego.yaw
                )
                grid_row = self.ego_grid_row - int(rel_x / self.cell_size)
                grid_col = self.ego_grid_col + int(rel_y / self.cell_size)
                
                if 0 <= grid_row < self.grid_rows and 0 <= grid_col < self.grid_cols:
                    grid[grid_row, grid_col] = grid_value
        
        # 3. LDM 포인트 추가 (장애물 정보) - 추후 Transformer 모델에서 사용 예정
        # for ldm in ldm_points:
        #     rel_x, rel_y = CoordinateTransformer.global_to_ego(
        #         ldm.x, ldm.y, ego_info.x, ego_info.y, ego_info.yaw
        #     )
        #     
        #     grid_row = self.ego_grid_row - int(rel_x / self.cell_size)
        #     grid_col = self.ego_grid_col + int(rel_y / self.cell_size)
        #     
        #     if 0 <= grid_row < self.grid_rows and 0 <= grid_col < self.grid_cols:
        #         if ldm.obstacles_type >= 1:  # 장애물이 있는 경우
        #             grid[grid_row, grid_col] = max(grid[grid_row, grid_col], 0.2)
        
        # 4. GT trajectory 별도 처리 (grid에 표시하지 않음)
        trajectory_array = np.zeros((len(trajectory), 4), dtype=np.float32)  # [x, y, yaw, velocity]
        for i, traj_point in enumerate(trajectory):
            # Episode 시작점 기준 상대 좌표
            rel_x, rel_y = CoordinateTransformer.global_to_ego(
                traj_point.x, traj_point.y, episode_start_ego.x, episode_start_ego.y, episode_start_ego.yaw
            )
            # 상대 yaw (Episode 시작점 기준)
            rel_yaw = traj_point.yaw - episode_start_ego.yaw
            trajectory_array[i] = [rel_x, rel_y, rel_yaw, traj_point.velocity]
        
        # 5. 방향 벡터 정보 준비 (별도 벡터로 저장)
        direction_vectors = np.zeros(4, dtype=np.float32)  # [near_x, near_y, dest_x, dest_y]
        
        if near_direction:
            near_x = near_direction.get('x', 0)
            near_y = near_direction.get('y', 0)
            
            # 방향 벡터를 자차 좌표계로 변환 (벡터이므로 회전만 적용)
            cos_yaw = np.cos(-ego_info.yaw)
            sin_yaw = np.sin(-ego_info.yaw)
            ego_near_x = near_x * cos_yaw - near_y * sin_yaw
            ego_near_y = near_x * sin_yaw + near_y * cos_yaw
            
            # 정규화 (단위 벡터로)
            length = np.sqrt(ego_near_x**2 + ego_near_y**2)
            if length > 0.01:
                direction_vectors[0] = ego_near_x / length
                direction_vectors[1] = ego_near_y / length
        
        if destination_direction:
            dest_x = destination_direction.get('x', 0)
            dest_y = destination_direction.get('y', 0)
            
            # 방향 벡터를 자차 좌표계로 변환
            cos_yaw = np.cos(-ego_info.yaw)
            sin_yaw = np.sin(-ego_info.yaw)
            ego_dest_x = dest_x * cos_yaw - dest_y * sin_yaw
            ego_dest_y = dest_x * sin_yaw + dest_y * cos_yaw
            
            # 정규화 (단위 벡터로)
            length = np.sqrt(ego_dest_x**2 + ego_dest_y**2)
            if length > 0.01:
                direction_vectors[2] = ego_dest_x / length
                direction_vectors[3] = ego_dest_y / length
        
        # 3차원 shape이 필요한 경우 차원 추가
        if len(self.observation_shape) == 3:
            grid = grid[np.newaxis, :, :]  # (H, W) -> (1, H, W)
        
        return {
            'grid': grid,
            'vectors': direction_vectors,
            'trajectory': trajectory_array
        }
    
    def decode_observation(self, observation: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """observation을 디버깅용으로 디코드"""
        info = {
            'ego_position': None,  # Episode 시작점 기준이므로 ego 위치는 별도 계산 필요
            'objects': [],
            'trajectory': [],
            'obstacles': [],
            'near_direction': None,
            'destination_direction': None
        }
        
        # 그리드 정보 디코드
        grid = observation['grid']
        vectors = observation['vectors']
        trajectory = observation.get('trajectory', np.array([]))
        
        # 3D observation을 2D로 변환 (채널이 있는 경우)
        if len(grid.shape) == 3:
            grid = grid[0]  # 첫 번째 채널만 사용
        
        # Ego 위치 찾기
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                value = grid[row, col]
                if abs(value - 1.0) < 0.01:  # Ego vehicle
                    info['ego_position'] = (row, col)
                elif value >= 0.6:
                    info['objects'].append(('vehicle', row, col))
                elif value >= 0.3:
                    info['objects'].append(('pedestrian', row, col))
                elif value >= 0.2:
                    info['obstacles'].append((row, col))
        
        # 방향 벡터 정보 디코드
        if vectors is not None and len(vectors) >= 4:
            info['near_direction'] = (vectors[0], vectors[1]) if abs(vectors[0]) > 0.01 or abs(vectors[1]) > 0.01 else None
            info['destination_direction'] = (vectors[2], vectors[3]) if abs(vectors[2]) > 0.01 or abs(vectors[3]) > 0.01 else None
        
        # GT trajectory 정보 디코드
        if len(trajectory) > 0:
            info['trajectory'] = [(pt[0], pt[1], pt[2], pt[3]) for pt in trajectory]  # (x, y, yaw, velocity)
        
        return info