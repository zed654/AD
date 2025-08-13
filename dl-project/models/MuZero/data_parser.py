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

# rosetta.py에서 상수와 함수들 import
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / 'data' / 'db3_parser'))
from rosetta import (
    # Grid values
    GRID_VALUE_EMPTY, GRID_VALUE_PEDESTRIAN, GRID_VALUE_OTHER_OBJECT,
    GRID_VALUE_VEHICLE,
    GRID_VALUE_CLIFF, GRID_VALUE_STATIC_OBSTACLE, GRID_VALUE_MISSING_POINTS,
    # Functions
    get_object_name, get_object_color,
    get_ldm_segmentation_name, get_ldm_segmentation_color,
    get_object_category, is_vehicle, is_person,
    get_grid_value_from_object_class_id, get_grid_value_from_ldm_obstacles_type,
    get_grid_value_info
)


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
            filepath: JSONL 파일 경로 또는 폴더 경로
        """
        self.filepath = Path(filepath)
        self.data = self._load_jsonl_data()
        self.current_index = 0
    
    def _load_jsonl_data(self) -> List[Dict]:
        """JSONL 파일(들)에서 데이터를 로드합니다."""
        data = []
        
        if self.filepath.is_file():
            # 단일 파일 처리
            if not self.filepath.exists():
                raise FileNotFoundError(f"JSONL 파일을 찾을 수 없습니다: {self.filepath}")
            data.extend(self._load_single_jsonl(self.filepath))
            print(f"단일 JSONL 파일 로드: {self.filepath} ({len(data)} 프레임)")
            
        elif self.filepath.is_dir():
            # 폴더 내 모든 JSONL 파일 처리
            jsonl_files = list(self.filepath.glob("*.jsonl"))
            if not jsonl_files:
                raise FileNotFoundError(f"폴더에 JSONL 파일이 없습니다: {self.filepath}")
            
            print(f"폴더에서 {len(jsonl_files)}개 JSONL 파일 발견: {self.filepath}")
            for jsonl_file in sorted(jsonl_files):  # 정렬된 순서로 로드
                file_data = self._load_single_jsonl(jsonl_file)
                data.extend(file_data)
                print(f"  - {jsonl_file.name}: {len(file_data)} 프레임")
            
            print(f"총 {len(data)} 프레임 로드 완료")
        else:
            raise FileNotFoundError(f"파일 또는 폴더를 찾을 수 없습니다: {self.filepath}")
            
        return data
    
    def _load_single_jsonl(self, filepath: Path) -> List[Dict]:
        """단일 JSONL 파일에서 데이터를 로드합니다."""
        data = []
        with open(filepath, 'r') as f:
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
        near_destination_direction = input_env.get('near_destination_direction_vector')
        final_destination_direction = input_env.get('final_destination_direction_vector')
        return near_destination_direction, final_destination_direction
    
    def get_timestamp(self, frame_data: Dict) -> int:
        """프레임의 타임스탬프 추출"""
        header = frame_data.get('input_environment', {}).get('_header', {})
        stamp = header.get('_stamp', {})
        if stamp:
            return stamp.get('_sec', 0) * 1_000_000_000 + stamp.get('_nanosec', 0)
        return 0










def interpret_class_id(class_id: float, is_ldm: bool = False) -> str:
    """Channel 2의 class_id 값을 해석합니다.
    
    Args:
        class_id: Channel 2에서 읽은 class_id 값
        is_ldm: LDM semantic class인지 여부
        
    Returns:
        해석된 class 이름
    """
    class_id_int = int(class_id)
    
    if class_id_int == 0:
        return "Empty/Background"
    elif is_ldm or (0 < class_id_int < 100):
        # LDM semantic class (1-25, 99)
        return get_ldm_segmentation_name(class_id_int)
    else:
        # Object class (100+) - 하지만 Channel 2에는 LDM만 들어가야 함
        return get_object_name(class_id_int)


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
                         near_destination_direction: Optional[Dict] = None,
                         final_destination_direction: Optional[Dict] = None,
                         episode_start_ego: Optional[EgoInfo] = None) -> Dict[str, np.ndarray]:
        """
        프레임 데이터로부터 MuZero 학습용 observation 생성
        
        Args:
            episode_start_ego: Episode 시작 시점의 ego 정보 (None이면 현재 ego를 기준으로 함)
            
        Returns:
            observation_dict: 'grid', 'vectors' 키를 가진 딕셔너리
                - grid: shape이 config.observation_shape인 numpy 배열 (3채널: 동적객체, 정적환경, 지면class_id)
                - vectors: 방향 벡터들 (near_destination_direction, final_destination_direction)
        """
        # 빈 그리드 초기화 (3채널)
        # Channel 0, 1: float32 (0.0 ~ 1.0 값들)
        # Channel 2: int32 (class_id)
        grid = np.zeros(self.observation_shape, dtype=np.float32)  # (3, 99, 99)
        grid[:2, :, :] = GRID_VALUE_EMPTY  # Channel 0, 1은 EMPTY로 초기화
        # Channel 2는 0으로 초기화 (Empty/Background)
        
        # Episode 시작점 기준 좌표계 설정
        if episode_start_ego is None:
            episode_start_ego = ego_info
            
        # 1. 객체들을 그리드에 배치 (episode 시작점 기준)
        # 주의: Ego는 Grid에 표시하지 않음. ego_pos로 별도 전달됨
        for obj in objects:
            # 객체의 중심점과 모든 꼭짓점을 그리드에 표시
            grid_value = get_grid_value_from_object_class_id(obj.class_id)
            
            # 중심점 표시
            rel_x, rel_y = CoordinateTransformer.global_to_ego(
                obj.global_x, obj.global_y, episode_start_ego.x, episode_start_ego.y, episode_start_ego.yaw
            )
            grid_row = self.ego_grid_row + int(rel_x / self.cell_size)  # 180도 회전: 전방이 아래쪽
            grid_col = self.ego_grid_col - int(rel_y / self.cell_size)  # 180도 회전: 오른쪽이 왼쪽
            
            if 0 <= grid_row < self.grid_rows and 0 <= grid_col < self.grid_cols:
                grid[0, grid_row, grid_col] = grid_value  # Channel 0: Dynamic objects
                # Channel 2는 LDM semantic 정보만 저장, objects는 저장하지 않음
            
            # 모든 꼭짓점(points) 표시
            for point_x, point_y in obj.global_points:
                rel_x, rel_y = CoordinateTransformer.global_to_ego(
                    point_x, point_y, episode_start_ego.x, episode_start_ego.y, episode_start_ego.yaw
                )
                grid_row = self.ego_grid_row + int(rel_x / self.cell_size)  # 180도 회전
                grid_col = self.ego_grid_col - int(rel_y / self.cell_size)  # 180도 회전
                
                if 0 <= grid_row < self.grid_rows and 0 <= grid_col < self.grid_cols:
                    grid[0, grid_row, grid_col] = grid_value  # Channel 0: Dynamic objects
                    # Channel 2는 LDM semantic 정보만 저장, objects는 저장하지 않음
        
        # 3. LDM 포인트 추가 (모든 타입을 Channel 1에 통합)
        for ldm in ldm_points:
            # Episode 시작점 기준 상대 좌표로 변환
            rel_x, rel_y = CoordinateTransformer.global_to_ego(
                ldm.x, ldm.y, episode_start_ego.x, episode_start_ego.y, episode_start_ego.yaw
            )
            
            grid_row = self.ego_grid_row + int(rel_x / self.cell_size)  # 180도 회전: 전방이 아래쪽
            grid_col = self.ego_grid_col - int(rel_y / self.cell_size)  # 180도 회전: 오른쪽이 왼쪽
            
            if 0 <= grid_row < self.grid_rows and 0 <= grid_col < self.grid_cols:
                # rosetta.py의 함수 사용하여 그리드 값 변환
                grid_value = get_grid_value_from_ldm_obstacles_type(ldm.obstacles_type)
                if grid_value > 0:  # GRID_VALUE_EMPTY가 아닌 경우만 설정
                    grid[1, grid_row, grid_col] = grid_value
                
                # LDM의 semantic class_id를 Channel 2에 저장 (obstacles_type과 무관하게 모든 LDM 포인트)
                grid[2, grid_row, grid_col] = ldm.class_id
        
        # 4. 방향 벡터 정보 준비 (별도 벡터로 저장)
        direction_vectors = np.zeros(4, dtype=np.float32)  # [near_x, near_y, dest_x, dest_y]
        
        if near_destination_direction:
            near_x = near_destination_direction.get('x', 0)
            near_y = near_destination_direction.get('y', 0)
            
            # 방향 벡터는 이미 차량 좌표계 기준으로 저장됨
            # 정규화 (단위 벡터로)
            length = np.sqrt(near_x**2 + near_y**2)
            if length > 0.01:
                direction_vectors[0] = near_x / length
                direction_vectors[1] = near_y / length
        
        if final_destination_direction:
            dest_x = final_destination_direction.get('x', 0)
            dest_y = final_destination_direction.get('y', 0)
            
            # 방향 벡터는 이미 차량 좌표계 기준으로 저장됨
            # 정규화 (단위 벡터로)
            length = np.sqrt(dest_x**2 + dest_y**2)
            if length > 0.01:
                direction_vectors[2] = dest_x / length
                direction_vectors[3] = dest_y / length
        
        return {
            'grid': grid,
            'vectors': direction_vectors
        }
    
    def decode_observation(self, observation: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """observation을 디버깅용으로 디코드"""
        info = {
            'ego_position': None,  # Episode 시작점 기준이므로 ego 위치는 별도 계산 필요
            'objects': [],
            'trajectory': [],
            'obstacles': [],
            'near_destination_direction': None,
            'final_destination_direction': None
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
            info['near_destination_direction'] = (vectors[0], vectors[1]) if abs(vectors[0]) > 0.01 or abs(vectors[1]) > 0.01 else None
            info['final_destination_direction'] = (vectors[2], vectors[3]) if abs(vectors[2]) > 0.01 or abs(vectors[3]) > 0.01 else None
        
        # GT trajectory 정보 디코드
        if len(trajectory) > 0:
            info['trajectory'] = [(pt[0], pt[1], pt[2], pt[3]) for pt in trajectory]  # (x, y, yaw, velocity)
        
        return info


