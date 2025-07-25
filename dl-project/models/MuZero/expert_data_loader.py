"""
전문가 데이터(JSON)에서 Episode를 생성하는 모듈
GT 데이터를 학습용 Episode로 변환
"""
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from .replay_buffer import Episode
from .data_parser import GTDataParser
from .observation_provider import ObservationBuilder, ObservationProvider
from .action import Action


class ExpertDataLoader:
    """JSON GT 데이터에서 학습용 Episode 생성"""
    
    def __init__(self, config, observation_provider: Optional['ObservationProvider'] = None):
        """
        Args:
            config: MuZero 설정
            observation_provider: GT 데이터 제공자 (선택적)
        """
        self.config = config
        self.discount = config.discount
        
        # ObservationProvider에서 parser와 데이터 가져오기
        if observation_provider and hasattr(observation_provider, 'get_parser'):
            self.parser = observation_provider.get_parser()
        else:
            # ObservationProvider가 없으면 직접 생성
            self.parser = GTDataParser(config.gt_data_path)
            
        self.obs_builder = ObservationBuilder(config)
        
        if self.parser:
            print(f"전문가 데이터 로드: {self.parser.get_frame_count()}개 프레임")
        else:
            print("경고: GT 데이터 파서를 찾을 수 없습니다")
    
    def create_episodes_from_json(self, 
                                 num_episodes: int = 10,
                                 episode_length: int = 50,
                                 overlap: int = 25) -> List[Episode]:
        """
        JSON 데이터에서 Episode 생성
        
        Args:
            num_episodes: 생성할 에피소드 수
            episode_length: 각 에피소드 길이
            overlap: 에피소드 간 겹치는 프레임 수
        
        Returns:
            Episode 리스트
        """
        if not self.parser:
            print("경고: GT 데이터가 없어 Episode를 생성할 수 없습니다")
            return []
            
        episodes = []
        total_frames = self.parser.get_frame_count()
        
        # 시작 위치 계산 (overlap 고려)
        step = episode_length - overlap
        start_positions = list(range(0, total_frames - episode_length, step))[:num_episodes]
        
        for ep_idx, start_idx in enumerate(start_positions):
            print(f"Episode {ep_idx + 1}/{num_episodes} 생성 중...")
            
            episode = self._create_single_episode(start_idx, episode_length, ep_idx)
            if episode and len(episode) > 10:  # 최소 길이 체크
                episodes.append(episode)
        
        print(f"\n생성된 Episode: {len(episodes)}개")
        return episodes
    
    def _create_single_episode(self, start_idx: int, length: int, ep_idx: int = 0) -> Optional[Episode]:
        """단일 Episode 생성"""
        episode = Episode()
        
        # 상태 추적
        self.episode_start_ego = None  # Episode 시작 위치
        episode_near_direction = None  # Episode 전체에서 사용할 근거리 방향 (첫 스텝에서 고정)
        episode_final_direction = None  # Episode 전체에서 사용할 목적지 방향 (첫 스텝에서 고정)
        observations = []
        expert_ego_positions = []  # GT에서 추출한 전문가 ego 위치 (그리드 좌표)
        
        for i in range(length):
            frame_idx = start_idx + i
            if frame_idx >= self.parser.get_frame_count() - 1:
                break
            
            # 현재 프레임 데이터
            frame_data = self.parser.get_frame_data(frame_idx)
            if not frame_data:
                break
            
            # Observation 생성
            ego_info = self.parser.parse_ego_info(frame_data)
            objects = self.parser.parse_objects(frame_data)
            ldm_points = self.parser.parse_ldm_points(frame_data)  # LDM 데이터 파싱
            near_destination_direction, final_destination_direction = self.parser.get_direction_vectors(frame_data)
            
            if not ego_info:
                break
            
            # Episode 시작 위치 설정 (첫 프레임에서만)
            if i == 0:
                self.episode_start_ego = ego_info
            
            observation = self.obs_builder.build_observation(
                ego_info, objects, ldm_points, near_destination_direction, final_destination_direction,
                episode_start_ego=self.episode_start_ego
            )
            
            # observation에서 필요한 정보 추출
            grid = observation['grid']  # 3채널 그리드 (3, 99, 99)
            vectors = observation['vectors']  # 방향 벡터 [near_x, near_y, dest_x, dest_y]
            
            # 첫 스텝에서만 방향 벡터를 저장하고, 이후 스텝에서는 재사용
            if i == 0 and vectors is not None and len(vectors) == 4:
                # 첫 스텝에서 Episode 전체에서 사용할 방향 저장
                if not (vectors[0] == 0 and vectors[1] == 0):  # near direction이 유효한 경우
                    episode_near_direction = {
                        'x': round(float(vectors[0]), 2), 
                        'y': round(float(vectors[1]), 2)
                    }
                if not (vectors[2] == 0 and vectors[3] == 0):  # destination direction이 유효한 경우
                    episode_final_direction = {
                        'x': round(float(vectors[2]), 2), 
                        'y': round(float(vectors[3]), 2)
                    }
            
            # observation은 dict 형태로 저장
            state_dict = {
                'obstacles': grid[0],  # Channel 0: 동적 장애물 (/obstacles_tracking_msg, 현재는 동적/정적 섞여있음. 정적은 ldm_msg 의 static 을 의미하는게 아님.)
                'static_environment': grid[1],  # Channel 1: 정적 환경 (/ldm_msg 의 Cliff, Static Obstacles, Missing Points)
                'ground_class_id': grid[2],  # Channel 2: 지면 semantic class ID (/ldm_msg 의 Semantic Segmentation Class ID)
                'near_destination_direction': episode_near_direction,  # 근거리 방향 (Episode 첫 스텝에서 고정)
                'final_destination_direction': episode_final_direction  # 목적지 방향 (Episode 첫 스텝에서 고정)
            }
            
            # 디버깅 포인트 1: observation 확인 (필요시 주석 해제)
            # if i == 0:  # 첫 프레임만
            #     print(f"\n=== Episode {ep_idx + 1}, Frame {i} ===")
            #     print(f"Observation shape: {grid_observation.shape}")
            #     print(f"Ego info: x={ego_info.x:.2f}, y={ego_info.y:.2f}, yaw={ego_info.yaw:.2f}")
            #     print(f"Objects: {len(objects)}개")
            #     print(f"Trajectory points: {len(trajectory)}개")
            #     if near_destination_direction:
            #         print(f"Near direction: x={near_destination_direction.get('x', 0):.2f}, y={near_destination_direction.get('y', 0):.2f}")
            #     if final_destination_direction:
            #         print(f"Destination direction: x={final_destination_direction.get('x', 0):.2f}, y={final_destination_direction.get('y', 0):.2f}")
            #     
            #     # Observation 내용 확인
            #     decoded = self.obs_builder.decode_observation(observation)
            #     print(f"Decoded observation: {decoded}")
            
            observations.append(state_dict)  # 전체 state dict 저장
            
            # GT ego 위치를 그리드 좌표로 변환하여 저장 (다음 프레임) - 마지막 프레임은 제외
            if i < length - 1 and frame_idx < self.parser.get_frame_count() - 1:
                next_frame = self.parser.get_frame_data(frame_idx + 1)
                next_ego = self.parser.parse_ego_info(next_frame) if next_frame else None
                
                if next_ego and self.episode_start_ego:
                    # 다음 프레임의 ego 위치를 Episode 시작점 기준 그리드 좌표로 변환
                    rel_x, rel_y = self._global_to_grid_coords(
                        next_ego.x, next_ego.y, 
                        self.episode_start_ego.x, self.episode_start_ego.y, 
                        self.episode_start_ego.yaw
                    )
                    expert_ego_pos = (rel_x, rel_y)  # 그리드 좌표 (row, col)
                    expert_ego_positions.append(expert_ego_pos)
        
        # Episode 구성 (마지막 프레임 제외)
        if len(expert_ego_positions) == 0:
            return None
        
        # Episode 생성
        for i in range(len(expert_ego_positions)):
            # 디버깅 포인트 2: Episode step 추가 (필요시 주석 해제)
            # if i < 3:  # 처음 3개 step만
            #     print(f"\n--- Step {i} ---")
            #     print(f"Expert Action: x={expert_actions[i][0]:.2f}, y={expert_actions[i][1]:.2f}")
            
            # state dict를 Episode에 추가 (obstacles와 static_environment를 tensor로 변환)
            state_with_tensor = observations[i].copy()
            state_with_tensor['obstacles'] = torch.tensor(observations[i]['obstacles'], dtype=torch.float32)
            
            # static_environment가 있으면 tensor로 변환
            if 'static_environment' in observations[i] and isinstance(observations[i]['static_environment'], np.ndarray):
                state_with_tensor['static_environment'] = torch.tensor(observations[i]['static_environment'], dtype=torch.float32)
            
            episode.add_step(
                state=state_with_tensor,  # dict 형태의 state
                action=0,           # self-play action은 0으로 초기화 (POS0: 현재 위치)
                expert_ego_pos=expert_ego_positions[i],  # GT에서 추출한 전문가 ego 위치 (그리드 좌표)
                reward=0.0,         # GT에서는 항상 0
                policy=None,        # MCTS/self-play에서 계산됨 (Action 개수만큼 있어야함)
                value=0.0,          # 0으로 초기화
                done=(i == len(expert_ego_positions) - 1)
            )
        
        return episode
    
    def _global_to_grid_coords(self, global_x: float, global_y: float, 
                              ego_start_x: float, ego_start_y: float, ego_start_yaw: float) -> Tuple[int, int]:
        """전역 좌표를 그리드 좌표로 변환
        
        Args:
            global_x, global_y: 변환할 전역 좌표
            ego_start_x, ego_start_y, ego_start_yaw: Episode 시작점의 ego 정보
            
        Returns:
            (grid_row, grid_col): 그리드 좌표
        """
        # Episode 시작점 기준 상대 좌표로 변환
        dx = global_x - ego_start_x
        dy = global_y - ego_start_y
        
        # 자차 좌표계로 회전
        cos_yaw = np.cos(-ego_start_yaw)
        sin_yaw = np.sin(-ego_start_yaw)
        
        rel_x = dx * cos_yaw - dy * sin_yaw
        rel_y = dx * sin_yaw + dy * cos_yaw
        
        # 그리드 좌표로 변환 (중앙이 49, 49)
        # 주의: rel_x는 전방(forward)이고, rel_y는 좌우(left/right)
        # Grid에서 row는 y축(아래가 큰 값), col은 x축(오른쪽이 큰 값)
        grid_center = self.config.observation_shape[1] // 2  # 49
        grid_row = grid_center + int(rel_x / self.config.cell_size)  # 180도 회전: 전방이 아래쪽
        grid_col = grid_center - int(rel_y / self.config.cell_size)  # 180도 회전: 오른쪽이 왼쪽
        
        return grid_row, grid_col
    
    def _parse_ldm_to_grid(self, ldm_points: List, ego_info: Any, episode_start_ego: Any) -> Dict[str, np.ndarray]:
        """LDM 포인트를 그리드 좌표로 변환하고 obstacles_type별로 분리
        
        Args:
            ldm_points: LDM 포인트 리스트
            ego_info: 현재 프레임의 ego 정보
            episode_start_ego: Episode 시작점의 ego 정보
            
        Returns:
            obstacles_type별로 분리된 그리드 딕셔너리
            - 'cliff': 절벽/경사 (obstacles_type=1)
            - 'static_obstacles': 정적 장애물 (obstacles_type=2)
            - 'missing_points': 측정 불가 영역 (obstacles_type=3)
        """
        # 그리드 크기 (obstacles와 동일)
        grid_shape = (self.config.observation_shape[1], self.config.observation_shape[2])  # (99, 99)
        
        # 각 타입별 그리드 초기화
        static_env = {
            'cliff': np.zeros(grid_shape, dtype=np.float32),
            'static_obstacles': np.zeros(grid_shape, dtype=np.float32),
            'missing_points': np.zeros(grid_shape, dtype=np.float32)
        }
        
        if not ldm_points or not episode_start_ego:
            return static_env
        
        # 각 LDM 포인트를 그리드 좌표로 변환
        for ldm in ldm_points:
            # Episode 시작점 기준 그리드 좌표로 변환
            grid_row, grid_col = self._global_to_grid_coords(
                ldm.x, ldm.y,
                episode_start_ego.x, episode_start_ego.y, episode_start_ego.yaw
            )
            
            # 그리드 범위 체크
            if 0 <= grid_row < grid_shape[0] and 0 <= grid_col < grid_shape[1]:
                # obstacles_type에 따라 적절한 그리드에 표시
                if ldm.obstacles_type == 1:  # cliff
                    static_env['cliff'][grid_row, grid_col] = 1.0
                elif ldm.obstacles_type == 2:  # static_obstacles
                    static_env['static_obstacles'][grid_row, grid_col] = 1.0
                elif ldm.obstacles_type == 3:  # missing_points
                    static_env['missing_points'][grid_row, grid_col] = 1.0
                # obstacles_type이 0 또는 기타 값인 경우는 무시
        
        return static_env
    
    
    
    
