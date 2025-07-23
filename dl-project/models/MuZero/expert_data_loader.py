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
from .observation_provider import ObservationBuilder
from .action import Action


class ExpertDataLoader:
    """JSON GT 데이터에서 학습용 Episode 생성"""
    
    def __init__(self, config, data_path: Optional[Path] = None):
        """
        Args:
            config: MuZero 설정
            data_path: GT 데이터 파일 경로 (JSONL 형식)
        """
        self.config = config
        self.data_path = data_path or config.gt_data_path
        self.discount = config.discount
        
        # 데이터 파서 및 observation 빌더 초기화
        self.parser = GTDataParser(self.data_path)
        self.obs_builder = ObservationBuilder(config)
        
        print(f"전문가 데이터 로드: {self.parser.get_frame_count()}개 프레임")
    
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
            # ldm_points = self.parser.parse_ldm_points(frame_data)  # 추후 Transformer 모델 입력으로 사용 예정
            ldm_points = []  # 임시로 빈 리스트
            trajectory = self.parser.parse_trajectory(frame_data)
            near_direction, destination_direction = self.parser.get_direction_vectors(frame_data)
            
            if not ego_info:
                break
            
            # Episode 시작 위치 설정 (첫 프레임에서만)
            if i == 0:
                self.episode_start_ego = ego_info
            
            observation = self.obs_builder.build_observation(
                ego_info, objects, ldm_points, trajectory, near_direction, destination_direction,
                episode_start_ego=self.episode_start_ego
            )
            
            # LDM을 static_environment로 파싱 (obstacles_type별로 분리)
            static_environment = self._parse_ldm_to_grid(ldm_points, ego_info, self.episode_start_ego)
            
            # observation은 dict 형태로 저장 (obstacles + 추가 정보)
            # 추가 정보 포함: static_environment, objects, near_direction, destination_direction, ego_pos
            state_dict = {
                'obstacles': observation['grid'],  # grid를 obstacles로 이름 변경
                'static_environment': static_environment,  # LDM을 그리드 좌표로 변환하고 타입별로 분리
                'objects': objects,  # 객체 정보
                'near_direction': near_direction,  # 근거리 방향
                'destination_direction': destination_direction,  # 목적지 방향
                'ego_pos': (ego_info.x, ego_info.y)  # 자차 위치 (전역좌표)
            }
            
            # 디버깅 포인트 1: observation 확인 (필요시 주석 해제)
            # if i == 0:  # 첫 프레임만
            #     print(f"\n=== Episode {ep_idx + 1}, Frame {i} ===")
            #     print(f"Observation shape: {grid_observation.shape}")
            #     print(f"Ego info: x={ego_info.x:.2f}, y={ego_info.y:.2f}, yaw={ego_info.yaw:.2f}")
            #     print(f"Objects: {len(objects)}개")
            #     print(f"Trajectory points: {len(trajectory)}개")
            #     if near_direction:
            #         print(f"Near direction: x={near_direction.get('x', 0):.2f}, y={near_direction.get('y', 0):.2f}")
            #     if destination_direction:
            #         print(f"Destination direction: x={destination_direction.get('x', 0):.2f}, y={destination_direction.get('y', 0):.2f}")
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
            
            # state dict를 Episode에 추가 (obstacles tensor만 torch.tensor로 변환)
            state_with_tensor_obstacles = observations[i].copy()
            state_with_tensor_obstacles['obstacles'] = torch.tensor(observations[i]['obstacles'], dtype=torch.float32)
            
            episode.add_step(
                state=state_with_tensor_obstacles,  # dict 형태의 state
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
        grid_center = self.config.observation_shape[1] // 2  # 49
        grid_row = grid_center - int(rel_x / self.config.cell_size)
        grid_col = grid_center + int(rel_y / self.config.cell_size)
        
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
    
    
    
    
