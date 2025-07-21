"""
전문가 데이터(JSON)에서 Episode를 생성하는 모듈
GT 데이터를 학습용 Episode로 변환
"""
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Tuple
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
        prev_ego_info = None
        self.episode_start_ego = None  # Episode 시작 위치
        observations = []
        actions = []
        rewards = []
        
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
            
            # observation은 이제 dict이므로 grid만 추출
            grid_observation = observation['grid']
            
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
            
            observations.append(grid_observation)  # grid만 저장
            
            # Action 추론 (다음 프레임과 비교)
            if i < length - 1:
                next_frame = self.parser.get_frame_data(frame_idx + 1)
                if next_frame:
                    next_ego = self.parser.parse_ego_info(next_frame)
                    if next_ego:
                        # Episode 시작점 기준 상대 위치로 action 추론
                        action = self._infer_action_relative(ego_info, next_ego, self.episode_start_ego)
                        actions.append(action)
                        
                        # Reward 계산
                        reward = self._calculate_reward(
                            ego_info, next_ego, objects, trajectory
                        )
                        rewards.append(reward)
            
            prev_ego_info = ego_info
        
        # Episode 구성 (마지막 프레임 제외)
        if len(actions) == 0:
            return None
        
        # Episode 생성
        for i in range(len(actions)):
            # Policy는 one-hot (GT 액션이 정답)
            policy = np.zeros(len(self.config.action_space))
            policy[actions[i]] = 1.0
            
            # 디버깅 포인트 2: Episode step 추가 (필요시 주석 해제)
            # if i < 3:  # 처음 3개 step만
            #     print(f"\n--- Step {i} ---")
            #     print(f"Action: {actions[i]} (GO={actions[i]<8}, direction={actions[i]%8})")
            #     print(f"Reward: {rewards[i]:.3f}")
            #     print(f"Policy: {np.argmax(policy)}")
            
            episode.add_step(
                state=torch.tensor(observations[i], dtype=torch.float32),
                action=actions[i],
                reward=rewards[i],
                policy=policy,
                value=0.0,  # 학습 시 계산할 예정
                done=(i == len(actions) - 1)
            )
        
        return episode
    
    def _infer_action_relative(self, current_ego, next_ego, episode_start_ego) -> int:
        """
        Episode 시작점 기준 상대 좌표로 action 추론
        
        Args:
            current_ego: 현재 프레임의 ego 정보
            next_ego: 다음 프레임의 ego 정보  
            episode_start_ego: Episode 시작 시점의 ego 정보
        """
        # Episode 시작점 기준 상대 좌표 계산
        from models.MuZero.data_parser import CoordinateTransformer
        
        # 현재 위치를 episode 시작점 기준으로 변환
        curr_rel_x, curr_rel_y = CoordinateTransformer.global_to_ego(
            current_ego.x, current_ego.y, 
            episode_start_ego.x, episode_start_ego.y, episode_start_ego.yaw
        )
        
        # 다음 위치를 episode 시작점 기준으로 변환
        next_rel_x, next_rel_y = CoordinateTransformer.global_to_ego(
            next_ego.x, next_ego.y,
            episode_start_ego.x, episode_start_ego.y, episode_start_ego.yaw
        )
        
        # 이동 거리 계산 (상대 좌표 기준)
        dx = next_rel_x - curr_rel_x
        dy = next_rel_y - curr_rel_y
        distance = np.sqrt(dx**2 + dy**2)
        
        # 다음 위치에 따른 그리드 셀 위치 계산
        # 0.2m 셀 크기 기준
        cell_size = 0.2
        grid_dx = int(round(dx / cell_size))
        grid_dy = int(round(dy / cell_size))
        
        # Position 매핑 (9개 위치)
        # POS0: 현재 위치 (0, 0)
        # POS1: 좌측상단 (-1, 1)
        # POS2: 상단 (0, 1)  
        # POS3: 우측상단 (1, 1)
        # POS4: 우측 (1, 0)
        # POS5: 우측하단 (1, -1)
        # POS6: 하단 (0, -1)
        # POS7: 좌측하단 (-1, -1)
        # POS8: 좌측 (-1, 0)
        
        position_map = {
            (0, 0): 0,   # POS0
            (-1, 1): 1,  # POS1
            (0, 1): 2,   # POS2
            (1, 1): 3,   # POS3
            (1, 0): 4,   # POS4
            (1, -1): 5,  # POS5
            (0, -1): 6,  # POS6
            (-1, -1): 7, # POS7
            (-1, 0): 8,  # POS8
        }
        
        # 가장 가까운 그리드 셀로 반올림
        grid_dx = max(min(grid_dx, 1), -1)
        grid_dy = max(min(grid_dy, 1), -1)
        
        position_idx = position_map.get((grid_dx, grid_dy), 0)  # 기본값 POS0
        
        # Action 인덱스는 position 인덱스와 동일 (0~8)
        return position_idx
    
    def _infer_action(self, current_ego, next_ego) -> int:
        """기존 메서드 - 호환성을 위해 유지 (더 이상 사용하지 않음)"""
        # 위치 변화
        dx = next_ego.x - current_ego.x
        dy = next_ego.y - current_ego.y
        
        # 진행 방향 계산 (ego 좌표계)
        angle = np.arctan2(dy, dx)
        ego_angle = current_ego.yaw
        relative_angle = angle - ego_angle
        
        # -π ~ π 범위로 정규화
        while relative_angle > np.pi:
            relative_angle -= 2 * np.pi
        while relative_angle < -np.pi:
            relative_angle += 2 * np.pi
        
        # 8방향으로 양자화
        direction_idx = int((relative_angle + np.pi) / (2 * np.pi / 8)) % 8
        
        # Action 인덱스 계산 (9개 위치 중 하나)
        return direction_idx  # 0~7, POS0은 정지 상태에서만 사용
    
    def _calculate_reward(self, current_ego, next_ego, objects, trajectory) -> float:
        """프레임 간 보상 계산"""
        reward = 0.0
        
        # 1. 생존 보상
        reward -= 0.1
        
        # 2. 전진 보상
        if current_ego.velocity > 0:
            reward += min(current_ego.velocity / 10.0, 0.5)
        
        # 3. 충돌 위험 페널티
        min_distance = float('inf')
        for obj in objects:
            distance = np.sqrt(
                (obj.global_x - current_ego.x)**2 + 
                (obj.global_y - current_ego.y)**2
            )
            min_distance = min(min_distance, distance)
        
        if min_distance < 2.0:
            reward -= (2.0 - min_distance) * 2.0
            
        # 4. 경로 추종 보상
        if trajectory and len(trajectory) > 5:
            goal = trajectory[5]
            distance_to_goal = np.sqrt(
                (goal.x - current_ego.x)**2 + 
                (goal.y - current_ego.y)**2
            )
            if distance_to_goal < 3.0:
                reward += 10.0
        
        return reward
    
