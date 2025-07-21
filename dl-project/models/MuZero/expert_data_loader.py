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
            
            episode = self._create_single_episode(start_idx, episode_length)
            if episode and len(episode) > 10:  # 최소 길이 체크
                episodes.append(episode)
        
        print(f"\n생성된 Episode: {len(episodes)}개")
        return episodes
    
    def _create_single_episode(self, start_idx: int, length: int) -> Optional[Episode]:
        """단일 Episode 생성"""
        episode = Episode()
        
        # 상태 추적
        prev_ego_info = None
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
            ldm_points = self.parser.parse_ldm_points(frame_data)
            trajectory = self.parser.parse_trajectory(frame_data)
            
            if not ego_info:
                break
            
            observation = self.obs_builder.build_observation(
                ego_info, objects, ldm_points, trajectory
            )
            observations.append(observation)
            
            # Action 추론 (다음 프레임과 비교)
            if i < length - 1:
                next_frame = self.parser.get_frame_data(frame_idx + 1)
                if next_frame:
                    next_ego = self.parser.parse_ego_info(next_frame)
                    if next_ego:
                        action = self._infer_action(ego_info, next_ego)
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
        
        # 충돌 여부 확인
        done_reason = self._check_episode_termination(
            observations, rewards, objects if 'objects' in locals() else []
        )
        
        # Target values 계산 (cascade 방식)
        target_values = self._calculate_target_values(rewards, done_reason)
        
        # Episode 생성
        for i in range(len(actions)):
            # Policy는 one-hot (GT 액션이 정답)
            policy = np.zeros(len(self.config.action_space))
            policy[actions[i]] = 1.0
            
            episode.add_step(
                state=torch.tensor(observations[i], dtype=torch.float32),
                action=actions[i],
                reward=rewards[i],
                policy=policy,
                value=target_values[i],
                done=(i == len(actions) - 1)
            )
        
        return episode
    
    def _infer_action(self, current_ego, next_ego) -> int:
        """두 프레임 간 ego 상태 변화로 액션 추론"""
        # 위치 변화
        dx = next_ego.x - current_ego.x
        dy = next_ego.y - current_ego.y
        
        # 속도 확인
        is_stopped = abs(current_ego.velocity) < 0.1
        
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
        
        # Action 인덱스 계산
        if is_stopped:
            return direction_idx + 8  # STOP 액션
        else:
            return direction_idx  # GO 액션
    
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
    
    def _check_episode_termination(self, observations, rewards, final_objects) -> str:
        """에피소드 종료 이유 확인"""
        # 충돌 체크 (마지막 객체들과 거리)
        if final_objects and rewards:
            if rewards[-1] < -0.5:  # 큰 페널티는 충돌 가능성
                return 'collision'
        
        # 목표 도달 (큰 보상)
        if rewards and rewards[-1] > 5.0:
            return 'goal_reached'
        
        return 'max_steps'
    
    def _calculate_target_values(self, rewards: List[float], done_reason: str) -> List[float]:
        """Cascade 방식으로 target value 계산"""
        n = len(rewards)
        target_values = [0.0] * n
        
        # 마지막 value 설정
        if done_reason == 'collision':
            target_values[-1] = -1.0
        elif done_reason == 'goal_reached':
            target_values[-1] = 1.0
        else:
            target_values[-1] = 0.0
        
        # 역방향으로 계산
        for i in range(n - 2, -1, -1):
            target_values[i] = rewards[i] + self.discount * target_values[i + 1]
        
        return target_values