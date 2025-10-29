"""
Self-play 관리자
MuZero의 Self-play 게임을 실행하고 데이터를 수집
"""
import torch
import torch.multiprocessing as mp
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from .mcts import MCTS, IllegalStateException
from .environment import BaseEnvironment
from .action import Action
from .replay_buffer import ReplayBuffer, Episode
from .utils import build_observation_tensor
from .models import get_network
from .config import MuZeroConfig
import time

# rosetta.py에서 Grid value 상수 import
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / 'data' / 'db3_parser'))
from rosetta import (
    GRID_VALUE_CLIFF, GRID_VALUE_STATIC_OBSTACLE, GRID_VALUE_MISSING_POINTS
)


class GameHistory:
    """단일 게임의 히스토리를 저장"""
    
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.pi = []
        self.values = []
        
    def store(self, observation, action, reward, pi, value):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.pi.append(pi)
        self.values.append(value)
        
    def get_length(self):
        return len(self.actions)


class SelfPlayManager:
    """Self-play 게임 관리자"""
    
    def __init__(self,
                 network,
                 environment: BaseEnvironment,
                 config,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.network = network
        self.environment = environment
        self.config = config
        self.device = device
        
        # MCTS 초기화
        self.mcts = MCTS(
            network=network,
            action_space_size=len(config.action_space),
            num_simulations=config.num_simulations,
            mcts_value_backup_discount=config.mcts_value_backup_discount,
            puct_discount=config.puct_discount,
            c1=config.pb_c_init,
            c2=config.pb_c_base,
            device=device,
            config=config  # Config 전달 추가
        )

    # FIXME: 추후 삭제
    def enrich_episode_with_mcts(self, episode: 'Episode') -> 'Episode':
        # print(f"\nEpisode {ep_idx + 1}/{len(episodes)}에 MCTS 적용 중...")
        # print(f"  - MCTS 시뮬레이션 횟수: {self.config.num_simulations}회")

        # Simulated ego position 추적 (그리드 좌표)
        simulated_ego_row, simulated_ego_col = 49, 49  # 그리드 중앙에서 시작. self.observation_shape 의 중앙임

        # dones 필드 초기화 (충돌 tracking용)
        if not hasattr(episode, 'dones'):
            episode.dones = [False] * len(episode.states)

        # Episode 유효성 플래그
        episode_is_valid = True

        # 각 스텝에 대해 MCTS 실행
        for step_idx in range(len(episode.states)):
            # observation 가져오기 (dict 형태)
            state_dict = episode.states[step_idx]
            
            # observation tensor 추출
            if isinstance(state_dict, dict):
                # 현재 simulated_ego_pos 업데이트 후 shift 적용
                # import copy
                # state_dict_copy = copy.deepcopy(state_dict)
                # state_dict_copy['simulated_ego_pos'] = (simulated_ego_row, simulated_ego_col)
                episode.states[step_idx]['simulated_ego_pos'] = (simulated_ego_row, simulated_ego_col)

                # Ego 위치 기준으로 observation shift (처음에는 제자리임)
                # Episode 의 Step 들은 Root Step 의 위치를 공유함. 단, MCTS 시뮬레이션에 의해 선택된 Action 을 기반으로 중심점이 조정되도록 유도.
                episode.states[step_idx] = self.shift_observation(episode.states[0], (simulated_ego_row, simulated_ego_col))

                # 6채널 observation 생성 (공용 함수 사용) -> simulated_ego_pos 가 변해도(=step 이 진행되어도) 환경은 root 시점 그대로 고정됨.
                device = self.device if hasattr(self, 'device') else 'cpu'
                observation_tensor = build_observation_tensor(episode.states[step_idx], self.config.observation_shape, device, self.config.use_obstacle_channels)

                # Legal mask 계산
                # Episode 의 첫 Step 위치가 장애물 위 일 경우, 모든 액션이 불법일 수 있음. 발생사례 있음. (=step_idx 가 0인 사례)
                legal_mask = self._calculate_legal_mask(
                    episode.states[step_idx],
                    (49, 49)
                    # (simulated_ego_row, simulated_ego_col)
                )

                # for legal_mask_idx in range(len(legal_mask)):
                #     if legal_mask[legal_mask_idx] == 0.0:
                #         print(f"    - 스텝 {step_idx}에서 액션 {legal_mask_idx}는 불법입니다.")

                # MCTS 실행
                # - representation(): observation → latent
                # - 시뮬레이션: dynamics()로 미래 예측, prediction()으로 policy/value 계산
                # - 최적 action 선택 및 root node의 value/π 반환
                try:
                    with torch.no_grad():
                        mcts_result = self.mcts.run(
                            observation_tensor,
                            add_exploration_noise=True,
                            legal_mask=legal_mask
                        )
                except IllegalStateException as e:
                    # 모든 액션이 불법인 경우 Episode 즉시 종료
                    print(f"  [Episode {ep_idx + 1}] 스텝 {step_idx}에서 {str(e)}")
                    # Episode를 현재 스텝 이전까지만 잘라냄
                    # episode.states = episode.states[:step_idx]
                    # episode.actions = episode.actions[:step_idx]
                    # episode.rewards = episode.rewards[:step_idx]
                    # episode.pi = episode.pi[:step_idx]
                    # episode.values = episode.values[:step_idx]
                    # episode.dones = episode.dones[:step_idx]
                    # episode.legal_mask = episode.legal_mask[:step_idx]
                    # if hasattr(episode, 'expert_ego_pos'):
                    #     episode.expert_ego_pos = episode.expert_ego_pos[:step_idx]
                    # 이 Episode는 너무 짧으면 버림
                    # if len(episode.states) < 3:
                        # print(f"  [Episode {ep_idx + 1}] Episode가 너무 짧아서 버립니다. (길이: {len(episode.states)})")
                    episode_is_valid = False
                    # Episode 종료
                    break

                # Episode에 MCTS 결과 업데이트
                episode.pi[step_idx] = mcts_result['policy_targets']
                # MCTS가 계산한 value 저장 (bootstrap에 사용됨)
                episode.values[step_idx] = mcts_result['root_value']
                # Legal mask 저장
                episode.legal_mask[step_idx] = legal_mask

                # MCTS가 선택한 action 저장
                selected_action = mcts_result['action']
                # selected_action = 2  # Action 고정 테스트 (2=POS2=전방/상단)
                episode.actions[step_idx] = selected_action
                
                # Action에 따라 simulated_ego_pos 업데이트 (최종 value 계산용)
                action = Action.from_index(selected_action)
                dx, dy = action.get_movement_vector()
                # 좌표계: x -> 상단으로 + / y -> 우측으로 +
                simulated_ego_row += dx  # 전방/후방
                simulated_ego_col += dy  # 좌/우 (environment.py와 일치)
                
                # Episode state에도 업데이트 (최종 거리 계산을 위해)
                episode.states[step_idx]['simulated_ego_pos'] = (simulated_ego_row, simulated_ego_col)

                # 디버깅: 데이터 타입 확인 (처음 몇 스텝만)
                # if step_idx < 2:
                #     obs_type = type(episode.states[step_idx].get('obstacles', None))
                #     print(f"    - 스텝 {step_idx} 후 obstacles 타입: {obs_type}")
                
                # GT 학습을 위한 환경 reward 계산 (Step마다)
                reward = self._calculate_target_reward(
                    episode, step_idx, selected_action, 
                    (simulated_ego_row, simulated_ego_col)
                )
                episode.rewards[step_idx] = reward
                
                # # 충돌 발생 시 Episode 종료
                # # 합법 마스킹 적용된 상황에서는 절대 발생되면 안됨
                # if reward == self.config.collision_penalty:
                #     print(f"    - 충돌로 인해 Episode 종료 (스텝 {step_idx + 1}/{len(episode.states)})")
                #     # 현재 스텝을 터미널로 표시 (잘라내기 전에 먼저 설정)
                #     if step_idx < len(episode.dones):
                #         episode.dones[step_idx] = True
                #     # Episode 길이를 충돌 지점까지로 잘라냄
                #     episode.states = episode.states[:step_idx + 1]
                #     episode.rewards = episode.rewards[:step_idx + 1]
                #     episode.pi = episode.pi[:step_idx + 1]
                #     episode.values = episode.values[:step_idx + 1]
                #     episode.actions = episode.actions[:step_idx + 1]
                #     episode.dones = episode.dones[:step_idx + 1]
                #     episode.legal_mask = episode.legal_mask[:step_idx + 1]
                #     # 정답 위치는 유지. Target Reward 의 목표로 활용해야함
                #     # if hasattr(episode, 'expert_ego_pos'):
                #     #     episode.expert_ego_pos = episode.expert_ego_pos[:step_idx + 1]
                #     break  # MCTS 적용 루프 종료
        # Episode가 유효한 경우만 추가
        if episode_is_valid:
            # n-step bootstrap target value 계산
            # MCTS root values는 보존하고 target_values는 별도로 계산됨
            episode.target_values = self._calculate_target_value(episode)
            return episode
            # enriched_episodes.append(episode)
        return None

    # FIXME: 추후 삭제
    def enrich_episodes_with_mcts(self, episodes: List['Episode']) -> List['Episode']:
        # MCTS Simulation 게임 수행되는 곳
        """GT Episode에 MCTS 결과(π, value) 추가
        
        Args:
            episodes: GT 데이터로부터 생성된 Episode 리스트
            
        Returns:
            MCTS 결과가 추가된 Episode 리스트
        """
        enriched_episodes = []

        # 각 Episode에 대해 MCTS 적용하여 episode 의 step 완성
        for ep_idx, episode in enumerate(episodes):
            episode = self.enrich_episode_with_mcts(episode)
            if episode is not None:
                enriched_episodes.append(episode)

        return enriched_episodes
    
    # FIXME: 추후 삭제
    def shift_observation(self, state_dict: dict, new_pos: Tuple[int, int]) -> dict:
        """observation 내의 그리드 채널들을 새로운 simulated_ego_pos에 맞게 쉬프트

        Args:
            state_dict: 원본 observation dictionary
            new_pos: 새로운 simulated_ego_pos (row, col)

        Returns:
            shifted_state_dict: 쉬프트된 observation dictionary
        """
        import copy

        # Deep copy로 원본 보호
        state_dict_copy = copy.deepcopy(state_dict)

        # Shift 계산: new_pos를 (49, 49)로 보내려면 얼마나 이동해야 하는지
        center = 49
        new_row, new_col = new_pos
        shift_row = center - new_row  # Ego를 중앙으로 이동시키는 shift
        shift_col = center - new_col

        # Shift가 0이면 그대로 반환
        if shift_row == 0 and shift_col == 0:
            return state_dict_copy

        # Channel별로 shift 적용 (slicing 기반, 경계는 0으로 채움)
        def apply_shift(data: np.ndarray, shift_r: int, shift_c: int) -> np.ndarray:
            """2D numpy array를 shift (circular shift 아님)"""
            H, W = data.shape
            shifted = np.zeros_like(data)

            # Source region (원본에서 복사할 영역)
            src_y_start = max(0, -shift_r)
            src_y_end = min(H, H - shift_r)
            src_x_start = max(0, -shift_c)
            src_x_end = min(W, W - shift_c)

            # Destination region (shifted 배열에 붙여넣을 영역)
            dst_y_start = max(0, shift_r)
            dst_y_end = dst_y_start + (src_y_end - src_y_start)
            dst_x_start = max(0, shift_c)
            dst_x_end = dst_x_start + (src_x_end - src_x_start)

            # Copy (나머지는 0으로 유지)
            shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                data[src_y_start:src_y_end, src_x_start:src_x_end]

            return shifted

        # Channels 0, 1, 2 shift 적용
        for key in ['obstacles', 'static_environment', 'ground_class_id']:
            if key in state_dict_copy:
                data = state_dict_copy[key]

                # Torch Tensor를 numpy array로 변환
                if isinstance(data, torch.Tensor):
                    data = data.cpu().numpy()

                if isinstance(data, np.ndarray) and data.ndim == 2:
                    state_dict_copy[key] = apply_shift(data, shift_row, shift_col)
                elif data is not None:
                    print(f"    [WARNING] '{key}' shift 실패: type={type(data)}, shape={data.shape if hasattr(data, 'shape') else 'N/A'}")

        # simulated_ego_pos 업데이트 (항상 중앙으로)
        state_dict_copy['simulated_ego_pos'] = (center, center)

        return state_dict_copy

    # FIXME: 추후 삭제
    def _calculate_legal_mask(self, state_dict: dict, current_pos: Tuple[int, int]) -> np.ndarray:
        """합법 액션 마스크 계산

        현재 위치에서 각 action을 취했을 때 충돌 여부를 체크하여
        합법 마스크 생성 (1=합법, 0=불법)

        Args:
            state_dict: observation dictionary (obstacles, static_environment 포함)
            current_pos: 현재 simulated ego 위치 (row, col)

        Returns:
            legal_mask: 9개 원소의 0/1 배열 (Action 개수만큼)
        """
        legal_mask = np.ones(Action.get_action_space_size(), dtype=np.float32)

        current_row, current_col = current_pos
        boundary = self.config.obstacles_collision_boundary
        half_boundary = boundary // 2

        # 각 action에 대해 충돌 체크
        for action_idx in range(Action.get_action_space_size()):
            action = Action.from_index(action_idx)
            dx, dy = action.get_movement_vector()

            # Action 적용 후 위치
            next_row = current_row + dx
            next_col = current_col + dy

            # 그리드 경계 체크
            if next_row < 0 or next_row >= 99 or next_col < 0 or next_col >= 99:
                legal_mask[action_idx] = 0.0
                continue

            # 장애물 충돌 체크 (obstacles_collision_boundary 영역)
            has_collision = False
            for dr in range(-half_boundary, half_boundary + 1):
                for dc in range(-half_boundary, half_boundary + 1):
                    check_row = next_row + dr
                    check_col = next_col + dc

                    # 그리드 범위 체크
                    if 0 <= check_row < 99 and 0 <= check_col < 99:
                        # Channel 0: Dynamic objects 체크 (모든 값 > 0)
                        if 'obstacles' in state_dict and state_dict['obstacles'] is not None:
                            obs = state_dict['obstacles']
                            if hasattr(obs, 'shape') and len(obs.shape) >= 2:
                                try:
                                    if isinstance(obs, torch.Tensor):
                                        val = obs[check_row, check_col].item()
                                    else:
                                        val = obs[check_row, check_col]
                                    if val > 0:
                                        has_collision = True
                                        break
                                except (IndexError, TypeError):
                                    pass

                        # Channel 1: Static environment 체크
                        # Cliff, Static Obstacle, Missing Points만 충돌로 간주
                        if not has_collision and 'static_environment' in state_dict and state_dict['static_environment'] is not None:
                            static = state_dict['static_environment']
                            if hasattr(static, 'shape') and len(static.shape) >= 2:
                                try:
                                    if isinstance(static, torch.Tensor):
                                        static_val = static[check_row, check_col].item()
                                    else:
                                        static_val = static[check_row, check_col]

                                    # Cliff, Static Obstacle, Missing Points 체크
                                    if (abs(static_val - GRID_VALUE_CLIFF) < 0.01 or
                                        abs(static_val - GRID_VALUE_STATIC_OBSTACLE) < 0.01 or
                                        abs(static_val - GRID_VALUE_MISSING_POINTS) < 0.01):
                                        has_collision = True
                                        break
                                except (IndexError, TypeError):
                                    pass

                if has_collision:
                    break

            if has_collision:
                legal_mask[action_idx] = 0.0

        return legal_mask

    # FIXME: 추후 삭제
    def _calculate_collision_reward(self, episode: Episode, step_idx: int) -> float:
        """충돌 기반 reward 계산
        
        특정 step_idx 시점에서 충돌 발생 여부 체크
        충돌 시 큰 페널티 부여
        """
        
        # step_idx 시점에서만 충돌 체크
        if step_idx < len(episode.states) and isinstance(episode.states[step_idx], dict):
            state = episode.states[step_idx]
            sim_pos = state.get('simulated_ego_pos', (49, 49))
            
            # 3x3 영역 체크 (중심 + 8방향)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    check_row = sim_pos[0] + dr
                    check_col = sim_pos[1] + dc
                    
                    # 그리드 범위 체크
                    if 0 <= check_row < 99 and 0 <= check_col < 99:
                        # Channel 0: Dynamic objects 체크
                        if 'obstacles' in state and state['obstacles'] is not None:
                            obs = state['obstacles']
                            # torch.Tensor와 numpy.ndarray 모두 처리
                            if hasattr(obs, 'shape') and len(obs.shape) >= 2:  # 2D tensor나 array인 경우
                                try:
                                    if isinstance(obs, torch.Tensor):
                                        val = obs[check_row, check_col].item()
                                    else:  # numpy array
                                        val = obs[check_row, check_col]
                                    if val > 0:
                                        # 충돌 감지됨
                                        return self.config.collision_penalty  # 충돌 시 최대 페널티
                                except (IndexError, TypeError):
                                    pass  # 인덱싱 실패 시 무시
                        
                        # Channel 1: Static environment 체크
                        if 'static_environment' in state and state['static_environment'] is not None:
                            static = state['static_environment']
                            # torch.Tensor와 numpy.ndarray 모두 처리
                            if hasattr(static, 'shape') and len(static.shape) >= 2:  # 2D tensor나 array인 경우
                                try:
                                    if isinstance(static, torch.Tensor):
                                        val = static[check_row, check_col].item()
                                    else:  # numpy array
                                        val = static[check_row, check_col]
                                    if val > 0:
                                        # 충돌 감지됨
                                        return self.config.collision_penalty  # 충돌 시 최대 페널티
                                except (IndexError, TypeError):
                                    pass  # 인덱싱 실패 시 무시
        
        return 0.0  # 충돌 없음
    
    # FIXME: 추후 삭제
    def _calculate_smoothness_reward(self, episode: Episode, step_idx: int) -> float:
        """직진성 강화: 방향 일관성(코사인)
           반환: [-0.02, 0.02]"""
        if step_idx < 2:
            return 0.0

        st = episode.states
        for i in (step_idx-2, step_idx-1, step_idx):
            if i >= len(st) or not isinstance(st[i], dict):
                return 0.0
            if st[i].get('simulated_ego_pos') is None:
                return 0.0

        p0 = np.array(st[step_idx-2]['simulated_ego_pos'], dtype=float)
        p1 = np.array(st[step_idx-1]['simulated_ego_pos'], dtype=float)
        p2 = np.array(st[step_idx  ]['simulated_ego_pos'], dtype=float)

        v_prev = p1 - p0
        v_curr = p2 - p1
        prev_magnitude = np.linalg.norm(v_prev)
        curr_magnitude = np.linalg.norm(v_curr)

        # # 정체/미세 이동 페널티(단위-불변으로 너무 크게 주지 않음)
        # if prev_magnitude < 1e-3 or curr_magnitude < 1e-3:
        #     return -0.005

        cos = float(np.dot(v_prev, v_curr) / (prev_magnitude * curr_magnitude + 1e-8))

        # 직진이면 +, 방향 전환이면 - (EMA 제거)
        base = 0.02 * cos  # +0.02(동일방향) ~ -0.02(반대방향)
        return float(np.clip(base, -0.02, 0.02))
    
    # FIXME: 추후 삭제
    def _calculate_goal_following_reward(self, episode: Episode, step_idx: int) -> float:
        """간단한 목표 추종 reward: 가까워지면 +0.1, 멀어지면 -0.1"""

        st = episode.states
        if step_idx >= len(st) or not hasattr(episode, 'expert_ego_pos'):
            return 0.0

        # 목표 위치
        goal = episode.expert_ego_pos[-1]

        # 이전 위치 (첫 스텝이면 시작점 49,49)
        if step_idx == 0:
            prev_pos = (49, 49)
        else:
            prev_pos = st[step_idx - 1].get('simulated_ego_pos')
            if prev_pos is None:
                return 0.0

        # 현재 위치
        curr_pos = st[step_idx].get('simulated_ego_pos')
        if curr_pos is None:
            return 0.0

        # 거리 계산 (유클리드)
        prev_dist = np.sqrt((prev_pos[0] - goal[0])**2 + (prev_pos[1] - goal[1])**2)
        curr_dist = np.sqrt((curr_pos[0] - goal[0])**2 + (curr_pos[1] - goal[1])**2)

        # 가까워지면 +0.1, 멀어지면 -0.1, 동일하면 0
        if curr_dist < prev_dist:
            return 0.1
        elif curr_dist > prev_dist:
            return -0.1
        else:
            return 0.0
        
    # FIXME: 추후 삭제
    def _calculate_goal_direction_reward(self, episode: Episode, step_idx: int) -> float:
        """목표 방향과 이동 방향의 내적 기반 유사도 활용하는 reward"""

        st = episode.states
        # 현재 위치와 목표 위치
        curr_pos = st[step_idx].get('simulated_ego_pos')
        goal_pos = episode.expert_ego_pos[-1]
        # goal_pos = (10, 49); # FIXME: 의도적으로 아래로 가도록 유도 

        # 이전 위치 (이동 벡터 계산용)
        if step_idx == 0:
            prev_pos = (49, 49)
        else:
            prev_pos = st[step_idx - 1].get('simulated_ego_pos')
        
        init_pos = (49, 49)

        # 벡터 계산
        move_vector = np.array(curr_pos) - np.array(prev_pos)  # 실제 이동 방향
        goal_vector = np.array(goal_pos) - np.array(init_pos)  # 목표 방향

        # 내적으로 유사도 계산 (코사인 유사도)
        move_mag = np.linalg.norm(move_vector)
        goal_mag = np.linalg.norm(goal_vector)

        if move_mag < 1e-6 or goal_mag < 1e-6:
            return 0.0

        cos_similarity = np.dot(move_vector, goal_vector) / (move_mag * goal_mag)

        # 목표 방향이면 +0.1, 반대면 -0.1
        return 0.1 * cos_similarity  # [-0.1, +0.1] 범위

    # FIXME: 추후 삭제
    def _calculate_target_reward(self, episode: Episode, step_idx: int, action: int, 
                             sim_pos: Tuple[int, int]) -> float:
        """GT 학습을 위한 환경 reward 계산
        
        trajectory_following_value의 스텝별 버전
        
        Args:
            episode: 현재 Episode
            step_idx: 현재 스텝
            action: 선택된 action (미래 확장용)
            sim_pos: 시뮬레이션 위치 (미래 확장용)
            
        Returns:
            환경 reward (-1 ~ 1 범위)
        """
        # 예외처리 - step_idx가 episode.states 범위를 벗어나는 경우
        if step_idx >= len(episode.states):
            print(f"  - [Target Reward 계산 이상] 스텝 {step_idx}에서 episode.states가 부족합니다. value: 0.0")
            return 0.0
        
        # 목표: GT 만을 추종하는게 아닌, 다중 해를 허용하는 Target Reward 를 설계하는 것.

        # 1. 직진성 강화하는 reward (즉, Path 의 오실레이션이 최소화 되어야함)
        trajectory_smooth_reward = self._calculate_smoothness_reward(episode, step_idx)
        trajectory_smooth_reward = 0 # FIXME: 제거. 테스트용

        # 2. 이전 경로에서 Episode 의 끝 위치로 가까워지는 행동이면 +, 아니면 -
        # Potential-based reward shaping으로 목표 접근을 학습
        # trajectory_goal_follow_reward = self._calculate_goal_following_reward(episode, step_idx)
        trajectory_goal_follow_reward = self._calculate_goal_direction_reward(episode, step_idx)

        # 3. 장애물과 충돌 상황에 대한 패널티
        collision_reward = self._calculate_collision_reward(episode, step_idx)
        
        # # 합법 마스킹 적용된 상황에서는 절대 발생되면 안됨
        # if collision_reward == self.config.collision_penalty:
        #     # 충돌 발생 시 큰 페널티
        #     print(f"  - 충돌 발생! 스텝 {step_idx}에서 충돌 감지됨. 페널티: {self.config.collision_penalty}")
        #     return self.config.collision_penalty
        # else:
        #     # 충돌이 없으면 경로 추종 reward만 사용
        #     return trajectory_goal_follow_reward + trajectory_smooth_reward
        
        return trajectory_goal_follow_reward + trajectory_smooth_reward
        
    # FIXME: 추후 삭제
    def _calculate_target_value(self, episode: Episode):
        """MuZero n-step bootstrap target value 계산

        target[t] = r[t] + γ*r[t+1] + ... + γ^(k-1)*r[t+k-1] + γ^k*V[t+k]
        """
        # 예외 처리: rewards가 비어있는 경우
        if not episode.rewards:
            return []

        T = len(episode.rewards)
        # n_max가 T보다 크면 T-1로 제한 (항상 Bootstrap 사용)
        n_max = min(self.config.n_target_value_bootstrap, T - 1) if T > 1 else 1
        gamma = self.config.value_bootstrap_discount

        # Values와 dones 길이를 rewards에 맞춤
        values = episode.values if hasattr(episode, "values") and episode.values else [0.0] * T
        if len(values) < T:
            values = list(values) + [0.0] * (T - len(values))

        dones = list(episode.dones) if hasattr(episode, "dones") and episode.dones else [False] * T
        if len(dones) < T:
            dones = dones + [False] * (T - len(dones))

        targets = []
        for t in range(T):
            target_value = 0.0

            # n-step discounted rewards 누적
            for k in range(n_max):
                idx = t + k
                if idx >= T:
                    break

                # Reward 누적
                target_value += (gamma ** k) * episode.rewards[idx]

                # Terminal 체크 (reward 누적 후)
                if dones[idx]:
                    break
            else: # break 없이 for 문이 정상 종료된 경우 실행됨
                # Bootstrap: Terminal 도달 안했을 때만 (for-else)
                bootstrap_idx = t + n_max
                if bootstrap_idx < T:
                    bootstrap_value = values[bootstrap_idx] if values[bootstrap_idx] is not None else 0.0
                    target_value += (gamma ** n_max) * bootstrap_value

            targets.append(target_value)

        return targets
    
    # def _infer_value_from_state(self, episode: Episode, idx: int) -> float:
    #     """주어진 상태에서 네트워크로 value 추론"""
    #     state_dict = episode.states[idx]
    #     obs = build_observation_tensor(state_dict, self.config.observation_shape, 
    #                                  self.device, self.config.use_obstacle_channels).unsqueeze(0)
        
    #     with torch.no_grad():
    #         value_logits, _, _, _ = self.network.initial_inference(obs)
            
    #         # Support distribution을 스칼라 값으로 변환
    #         support = torch.arange(-self.config.support_size, 
    #                               self.config.support_size + 1, 
    #                               device=self.device, dtype=torch.float32)
    #         prob = torch.softmax(value_logits, dim=1)
    #         value = torch.sum(prob * support, dim=1).item()
            
    #     return value 


def enrich_episode_with_mcts_parallel_worker(args) -> 'Episode':
    """병렬 처리용 Worker 함수 (모듈 레벨)

    Args:
        args: (episode_idx, network_state_dict, config, device, episode) 튜플
              - episode_idx: Episode 인덱스 (0~N-1)
              - network_state_dict: 학습된 네트워크 파라미터
              - config: MuZero 설정
              - device: 'cuda' or 'cpu'
              - episode: 처리할 Episode 객체

    Returns:
        MCTS enrichment가 완료된 Episode (실패 시 None)
    """
    episode_idx, network_state_dict, config, device, episode = args

    
    # GPU 분산: 여러 GPU가 있으면 episode_idx를 기반으로 자동 할당
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        num_gpus = min(torch.cuda.device_count(), config.use_gpu_device_max) if config.use_gpu_device_max is not None else torch.cuda.device_count()
        gpu_id = episode_idx % num_gpus
        device = f'cuda:{gpu_id}'

    # Progress Bar와 충돌하므로 개별 Episode 출력 제거
    # Network 재생성 (utils.get_network() 사용 - main_muzero.py와 동일)
    network = get_network(config, device)

    # 학습된 파라미터로 교체 (Iteration마다 업데이트된 파라미터)
    # state_dict의 텐서를 현재 device로 이동
    network.load_state_dict({
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in network_state_dict.items()
    })
    network.eval()

    # MCTS 재생성
    mcts = MCTS(
        network=network,
        action_space_size=len(config.action_space),
        num_simulations=config.num_simulations,
        mcts_value_backup_discount=config.mcts_value_backup_discount,
        puct_discount=config.puct_discount,
        c1=config.pb_c_init,
        c2=config.pb_c_base,
        device=device,
        config=config
    )
    
    # Simulated ego position 추적 (그리드 좌표)
    simulated_ego_row, simulated_ego_col = 49, 49  # 그리드 중앙에서 시작. self.observation_shape 의 중앙임

    # dones 필드 초기화 (충돌 tracking용)
    if not hasattr(episode, 'dones'):
        episode.dones = [False] * len(episode.states)

    # Episode 유효성 플래그
    episode_is_valid = True

    # 각 스텝에 대해 MCTS 실행
    for step_idx in range(len(episode.states)):
        # observation 가져오기 (dict 형태)
        state_dict = episode.states[step_idx]
        
        # observation tensor 추출
        if isinstance(state_dict, dict):
            # 현재 simulated_ego_pos 업데이트 후 shift 적용
            # import copy
            # state_dict_copy = copy.deepcopy(state_dict)
            # state_dict_copy['simulated_ego_pos'] = (simulated_ego_row, simulated_ego_col)
            episode.states[step_idx]['simulated_ego_pos'] = (simulated_ego_row, simulated_ego_col)

            # Ego 위치 기준으로 observation shift (처음에는 제자리임)
            # Episode 의 Step 들은 Root Step 의 위치를 공유함. 단, MCTS 시뮬레이션에 의해 선택된 Action 을 기반으로 중심점이 조정되도록 유도.
            episode.states[step_idx] = shift_observation(episode.states[0], (simulated_ego_row, simulated_ego_col))

            # 6채널 observation 생성 (공용 함수 사용) -> simulated_ego_pos 가 변해도(=step 이 진행되어도) 환경은 root 시점 그대로 고정됨.
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            observation_tensor = build_observation_tensor(episode.states[step_idx], config.observation_shape, device, config.use_obstacle_channels)

            # Legal mask 계산
            # Episode 의 첫 Step 위치가 장애물 위 일 경우, 모든 액션이 불법일 수 있음. 발생사례 있음. (=step_idx 가 0인 사례)
            legal_mask = calculate_legal_mask(
                episode.states[step_idx],
                (49, 49)
                # (simulated_ego_row, simulated_ego_col)
            )

            # for legal_mask_idx in range(len(legal_mask)):
            #     if legal_mask[legal_mask_idx] == 0.0:
            #         print(f"    - 스텝 {step_idx}에서 액션 {legal_mask_idx}는 불법입니다.")

            # MCTS 실행
            # - representation(): observation → latent
            # - 시뮬레이션: dynamics()로 미래 예측, prediction()으로 policy/value 계산
            # - 최적 action 선택 및 root node의 value/π 반환
            try:
                with torch.no_grad():
                    mcts_result = mcts.run(
                        observation_tensor,
                        add_exploration_noise=True,
                        legal_mask=legal_mask
                    )
            except IllegalStateException as e:
                # 모든 액션이 불법인 경우 Episode 즉시 종료
                print(f"  [Episode {episode_idx}] 스텝 {step_idx}에서 {str(e)}")
                # Episode를 현재 스텝 이전까지만 잘라냄
                # episode.states = episode.states[:step_idx]
                # episode.actions = episode.actions[:step_idx]
                # episode.rewards = episode.rewards[:step_idx]
                # episode.pi = episode.pi[:step_idx]
                # episode.values = episode.values[:step_idx]
                # episode.dones = episode.dones[:step_idx]
                # episode.legal_mask = episode.legal_mask[:step_idx]
                # if hasattr(episode, 'expert_ego_pos'):
                #     episode.expert_ego_pos = episode.expert_ego_pos[:step_idx]
                # 이 Episode는 너무 짧으면 버림
                # if len(episode.states) < 3:
                    # print(f"  [Episode {ep_idx + 1}] Episode가 너무 짧아서 버립니다. (길이: {len(episode.states)})")
                episode_is_valid = False
                # Episode 종료
                break

            # Episode에 MCTS 결과 업데이트
            episode.pi[step_idx] = mcts_result['policy_targets']
            # MCTS가 계산한 value 저장 (bootstrap에 사용됨)
            episode.values[step_idx] = mcts_result['root_value']
            # Legal mask 저장
            episode.legal_mask[step_idx] = legal_mask

            # MCTS가 선택한 action 저장
            selected_action = mcts_result['action']
            # selected_action = 2  # Action 고정 테스트 (2=POS2=전방/상단)
            episode.actions[step_idx] = selected_action
            
            # Action에 따라 simulated_ego_pos 업데이트 (최종 value 계산용)
            action = Action.from_index(selected_action)
            dx, dy = action.get_movement_vector()
            # 좌표계: x -> 상단으로 + / y -> 우측으로 +
            simulated_ego_row += dx  # 전방/후방
            simulated_ego_col += dy  # 좌/우 (environment.py와 일치)
            
            # Episode state에도 업데이트 (최종 거리 계산을 위해)
            episode.states[step_idx]['simulated_ego_pos'] = (simulated_ego_row, simulated_ego_col)

            # 디버깅: 데이터 타입 확인 (처음 몇 스텝만)
            # if step_idx < 2:
            #     obs_type = type(episode.states[step_idx].get('obstacles', None))
            #     print(f"    - 스텝 {step_idx} 후 obstacles 타입: {obs_type}")
            
            # GT 학습을 위한 환경 reward 계산 (Step마다)
            reward = calculate_target_reward(
                config, episode, step_idx, selected_action, 
                (simulated_ego_row, simulated_ego_col)
            )
            episode.rewards[step_idx] = reward
            
    # Episode가 유효한 경우만 추가
    if episode_is_valid:
        # n-step bootstrap target value 계산
        # MCTS root values는 보존하고 target_values는 별도로 계산됨
        episode.target_values = calculate_target_value(config, episode)
        return episode
        # enriched_episodes.append(episode)
    return None


def shift_observation(state_dict: dict, new_pos: Tuple[int, int]) -> dict:
    """observation 내의 그리드 채널들을 새로운 simulated_ego_pos에 맞게 쉬프트

    Args:
        state_dict: 원본 observation dictionary
        new_pos: 새로운 simulated_ego_pos (row, col)

    Returns:
        shifted_state_dict: 쉬프트된 observation dictionary
    """
    import copy

    # Deep copy로 원본 보호
    state_dict_copy = copy.deepcopy(state_dict)

    # Shift 계산: new_pos를 (49, 49)로 보내려면 얼마나 이동해야 하는지
    center = 49
    new_row, new_col = new_pos
    shift_row = center - new_row  # Ego를 중앙으로 이동시키는 shift
    shift_col = center - new_col

    # Shift가 0이면 그대로 반환
    if shift_row == 0 and shift_col == 0:
        return state_dict_copy

    # Channel별로 shift 적용 (slicing 기반, 경계는 0으로 채움)
    def apply_shift(data: np.ndarray, shift_r: int, shift_c: int) -> np.ndarray:
        """2D numpy array를 shift (circular shift 아님)"""
        H, W = data.shape
        shifted = np.zeros_like(data)

        # Source region (원본에서 복사할 영역)
        src_y_start = max(0, -shift_r)
        src_y_end = min(H, H - shift_r)
        src_x_start = max(0, -shift_c)
        src_x_end = min(W, W - shift_c)

        # Destination region (shifted 배열에 붙여넣을 영역)
        dst_y_start = max(0, shift_r)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        dst_x_start = max(0, shift_c)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)

        # Copy (나머지는 0으로 유지)
        shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
            data[src_y_start:src_y_end, src_x_start:src_x_end]

        return shifted

    # Channels 0, 1, 2 shift 적용
    for key in ['obstacles', 'static_environment', 'ground_class_id']:
        if key in state_dict_copy:
            data = state_dict_copy[key]

            # Torch Tensor를 numpy array로 변환
            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy()

            if isinstance(data, np.ndarray) and data.ndim == 2:
                state_dict_copy[key] = apply_shift(data, shift_row, shift_col)
            elif data is not None:
                print(f"    [WARNING] '{key}' shift 실패: type={type(data)}, shape={data.shape if hasattr(data, 'shape') else 'N/A'}")

    # simulated_ego_pos 업데이트 (항상 중앙으로)
    state_dict_copy['simulated_ego_pos'] = (center, center)

    return state_dict_copy



def calculate_legal_mask(state_dict: dict, current_pos: Tuple[int, int]) -> np.ndarray:
    """합법 액션 마스크 계산

    현재 위치에서 각 action을 취했을 때 충돌 여부를 체크하여
    합법 마스크 생성 (1=합법, 0=불법)

    Args:
        state_dict: observation dictionary (obstacles, static_environment 포함)
        current_pos: 현재 simulated ego 위치 (row, col)

    Returns:
        legal_mask: 9개 원소의 0/1 배열 (Action 개수만큼)
    """
    config = MuZeroConfig()
    legal_mask = np.ones(Action.get_action_space_size(), dtype=np.float32)

    current_row, current_col = current_pos
    boundary = config.obstacles_collision_boundary
    half_boundary = boundary // 2

    # 각 action에 대해 충돌 체크
    for action_idx in range(Action.get_action_space_size()):
        action = Action.from_index(action_idx)
        dx, dy = action.get_movement_vector()

        # Action 적용 후 위치
        next_row = current_row + dx
        next_col = current_col + dy

        # 그리드 경계 체크
        if next_row < 0 or next_row >= 99 or next_col < 0 or next_col >= 99:
            legal_mask[action_idx] = 0.0
            continue

        # 장애물 충돌 체크 (obstacles_collision_boundary 영역)
        has_collision = False
        for dr in range(-half_boundary, half_boundary + 1):
            for dc in range(-half_boundary, half_boundary + 1):
                check_row = next_row + dr
                check_col = next_col + dc

                # 그리드 범위 체크
                if 0 <= check_row < 99 and 0 <= check_col < 99:
                    # Channel 0: Dynamic objects 체크 (모든 값 > 0)
                    if 'obstacles' in state_dict and state_dict['obstacles'] is not None:
                        obs = state_dict['obstacles']
                        if hasattr(obs, 'shape') and len(obs.shape) >= 2:
                            try:
                                if isinstance(obs, torch.Tensor):
                                    val = obs[check_row, check_col].item()
                                else:
                                    val = obs[check_row, check_col]
                                if val > 0:
                                    has_collision = True
                                    break
                            except (IndexError, TypeError):
                                pass

                    # Channel 1: Static environment 체크
                    # Cliff, Static Obstacle, Missing Points만 충돌로 간주
                    if not has_collision and 'static_environment' in state_dict and state_dict['static_environment'] is not None:
                        static = state_dict['static_environment']
                        if hasattr(static, 'shape') and len(static.shape) >= 2:
                            try:
                                if isinstance(static, torch.Tensor):
                                    static_val = static[check_row, check_col].item()
                                else:
                                    static_val = static[check_row, check_col]

                                # Cliff, Static Obstacle, Missing Points 체크
                                if (abs(static_val - GRID_VALUE_CLIFF) < 0.01 or
                                    abs(static_val - GRID_VALUE_STATIC_OBSTACLE) < 0.01 or
                                    abs(static_val - GRID_VALUE_MISSING_POINTS) < 0.01):
                                    has_collision = True
                                    break
                            except (IndexError, TypeError):
                                pass

            if has_collision:
                break

        if has_collision:
            legal_mask[action_idx] = 0.0

    return legal_mask


def calculate_target_value(config, episode: Episode):
    """MuZero n-step bootstrap target value 계산

    target[t] = r[t] + γ*r[t+1] + ... + γ^(k-1)*r[t+k-1] + γ^k*V[t+k]
    """
    # 예외 처리: rewards가 비어있는 경우
    if not episode.rewards:
        return []

    T = len(episode.rewards)
    # n_max가 T보다 크면 T-1로 제한 (항상 Bootstrap 사용)
    n_max = min(config.n_target_value_bootstrap, T - 1) if T > 1 else 1
    gamma = config.value_bootstrap_discount

    # Values와 dones 길이를 rewards에 맞춤
    values = episode.values if hasattr(episode, "values") and episode.values else [0.0] * T
    if len(values) < T:
        values = list(values) + [0.0] * (T - len(values))

    dones = list(episode.dones) if hasattr(episode, "dones") and episode.dones else [False] * T
    if len(dones) < T:
        dones = dones + [False] * (T - len(dones))

    targets = []
    for t in range(T):
        target_value = 0.0

        # n-step discounted rewards 누적
        for k in range(n_max):
            idx = t + k
            if idx >= T:
                break

            # Reward 누적
            target_value += (gamma ** k) * episode.rewards[idx]

            # Terminal 체크 (reward 누적 후)
            if dones[idx]:
                break
        else: # break 없이 for 문이 정상 종료된 경우 실행됨
            # Bootstrap: Terminal 도달 안했을 때만 (for-else)
            bootstrap_idx = t + n_max
            if bootstrap_idx < T:
                bootstrap_value = values[bootstrap_idx] if values[bootstrap_idx] is not None else 0.0
                target_value += (gamma ** n_max) * bootstrap_value

        targets.append(target_value)

    return targets



def calculate_target_reward(config, episode: Episode, step_idx: int, action: int, 
                            sim_pos: Tuple[int, int]) -> float:
    """GT 학습을 위한 환경 reward 계산
    
    trajectory_following_value의 스텝별 버전
    
    Args:
        episode: 현재 Episode
        step_idx: 현재 스텝
        action: 선택된 action (미래 확장용)
        sim_pos: 시뮬레이션 위치 (미래 확장용)
        
    Returns:
        환경 reward (-1 ~ 1 범위)
    """
    # 예외처리 - step_idx가 episode.states 범위를 벗어나는 경우
    if step_idx >= len(episode.states):
        print(f"  - [Target Reward 계산 이상] 스텝 {step_idx}에서 episode.states가 부족합니다. value: 0.0")
        return 0.0
    
    # 목표: GT 만을 추종하는게 아닌, 다중 해를 허용하는 Target Reward 를 설계하는 것.

    # 1. 직진성 강화하는 reward (즉, Path 의 오실레이션이 최소화 되어야함)
    trajectory_smooth_reward = calculate_smoothness_reward(episode, step_idx)
    trajectory_smooth_reward = 0 # FIXME: 제거. 테스트용

    # 2. 이전 경로에서 Episode 의 끝 위치로 가까워지는 행동이면 +, 아니면 -
    # Potential-based reward shaping으로 목표 접근을 학습
    # trajectory_goal_follow_reward = self._calculate_goal_following_reward(episode, step_idx)
    trajectory_goal_follow_reward = calculate_goal_direction_reward(episode, step_idx)

    # 3. 장애물과 충돌 상황에 대한 패널티
    collision_reward = calculate_collision_reward(config, episode, step_idx)
    
    # # 합법 마스킹 적용된 상황에서는 절대 발생되면 안됨
    # if collision_reward == self.config.collision_penalty:
    #     # 충돌 발생 시 큰 페널티
    #     print(f"  - 충돌 발생! 스텝 {step_idx}에서 충돌 감지됨. 페널티: {self.config.collision_penalty}")
    #     return self.config.collision_penalty
    # else:
    #     # 충돌이 없으면 경로 추종 reward만 사용
    #     return trajectory_goal_follow_reward + trajectory_smooth_reward
    
    return trajectory_goal_follow_reward + trajectory_smooth_reward



def calculate_collision_reward(config, episode: Episode, step_idx: int) -> float:
    """충돌 기반 reward 계산
    
    특정 step_idx 시점에서 충돌 발생 여부 체크
    충돌 시 큰 페널티 부여
    """
    
    # step_idx 시점에서만 충돌 체크
    if step_idx < len(episode.states) and isinstance(episode.states[step_idx], dict):
        state = episode.states[step_idx]
        sim_pos = state.get('simulated_ego_pos', (49, 49))
        
        # 3x3 영역 체크 (중심 + 8방향)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                check_row = sim_pos[0] + dr
                check_col = sim_pos[1] + dc
                
                # 그리드 범위 체크
                if 0 <= check_row < 99 and 0 <= check_col < 99:
                    # Channel 0: Dynamic objects 체크
                    if 'obstacles' in state and state['obstacles'] is not None:
                        obs = state['obstacles']
                        # torch.Tensor와 numpy.ndarray 모두 처리
                        if hasattr(obs, 'shape') and len(obs.shape) >= 2:  # 2D tensor나 array인 경우
                            try:
                                if isinstance(obs, torch.Tensor):
                                    val = obs[check_row, check_col].item()
                                else:  # numpy array
                                    val = obs[check_row, check_col]
                                if val > 0:
                                    # 충돌 감지됨
                                    return config.collision_penalty  # 충돌 시 최대 페널티
                            except (IndexError, TypeError):
                                pass  # 인덱싱 실패 시 무시
                    
                    # Channel 1: Static environment 체크
                    if 'static_environment' in state and state['static_environment'] is not None:
                        static = state['static_environment']
                        # torch.Tensor와 numpy.ndarray 모두 처리
                        if hasattr(static, 'shape') and len(static.shape) >= 2:  # 2D tensor나 array인 경우
                            try:
                                if isinstance(static, torch.Tensor):
                                    val = static[check_row, check_col].item()
                                else:  # numpy array
                                    val = static[check_row, check_col]
                                if val > 0:
                                    # 충돌 감지됨
                                    return config.collision_penalty  # 충돌 시 최대 페널티
                            except (IndexError, TypeError):
                                pass  # 인덱싱 실패 시 무시
    
    return 0.0  # 충돌 없음
    
def calculate_smoothness_reward(episode: Episode, step_idx: int) -> float:
    """직진성 강화: 방향 일관성(코사인)
        반환: [-0.02, 0.02]"""
    if step_idx < 2:
        return 0.0

    st = episode.states
    for i in (step_idx-2, step_idx-1, step_idx):
        if i >= len(st) or not isinstance(st[i], dict):
            return 0.0
        if st[i].get('simulated_ego_pos') is None:
            return 0.0

    p0 = np.array(st[step_idx-2]['simulated_ego_pos'], dtype=float)
    p1 = np.array(st[step_idx-1]['simulated_ego_pos'], dtype=float)
    p2 = np.array(st[step_idx  ]['simulated_ego_pos'], dtype=float)

    v_prev = p1 - p0
    v_curr = p2 - p1
    prev_magnitude = np.linalg.norm(v_prev)
    curr_magnitude = np.linalg.norm(v_curr)

    # # 정체/미세 이동 페널티(단위-불변으로 너무 크게 주지 않음)
    # if prev_magnitude < 1e-3 or curr_magnitude < 1e-3:
    #     return -0.005

    cos = float(np.dot(v_prev, v_curr) / (prev_magnitude * curr_magnitude + 1e-8))

    # 직진이면 +, 방향 전환이면 - (EMA 제거)
    base = 0.02 * cos  # +0.02(동일방향) ~ -0.02(반대방향)
    return float(np.clip(base, -0.02, 0.02))

def calculate_goal_following_reward(episode: Episode, step_idx: int) -> float:
    """간단한 목표 추종 reward: 가까워지면 +0.1, 멀어지면 -0.1"""

    st = episode.states
    if step_idx >= len(st) or not hasattr(episode, 'expert_ego_pos'):
        return 0.0

    # 목표 위치
    goal = episode.expert_ego_pos[-1]

    # 이전 위치 (첫 스텝이면 시작점 49,49)
    if step_idx == 0:
        prev_pos = (49, 49)
    else:
        prev_pos = st[step_idx - 1].get('simulated_ego_pos')
        if prev_pos is None:
            return 0.0

    # 현재 위치
    curr_pos = st[step_idx].get('simulated_ego_pos')
    if curr_pos is None:
        return 0.0

    # 거리 계산 (유클리드)
    prev_dist = np.sqrt((prev_pos[0] - goal[0])**2 + (prev_pos[1] - goal[1])**2)
    curr_dist = np.sqrt((curr_pos[0] - goal[0])**2 + (curr_pos[1] - goal[1])**2)

    # 가까워지면 +0.1, 멀어지면 -0.1, 동일하면 0
    if curr_dist < prev_dist:
        return 0.1
    elif curr_dist > prev_dist:
        return -0.1
    else:
        return 0.0

def calculate_goal_direction_reward(episode: Episode, step_idx: int) -> float:
    """목표 방향과 이동 방향의 내적 기반 유사도 활용하는 reward"""

    st = episode.states
    # 현재 위치와 목표 위치
    curr_pos = st[step_idx].get('simulated_ego_pos')
    goal_pos = episode.expert_ego_pos[-1]
    # goal_pos = (10, 49); # FIXME: 의도적으로 아래로 가도록 유도 

    # 이전 위치 (이동 벡터 계산용)
    if step_idx == 0:
        prev_pos = (49, 49)
    else:
        prev_pos = st[step_idx - 1].get('simulated_ego_pos')
    
    init_pos = (49, 49)

    # 벡터 계산
    move_vector = np.array(curr_pos) - np.array(prev_pos)  # 실제 이동 방향
    goal_vector = np.array(goal_pos) - np.array(init_pos)  # 목표 방향

    # 내적으로 유사도 계산 (코사인 유사도)
    move_mag = np.linalg.norm(move_vector)
    goal_mag = np.linalg.norm(goal_vector)

    if move_mag < 1e-6 or goal_mag < 1e-6:
        return 0.0

    cos_similarity = np.dot(move_vector, goal_vector) / (move_mag * goal_mag)

    # 목표 방향이면 +0.1, 반대면 -0.1
    return 0.1 * cos_similarity  # [-0.1, +0.1] 범위
