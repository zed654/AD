"""
Self-play 관리자
MuZero의 Self-play 게임을 실행하고 데이터를 수집
"""
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from .mcts import MCTS
from .environment import BaseEnvironment
from .action import Action
from .replay_buffer import ReplayBuffer, Episode
from .utils import build_observation_tensor


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
        
    def enrich_episodes_with_mcts(self, episodes: List['Episode']) -> List['Episode']:
        # MCTS Simulation 게임 수행되는 곳
        """GT Episode에 MCTS 결과(π, value) 추가
        
        Args:
            episodes: GT 데이터로부터 생성된 Episode 리스트
            
        Returns:
            MCTS 결과가 추가된 Episode 리스트
        """
        enriched_episodes = []
        
        for ep_idx, episode in enumerate(episodes):
            print(f"\nEpisode {ep_idx + 1}/{len(episodes)}에 MCTS 적용 중...")
            print(f"  - MCTS 시뮬레이션 횟수: {self.config.num_simulations}회")
            
            # Simulated ego position 추적 (그리드 좌표)
            simulated_ego_row, simulated_ego_col = 49, 49  # 그리드 중앙에서 시작. self.observation_shape 의 중앙임
            
            # dones 필드 초기화 (충돌 tracking용)
            if not hasattr(episode, 'dones'):
                episode.dones = [False] * len(episode.states)
            
            # 각 스텝에 대해 MCTS 실행
            for step_idx in range(len(episode.states)):
                # observation 가져오기 (dict 형태)
                state_dict = episode.states[step_idx]
                
                # observation tensor 추출
                if isinstance(state_dict, dict):
                    # 현재 simulated_ego_pos 업데이트
                    state_dict_copy = state_dict.copy()
                    state_dict_copy['simulated_ego_pos'] = (simulated_ego_row, simulated_ego_col)
                    
                    # 6채널 observation 생성 (공용 함수 사용)
                    device = self.device if hasattr(self, 'device') else 'cpu'
                    observation_tensor = build_observation_tensor(state_dict_copy, self.config.observation_shape, device, self.config.use_obstacle_channels)
                    
                    # MCTS 실행
                    # - representation(): observation → latent
                    # - 시뮬레이션: dynamics()로 미래 예측, prediction()으로 policy/value 계산
                    # - 최적 action 선택 및 root node의 value/π 반환
                    with torch.no_grad():
                        mcts_result = self.mcts.run(
                            observation_tensor,
                            add_exploration_noise=True
                        )
                    
                    # Episode에 MCTS 결과 업데이트
                    episode.pi[step_idx] = mcts_result['policy_targets']
                    # MCTS가 계산한 value 저장 (bootstrap에 사용됨)
                    episode.values[step_idx] = mcts_result['root_value']
                    
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
                    if step_idx < 2:
                        obs_type = type(episode.states[step_idx].get('obstacles', None))
                        print(f"    - 스텝 {step_idx} 후 obstacles 타입: {obs_type}")
                    
                    # GT 학습을 위한 환경 reward 계산 (Step마다)
                    reward = self._calculate_target_reward(
                        episode, step_idx, selected_action, 
                        (simulated_ego_row, simulated_ego_col)
                    )
                    episode.rewards[step_idx] = reward
                    
                    # 충돌 발생 시 Episode 종료
                    if reward == self.config.collision_penalty:
                        print(f"    - 충돌로 인해 Episode 종료 (스텝 {step_idx + 1}/{len(episode.states)})")
                        # 현재 스텝을 터미널로 표시 (잘라내기 전에 먼저 설정)
                        if step_idx < len(episode.dones):
                            episode.dones[step_idx] = True
                        # Episode 길이를 충돌 지점까지로 잘라냄
                        episode.states = episode.states[:step_idx + 1]
                        episode.rewards = episode.rewards[:step_idx + 1]
                        episode.pi = episode.pi[:step_idx + 1]
                        episode.values = episode.values[:step_idx + 1]
                        episode.actions = episode.actions[:step_idx + 1]
                        episode.dones = episode.dones[:step_idx + 1]
                        # 정답 위치는 유지. Target Reward 의 목표로 활용해야함
                        # if hasattr(episode, 'expert_ego_pos'):
                        #     episode.expert_ego_pos = episode.expert_ego_pos[:step_idx + 1]
                        break  # MCTS 적용 루프 종료
                
                if step_idx % 10 == 0:
                    print(f"  - 진행률: {step_idx}/{len(episode.states)} 스텝 (각 스텝마다 {self.config.num_simulations}회 시뮬레이션)")
            
            # n-step bootstrap target value 계산
            # MCTS root values는 보존하고 target_values는 별도로 계산됨
            episode.target_values = self._calculate_target_value(episode)
            
            enriched_episodes.append(episode)
            # target_values 출력 (학습에 사용되는 실제 타깃)
            avg_target_value = np.mean(episode.target_values) if hasattr(episode, 'target_values') else np.mean(episode.values)
            print(f"  - Episode {ep_idx + 1} 완료: {len(episode.states)} 스텝, "
                  f"평균 target value: {avg_target_value:.3f}")
        
        return enriched_episodes
    
    
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
        
        if collision_reward == self.config.collision_penalty:
            # 충돌 발생 시 큰 페널티
            print(f"  - 충돌 발생! 스텝 {step_idx}에서 충돌 감지됨. 페널티: {self.config.collision_penalty}")
            return self.config.collision_penalty
        else:
            # 충돌이 없으면 경로 추종 reward만 사용
            return trajectory_goal_follow_reward + trajectory_smooth_reward
        
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