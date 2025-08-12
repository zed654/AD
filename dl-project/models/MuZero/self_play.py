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
        self.policies = []
        self.values = []
        
    def store(self, observation, action, reward, policy, value):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.policies.append(policy)
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
            discount=config.discount,
            c1=config.pb_c_init,
            c2=config.pb_c_base,
            device=device
        )
        
    def play_game(self, temperature: float = 1.0, render: bool = False) -> GameHistory:
        """하나의 완전한 게임을 플레이"""
        game_history = GameHistory()
        
        # 환경 초기화
        observation = self.environment.reset()
        
        # observation이 dict인 경우, 6채널 observation tensor 생성
        if isinstance(observation, dict):
            # obstacles는 이미 tensor이므로 numpy로 변환
            if hasattr(observation['obstacles'], 'numpy'):
                obstacles_np = observation['obstacles'].squeeze(0).numpy()  # 배치 차원 제거
            else:
                obstacles_np = observation['obstacles'].squeeze(0)
            
            # state_dict 구성
            state_dict = {
                'obstacles': obstacles_np,
                'static_environment': observation.get('static_environment'),
                'ground_class_id': observation.get('ground_class_id'),
                'near_destination_direction': observation.get('near_destination_direction'),
                'final_destination_direction': observation.get('final_destination_direction'),
                'simulated_ego_pos': observation.get('simulated_ego_pos', (49, 49))
            }
            
            # 6채널 observation tensor 생성
            observation_tensor = build_observation_tensor(state_dict, self.config.observation_shape, self.device, self.config.use_obstacle_channels)
        else:
            observation_tensor = observation
        
        # 게임이 끝날 때까지 플레이
        step = 0
        while not self.environment.is_terminal():
            if render:
                print(f"\n=== Step {step} ===")
            
            # MCTS 실행 - 시뮬레이션 통해 액션 선택
            mcts_result = self.mcts.run(
                observation_tensor,
                add_exploration_noise=(temperature > 0)
            )
            
            # 액션 선택
            if step < self.config.num_stochastic_steps:
                # 초기 스텝에서는 확률적 선택
                action = self._select_action_stochastic(
                    mcts_result['policy_targets'],
                    temperature
                )
            else:
                # 이후에는 결정적 선택
                action = mcts_result['action']
            
            # 액션 수행
            next_observation, reward, done, info = self.environment.step(action)
            
            # 충돌 발생 시 value 조정
            if done and info.get('done_reason') == 'collision':
                # 충돌 발생 시 해당 상태의 value를 -1로 설정
                adjusted_value = -1.0
            else:
                # 정상적인 경우 MCTS가 예측한 value 사용
                adjusted_value = mcts_result['root_value']
            
            # 히스토리 저장
            game_history.store(
                observation=observation,  # dict 형태로 저장
                action=action,
                reward=reward,
                policy=mcts_result['policy_targets'],
                value=adjusted_value
            )
            
            if render:
                action_obj = Action.from_index(action)
                print(f"Action: {action_obj}")
                print(f"Reward: {reward:.2f}")
                print(f"Value: {adjusted_value:.2f} (MCTS: {mcts_result['root_value']:.2f})")
                print(f"Done: {done}")
                if done:
                    print(f"Done reason: {info['done_reason']}")
            
            # 다음 상태로 전환
            observation = next_observation
            
            # observation이 dict인 경우, 6채널 observation tensor 생성
            if isinstance(observation, dict):
                # obstacles는 이미 tensor이므로 numpy로 변환
                if hasattr(observation['obstacles'], 'numpy'):
                    obstacles_np = observation['obstacles'].squeeze(0).numpy()  # 배치 차원 제거
                else:
                    obstacles_np = observation['obstacles'].squeeze(0)
                
                # state_dict 구성
                state_dict = {
                    'obstacles': obstacles_np,
                    'static_environment': observation.get('static_environment'),
                    'ground_class_id': observation.get('ground_class_id'),
                    'near_destination_direction': observation.get('near_destination_direction'),
                    'final_destination_direction': observation.get('final_destination_direction'),
                    'simulated_ego_pos': observation.get('simulated_ego_pos', (49, 49))
                }
                
                # 6채널 observation tensor 생성
                observation_tensor = build_observation_tensor(state_dict, self.config.observation_shape, self.device, self.config.use_obstacle_channels)
            else:
                observation_tensor = observation
                
            step += 1
            
            # 최대 스텝 제한 (안전장치)
            if step >= self.config.max_game_length:
                break
        
        # 충돌로 종료된 경우 마지막 value를 -1로 설정
        if info.get('done_reason') == 'collision' and len(game_history.values) > 0:
            game_history.values[-1] = -1.0
        
        if render:
            print(f"\n게임 종료! 총 {step} 스텝")
            print(f"종료 이유: {info.get('done_reason', 'unknown')}")
            print(f"총 보상: {sum(game_history.rewards):.2f}")
        
        return game_history
    
    def play_games(self, num_games: int, temperature: float = 1.0) -> List[GameHistory]:
        """여러 게임을 플레이"""
        games = []
        for i in range(num_games):
            print(f"게임 {i+1}/{num_games} 플레이 중...")
            game = self.play_game(temperature=temperature, render=False)
            games.append(game)
            
            # 종료 이유 확인
            done_reason = self.environment.done_reason
            print(f"  - 길이: {game.get_length()} 스텝")
            print(f"  - 총 보상: {sum(game.rewards):.2f}")
            print(f"  - 종료 이유: {done_reason}")
        
        return games
    
    def _select_action_stochastic(self, policy: np.ndarray, temperature: float) -> int:
        """정책에 따라 확률적으로 액션 선택"""
        if temperature == 0:
            return np.argmax(policy)
        
        # 온도 적용
        policy_temp = policy ** (1 / temperature)
        policy_temp = policy_temp / policy_temp.sum()
        
        # 확률적 선택
        return np.random.choice(len(policy), p=policy_temp)
    
    def collect_self_play_data(self, num_games: int, buffer_capacity: int = 1000) -> ReplayBuffer:
        """Self-play 데이터 수집"""
        replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        
        games = self.play_games(num_games)
        
        for game in games:
            replay_buffer.save_game(game)
        
        print(f"\nReplay Buffer 상태:")
        print(f"  - 저장된 에피소드: {replay_buffer.size()}")
        print(f"  - 총 누적 에피소드: {replay_buffer.total_episodes()}")
        
        return replay_buffer
    
    def enrich_episodes_with_mcts(self, episodes: List['Episode']) -> List['Episode']:
        """GT Episode에 MCTS 결과(policy, value) 추가
        
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
                    # - 최적 action 선택 및 root node의 value/policy 반환
                    with torch.no_grad():
                        mcts_result = self.mcts.run(
                            observation_tensor,
                            add_exploration_noise=False  # GT 데이터는 노이즈 없이
                        )
                    
                    # Episode에 MCTS 결과 업데이트
                    episode.policies[step_idx] = mcts_result['policy_targets']
                    # MCTS가 계산한 value 저장 (bootstrap에 사용됨)
                    episode.values[step_idx] = mcts_result['root_value']
                    
                    # MCTS가 선택한 action 저장
                    selected_action = mcts_result['action']
                    # selected_action = 6  # Action 고정 테스트 (2=POS2=전방/상단)
                    episode.actions[step_idx] = selected_action
                    
                    # Action에 따라 simulated_ego_pos 업데이트 (최종 value 계산용)
                    action = Action.from_index(selected_action)
                    dx, dy = action.get_movement_vector()
                    # 180도 회전 좌표계에서 그리드 좌표 업데이트:
                    # dy > 0 (상단) → row 증가 (전방으로 이동 = 아래로)
                    # dx > 0 (우측) → col 감소 (180도 회전으로 오른쪽이 왼쪽)
                    simulated_ego_row += dy  # 전방/후방
                    simulated_ego_col -= dx  # 좌/우 (environment.py와 일치)
                    
                    # Episode state에도 업데이트 (최종 거리 계산을 위해)
                    episode.states[step_idx]['simulated_ego_pos'] = (simulated_ego_row, simulated_ego_col)
                    
                    # 디버깅: 데이터 타입 확인 (처음 몇 스텝만)
                    if step_idx < 2:
                        obs_type = type(episode.states[step_idx].get('obstacles', None))
                        print(f"    - 스텝 {step_idx} 후 obstacles 타입: {obs_type}")
                    
                    # GT 학습을 위한 환경 reward 계산
                    reward = self._calculate_target_reward(
                        episode, step_idx, selected_action, 
                        (simulated_ego_row, simulated_ego_col)
                    )
                    episode.rewards[step_idx] = reward
                    
                    # 충돌 발생 시 Episode 종료
                    if reward == -1.0:
                        print(f"    - 충돌로 인해 Episode 종료 (스텝 {step_idx + 1}/{len(episode.states)})")
                        # 현재 스텝을 터미널로 표시 (잘라내기 전에 먼저 설정)
                        if step_idx < len(episode.dones):
                            episode.dones[step_idx] = True
                        # Episode 길이를 충돌 지점까지로 잘라냄
                        episode.states = episode.states[:step_idx + 1]
                        episode.rewards = episode.rewards[:step_idx + 1]
                        episode.policies = episode.policies[:step_idx + 1]
                        episode.values = episode.values[:step_idx + 1]
                        episode.actions = episode.actions[:step_idx + 1]
                        episode.dones = episode.dones[:step_idx + 1]
                        if hasattr(episode, 'expert_ego_pos'):
                            episode.expert_ego_pos = episode.expert_ego_pos[:step_idx + 1]
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
    
    
    def _calculate_trajectory_following_reward(self, episode: Episode, step_idx: int) -> float:
        """경로 추종 정확도 기반 value 계산
        
        특정 step_idx에서 simulated_ego_pos와 expert_ego_pos 간의 거리 오차를 기반으로 계산
        """
        # step_idx에 해당하는 스텝에서만 계산
        if step_idx < len(episode.states) and isinstance(episode.states[step_idx], dict):
            state = episode.states[step_idx]
            sim_pos = state.get('simulated_ego_pos')
            
            # expert_ego_pos 추출
            if hasattr(episode, 'expert_ego_pos') and step_idx < len(episode.expert_ego_pos):
                expert_pos = episode.expert_ego_pos[step_idx]
                
                if sim_pos is not None and expert_pos is not None:
                    # 둘 다 (row, col) 그리드 좌표
                    if isinstance(expert_pos, tuple) and len(expert_pos) == 2 and \
                       isinstance(sim_pos, tuple) and len(sim_pos) == 2:
                        expert_row, expert_col = expert_pos
                        sim_row, sim_col = sim_pos
                        
                        # 유클리드 거리 계산 (그리드 좌표 기준)
                        distance = np.sqrt((sim_row - expert_row)**2 + (sim_col - expert_col)**2)
                        
                        # 거리를 value로 변환하여 반환
                        return self._distance_to_value(distance)
        
        # 정보가 없으면 0.0 반환
        print(f"  - [Target Reward 계산 이상] 스텝 {step_idx}에서 simulated_ego_pos 또는 expert_ego_pos 정보가 없음. value: 0.0")
        return 0.0
    
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
                                        return -1.0  # 충돌 시 최대 페널티
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
                                        return -1.0  # 충돌 시 최대 페널티
                                except (IndexError, TypeError):
                                    pass  # 인덱싱 실패 시 무시
        
        return 0.0  # 충돌 없음

    def _calculate_missing_rate_value(self, episode: Episode) -> float:
        """Missing Rate 기반 value 계산
        
        주행로에서 많이 떨어질 경우, 큰 페널티 부여
        """
        threshold = 15  # 15 cells = 3m (각 셀은 20cm)
        miss_count = 0
        total_count = 0
        
        # 모든 스텝에서 거리 체크
        for step_idx in range(len(episode.states)):
            if step_idx < len(episode.states) and isinstance(episode.states[step_idx], dict):
                state = episode.states[step_idx]
                sim_pos = state.get('simulated_ego_pos')
                
                # expert_ego_pos 추출
                if hasattr(episode, 'expert_ego_pos') and step_idx < len(episode.expert_ego_pos):
                    expert_pos = episode.expert_ego_pos[step_idx]
                    
                    if sim_pos is not None and expert_pos is not None:
                        # 유클리드 거리 계산 (그리드 좌표 기준)
                        if isinstance(expert_pos, tuple) and len(expert_pos) == 2 and \
                           isinstance(sim_pos, tuple) and len(sim_pos) == 2:
                            distance = np.sqrt((sim_pos[0] - expert_pos[0])**2 + 
                                             (sim_pos[1] - expert_pos[1])**2)
                            
                            total_count += 1
                            if distance > threshold:
                                miss_count += 1
        
        # Miss Rate 계산
        if total_count > 0:
            miss_rate = miss_count / total_count
            print(f"    - Miss Rate: {miss_rate:.2%} ({miss_count}/{total_count} 스텝이 3m 이상 벗어남)")
            
            # Miss Rate를 value로 변환 (0% -> 0.0, 100% -> -1.0)
            return -miss_rate  # 벗어난 비율만큼 페널티
        else:
            return 0.0

    
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
        
        # 1. 경로 추종 정확도 기반 reward
        # - 현재 simulated_ego_pos와 expert_ego_pos 간의 거리 오차 계산
        trajectory_reward = self._calculate_trajectory_following_reward(episode, step_idx)
        
        # 2. 충돌 기반 value
        collision_reward = self._calculate_collision_reward(episode, step_idx)

        if collision_reward == -1.0:
            # 충돌 발생 시 큰 페널티
            print(f"  - 충돌 발생! 스텝 {step_idx}에서 충돌 감지됨. 페널티: -1.0")
            return -1.0
        else:
            # 충돌이 없으면 경로 추종 reward만 사용
            return trajectory_reward
        
        # return np.clip(reward, -1.0, 1.0)
    
    def _calculate_grid_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """두 그리드 좌표 간 유클리드 거리 계산"""
        if pos1 is None or pos2 is None:
            return float('inf')
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _distance_to_value(self, distance: float) -> float:
        """거리 오차를 value로 변환 (continuous mapping)
        
        Args:
            distance: 평균 거리 오차 (그리드 셀 단위)
            
        Returns:
            value: -1.0 ~ 1.0 범위의 값
            - 0 거리 → 1.0
            - 10 거리 → 0.0  
            - 20 거리 → -1.0
        """
        value = 1.0 - (distance / 10.0)
        return max(-1.0, min(1.0, value))  # -1.0 ~ 1.0 범위로 제한
    
    def _calculate_target_value(self, episode: Episode):
        """
        MuZero n-step targets (정석)
        z_t = sum_{k=0}^{steps-1} γ^k r_{t+k} + γ^{steps} * V(s_{t+steps})
        - 여기서 steps = min(n, 터미널 도달 전까지의 실제 합산 스텝 수)
        - dones[j]==True면 그 전이에서 종료(부트스트랩 금지)
        - 예측 로그(root values)와 타깃은 분리 저장
        """
        n = int(self.config.unroll_steps)
        gamma = float(self.config.discount)

        rewards = episode.rewards
        T = len(rewards)
        
        # dones 필드 처리 - 길이를 rewards와 맞춤
        if hasattr(episode, "dones") and episode.dones is not None:
            dones = episode.dones
            # dones 길이가 rewards보다 짧으면 False로 채움
            if len(dones) < T:
                dones = list(dones) + [False] * (T - len(dones))
        else:
            dones = [False] * T

        # 예측 로그(root values) 소스: root_values > values > None
        root_values = getattr(episode, "root_values", None)
        if root_values is None:
            root_values = getattr(episode, "values", None)
        if root_values is None:
            root_values = [None] * T

        targets = [0.0] * T

        for t in range(T):
            G = 0.0
            discount = 1.0
            steps = 0
            hit_terminal = False

            # n-step 누적 (조기 종료)
            while steps < n and (t + steps) < T:
                j = t + steps
                G += discount * float(rewards[j])

                if dones[j]:                 # 전이 (s_j,a_j)->s_{j+1} 직후 터미널
                    hit_terminal = True
                    break                     # 부트스트랩 금지

                discount *= gamma
                steps += 1

            if hit_terminal:
                targets[t] = G
                continue

            # 부트스트랩: 비터미널인 경우에만, s_{t+steps}
            j_boot = t + steps
            if j_boot < T:
                v_boot = root_values[j_boot]
                if v_boot is None:
                    v_boot = self._infer_value_from_state(episode, j_boot)
                G += discount * float(v_boot)

            targets[t] = G

        # 예측 로그를 보존하고 타깃은 별도 필드에 저장
        # episode.target_values = targets
        return targets
    
    def _infer_value_from_state(self, episode: Episode, idx: int) -> float:
        """주어진 상태에서 네트워크로 value 추론"""
        state_dict = episode.states[idx]
        obs = build_observation_tensor(state_dict, self.config.observation_shape, 
                                     self.device, self.config.use_obstacle_channels).unsqueeze(0)
        
        with torch.no_grad():
            value_logits, _, _, _ = self.network.initial_inference(obs)
            
            # Support distribution을 스칼라 값으로 변환
            support = torch.arange(-self.config.support_size, 
                                  self.config.support_size + 1, 
                                  device=self.device, dtype=torch.float32)
            prob = torch.softmax(value_logits, dim=1)
            value = torch.sum(prob * support, dim=1).item()
            
        return value 