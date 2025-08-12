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
            
            # 각 스텝에 대해 MCTS 실행
            for step_idx in range(len(episode)):
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
                    episode.actions[step_idx] = selected_action
                    
                    # Action에 따라 simulated_ego_pos 업데이트 (최종 value 계산용)
                    action = Action.from_index(selected_action)
                    dx, dy = action.get_movement_vector()
                    # 180도 회전 좌표계에서 그리드 좌표 업데이트:
                    # dy > 0 (상단) → row 증가 (전방)
                    # dx > 0 (우측) → col 감소 (좌측)
                    simulated_ego_row += dy  # 전방/후방
                    simulated_ego_col -= dx  # 좌/우
                    
                    # Episode state에도 업데이트 (최종 거리 계산을 위해)
                    episode.states[step_idx]['simulated_ego_pos'] = (simulated_ego_row, simulated_ego_col)
                    
                    # GT 학습을 위한 환경 reward 계산
                    # TODO: 실제 환경 보상 정의 필요
                    episode.rewards[step_idx] = self._calculate_target_reward(
                        episode, step_idx, selected_action, 
                        (simulated_ego_row, simulated_ego_col)
                    )
                
                if step_idx % 10 == 0:
                    print(f"  - 진행률: {step_idx}/{len(episode)} 스텝 (각 스텝마다 {self.config.num_simulations}회 시뮬레이션)")
            
            # Value propagation 추가
            # 1. 모든 value를 0으로 초기화
            for i in range(len(episode)):
                episode.values[i] = 0.0
            
            # 2. GameOver 조건에 따른 최종 value 설정
            final_value = self._get_episode_game_over_value(episode)
            
            # 3. Backward value propagation
            self._calculate_target_value(episode, final_value)
            
            enriched_episodes.append(episode)
            print(f"  - Episode {ep_idx + 1} 완료: {len(episode)} 스텝, "
                  f"평균 value: {np.mean(episode.values):.3f}")
        
        return enriched_episodes
    
    def _get_episode_game_over_value(self, episode: Episode) -> float:
        """Episode의 GameOver 조건에 따른 최종 value 반환
        
        다양한 요소들을 고려하여 최종 value 계산:
        - Collision Rate - 충돌 여부
        - minADE - 경로 추종 정확도 (expert와의 거리 오차)
        - minFDE - 예측 경로 중, 예측의 최종 도착지점과 실제 도착지점의 차이
        - Missing Rate - 예측을 크게 빗나갈 확률 (Threshold 걸어넣고, 예측이 Threshold를 넘는 경우)
        """
        print("\nEpisode GameOver value 계산 중...")
        print(f"  - 총 스텝: {len(episode)}")
        
        # 1. 경로 추종 정확도 기반 value
        trajectory_value = self._calculate_trajectory_following_value(episode)
        
        # 2. 충돌 기반 value
        collision_value = self._calculate_collision_value(episode)
        
        # 3. Missing Rate 기반 value
        # missing_rate_value = self._calculate_missing_rate_value(episode)
        
        # 4. 기타 조건들 (TODO: 필요시 추가)
        # goal_reaching_value = self._calculate_goal_reaching_value(episode)
        # smoothness_value = self._calculate_smoothness_value(episode)
        
        # 최종 value 계산
        # 충돌이 발생하면 즉시 -1.0 반환
        if collision_value < 0:
            print("  - 충돌 발생! 최종 value: -1.0")
            return -1.0
        
        # 충돌이 없으면 다른 값들을 조합
        # trajectory_value: -1.0 ~ 1.0 (경로 추종 정확도)
        # missing_rate_value: -1.0 ~ 0.0 (벗어난 비율의 음수)
        # final_value = trajectory_value * 0.7 + missing_rate_value * 0.3  # 가중 평균
        final_value = trajectory_value
        
        print(f"  - 최종 value: {final_value:.3f} (경로추종: {trajectory_value:.3f}, Missing: {missing_rate_value:.3f})")
        
        return final_value
    
    def _calculate_trajectory_following_value(self, episode: Episode) -> float:
        """경로 추종 정확도 기반 value 계산
        
        simulated_ego_pos와 expert_ego_pos 간의 평균 거리 오차를 기반으로 계산
        """
        total_distance_error = 0.0
        num_comparisons = 0
        
        for i in range(len(episode)):
            # state에서 simulated_ego_pos 추출
            if i < len(episode.states) and isinstance(episode.states[i], dict):
                state = episode.states[i]
                sim_pos = state.get('simulated_ego_pos')
                
                # expert_ego_pos 추출
                if hasattr(episode, 'expert_ego_pos') and i < len(episode.expert_ego_pos):
                    expert_pos = episode.expert_ego_pos[i]
                    
                    if sim_pos is not None and expert_pos is not None:
                        # 둘 다 (row, col) 그리드 좌표
                        if isinstance(expert_pos, tuple) and len(expert_pos) == 2 and \
                           isinstance(sim_pos, tuple) and len(sim_pos) == 2:
                            expert_row, expert_col = expert_pos
                            sim_row, sim_col = sim_pos
                            
                            # 유클리드 거리 계산 (그리드 좌표 기준)
                            distance = np.sqrt((sim_row - expert_row)**2 + (sim_col - expert_col)**2)
                            total_distance_error += distance
                            num_comparisons += 1
        
        # 평균 거리 오차 계산
        if num_comparisons > 0:
            avg_distance_error = total_distance_error / num_comparisons
            print(f"  - 평균 거리 오차: {avg_distance_error:.2f} 그리드 셀")
            return self._distance_to_value(avg_distance_error)
        else:
            print("  - Expert 위치 정보 없음")
            return 0.0
    
    def _calculate_collision_value(self, episode: Episode) -> float:
        """충돌 기반 value 계산
        
        Episode 내 충돌 발생 여부 체크
        충돌 시 큰 페널티 부여
        """
        # 모든 스텝에서 충돌 체크
        for step_idx in range(len(episode)):
            if step_idx < len(episode.states) and isinstance(episode.states[step_idx], dict):
                state = episode.states[step_idx]
                sim_pos = state.get('simulated_ego_pos', (49, 49))
                
                # 3x3 영역 체크 (중심 + 8방향)
                collision_detected = False
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        check_row = sim_pos[0] + dr
                        check_col = sim_pos[1] + dc
                        
                        # 그리드 범위 체크
                        if 0 <= check_row < 99 and 0 <= check_col < 99:
                            # Channel 0: Dynamic objects 체크
                            if 'obstacles' in state and state['obstacles'] is not None:
                                if isinstance(state['obstacles'], np.ndarray):
                                    if state['obstacles'][check_row, check_col] > 0:
                                        collision_detected = True
                                        print(f"    - 충돌 감지! 스텝 {step_idx}에서 위치 ({check_row}, {check_col}) ; 대상객체: Obstacles Tracking")
                                        break
                            
                            # Channel 1: Static environment 체크
                            if 'static_environment' in state and state['static_environment'] is not None:
                                if isinstance(state['static_environment'], np.ndarray):
                                    if state['static_environment'][check_row, check_col] > 0:
                                        collision_detected = True
                                        print(f"    - 충돌 감지! 스텝 {step_idx}에서 위치 ({check_row}, {check_col}) ; 대상객체: Static Environment")
                                        break
                    
                    if collision_detected:
                        break
                
                if collision_detected:
                    print(f"    - 충돌 감지! 스텝 {step_idx}에서 위치 ({sim_pos[0]}, {sim_pos[1]})")
                    return -1.0  # 충돌 시 최대 페널티
        
        return 0.0  # 충돌 없음

    def _calculate_missing_rate_value(self, episode: Episode) -> float:
        """Missing Rate 기반 value 계산
        
        주행로에서 많이 떨어질 경우, 큰 페널티 부여
        """
        threshold = 15  # 15 cells = 3m (각 셀은 20cm)
        miss_count = 0
        total_count = 0
        
        # 모든 스텝에서 거리 체크
        for step_idx in range(len(episode)):
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
            action: 선택된 action (현재 미사용)
            sim_pos: 시뮬레이션 위치 (row, col)
            
        Returns:
            환경 reward (-1 ~ 1 범위)
        """
        reward = 0.0
        
        # Expert following reward 계산 (trajectory_following 로직 응용)
        if hasattr(episode, 'expert_ego_pos') and step_idx < len(episode.expert_ego_pos):
            expert_pos = episode.expert_ego_pos[step_idx]
            
            # 현재 거리 계산
            curr_dist = self._calculate_grid_distance(sim_pos, expert_pos)
            
            # Step-wise reward: 거리에 따른 즉각적인 피드백
            # 가까울수록 positive (최대 0.2), 멀수록 negative (최소 -0.2)
            distance_reward = 0.2 * (1.0 - min(curr_dist / 20.0, 2.0))  # 20 cells = neutral
            reward += distance_reward
            
            # 이전 스텝과의 개선도 보너스
            if step_idx > 0:
                prev_sim_pos = episode.states[step_idx - 1].get('simulated_ego_pos', (49, 49))
                prev_expert_pos = episode.expert_ego_pos[step_idx - 1]
                prev_dist = self._calculate_grid_distance(prev_sim_pos, prev_expert_pos)
                
                # 거리가 줄어들면 추가 보너스
                improvement = prev_dist - curr_dist
                reward += improvement * 0.05  # 작은 보너스
        
        # 경계 체크 페널티 (그리드 밖)
        if sim_pos[0] < 5 or sim_pos[0] >= 94 or sim_pos[1] < 5 or sim_pos[1] >= 94:
            reward -= 0.3  # 경계 근처 페널티
        
        # Reward를 -1 ~ 1 범위로 클리핑
        return np.clip(reward, -1.0, 1.0)
    
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
    
    def _calculate_target_value(self, episode: Episode, final_value: float):
        """
        n-step return + bootstrap value (MuZero target)
        z_t^{(n)} = sum_{k=0}^{n-1} γ^k r_{t+k} + γ^n * v0(s_{t+n})
        터미널(j_boot >= T)이면:
        - terminal_bootstrap == "none": 부트스트랩 생략
        - terminal_bootstrap == "final_value": γ^{steps_added} * final_value 추가
        """
        n = self.config.unroll_steps
        gamma = self.config.discount
        terminal_bootstrap = getattr(self.config, "terminal_bootstrap", "final_value")  # "none" or "final_value"

        # 원본 MCTS value 보존(있으면 사용, 없으면 네트워크 추론)
        mcts_values = episode.values.copy()

        T = len(episode)
        target_values = [0.0] * T

        for i in range(T):
            G = 0.0
            gpow = 1.0
            steps_added = 0

            # n-step 보상 합
            for k in range(n):
                j = i + k
                if j >= T:
                    break
                G += gpow * float(episode.rewards[j])
                gpow *= gamma
                steps_added += 1
                # dones가 있다면 여기서 끊기: if episode.dones[j]: break

            # 부트스트랩
            j_boot = i + n
            if j_boot < T:
                v_boot = mcts_values[j_boot]
                if v_boot is None:
                    v_boot = self._infer_value_from_state(episode, j_boot)  # v0(s_{i+n})
                G += gpow * float(v_boot)          # gpow == γ^{steps_added}
            else:
                # 터미널 도달: 정책 선택
                if terminal_bootstrap == "final_value":
                    G += gpow * float(final_value)  # γ^{steps_added} * final_value
                # "none"이면 아무것도 더하지 않음(= MC 합만)

            target_values[i] = float(G)

        # 학습 타깃 기록
        for i in range(T):
            episode.values[i] = target_values[i]
    
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