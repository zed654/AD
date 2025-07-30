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
from .data_parser import build_observation_tensor


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
            observation_tensor = build_observation_tensor(state_dict, self.config.observation_shape, self.device)
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
                observation_tensor = build_observation_tensor(state_dict, self.config.observation_shape, self.device)
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
                    observation_tensor = build_observation_tensor(state_dict_copy, self.config.observation_shape, device)
                    
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
                    
                    # 보상은 dynamics 네트워크가 예측한 값을 사용해야 함
                    # GT Episode enrichment에서는 reward를 0으로 설정 (나중에 backward propagation으로 계산)
                    episode.rewards[step_idx] = 0.0
                
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
        - Miss Rate - 예측을 크게 빗나갈 확률 (Threshold 걸어넣고, 예측이 Threshold를 넘는 경우)
        """
        print("\nEpisode GameOver value 계산 중...")
        print(f"  - 총 스텝: {len(episode)}")
        
        # 1. 경로 추종 정확도 기반 value
        trajectory_value = self._calculate_trajectory_following_value(episode)
        
        # 2. 충돌 기반 value (TODO: 구현 예정)
        collision_value = self._calculate_collision_value(episode)
        
        # 3. 기타 조건들 (TODO: 필요시 추가)
        # goal_reaching_value = self._calculate_goal_reaching_value(episode)
        # smoothness_value = self._calculate_smoothness_value(episode)
        
        # 최종 value 계산 (가중 평균 또는 다른 방식)
        final_value = trajectory_value  # 현재는 경로 추종만 고려
        
        print(f"  - 최종 value: {final_value:.3f}")
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
        
        TODO: 구현 예정
        - Episode 내 충돌 발생 여부 체크
        - 충돌 시 큰 페널티 부여
        """
        # 현재는 기본값 반환
        return 0.0

    
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
        """n-step return 계산
    
        MuZero는 n-step return + bootstrap value를 사용합니다.
        target_value = sum(discount^k * reward_k) + discount^n * value_n
        bootstrap value 는 n_steps 만큼 적용됨 (n_steps 만큼 미래의 가치를 현재에 대입하는 것)
        
        * Bellman equation 원리임 - Baysian Theory 의 반대 개념으로, 미래의 가치를 현재에 대입하는 것
        """

        """Episode의 value를 backward propagation으로 계산
        
        V[t] = r[t] + γ * V[t+1]
        """
        for i in reversed(range(len(episode))):
            if i == len(episode) - 1:
                # 마지막 스텝은 final_value 사용
                episode.values[i] = final_value
            else:
                # 이전 스텝들은 reward + discounted next value
                episode.values[i] = episode.rewards[i] + self.config.discount * episode.values[i + 1] 