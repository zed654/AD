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
        
        # observation이 dict인 경우, 'obstacles' tensor만 추출
        if isinstance(observation, dict):
            observation_tensor = observation['obstacles']
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
            
            # observation이 dict인 경우, 'obstacles' tensor만 추출
            if isinstance(observation, dict):
                observation_tensor = observation['obstacles']
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
            
            # 각 스텝에 대해 MCTS 실행
            for step_idx in range(len(episode)):
                # observation 가져오기 (dict 형태)
                state_dict = episode.states[step_idx]
                
                # observation tensor 추출
                if isinstance(state_dict, dict):
                    # 5채널 observation 생성 (공용 함수 사용)
                    device = self.device if hasattr(self, 'device') else 'cpu'
                    observation_tensor = build_observation_tensor(state_dict, self.config.observation_shape, device) # train.py 에서도 사용되서, 함수화했음

                    
                    # MCTS 실행하여 policy와 value 계산
                    with torch.no_grad():
                        mcts_result = self.mcts.run(
                            observation_tensor,
                            add_exploration_noise=False  # GT 데이터는 노이즈 없이
                        )
                    
                    # Episode에 MCTS 결과 업데이트
                    episode.policies[step_idx] = mcts_result['policy_targets']
                    episode.values[step_idx] = mcts_result['root_value']
                    
                    # 보상 계산 (GT에서는 목표 도달 거리 기반)
                    if hasattr(episode, 'expert_ego_pos') and step_idx < len(episode.expert_ego_pos) - 1:
                        # 현재와 다음 스텝의 expert 위치
                        if episode.expert_ego_pos[step_idx] is not None:
                            current_expert_pos = episode.expert_ego_pos[step_idx]
                            
                            # 이전 위치와의 거리 계산
                            if step_idx > 0 and episode.expert_ego_pos[step_idx - 1] is not None:
                                prev_expert_pos = episode.expert_ego_pos[step_idx - 1]
                                distance = np.sqrt((current_expert_pos[0] - prev_expert_pos[0])**2 + 
                                                 (current_expert_pos[1] - prev_expert_pos[1])**2)
                                # 이동했으면 양의 보상, 정지했으면 음의 보상
                                episode.rewards[step_idx] = distance / 10.0 - 0.1 if distance > 0.5 else -0.1
                            else:
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
        
        Action에 따른 이동 경로와 expert 경로의 거리 오차를 기반으로 계산
        """
        print("\nEpisode GameOver value 계산 중...")
        print(f"  - 총 스텝: {len(episode)}")
        
        # self.cell_size = 0.2  # 각 그리드 셀의 크기 (m)

        # Action 기반 Ego 위치 계산
        ego_row, ego_col = 49, 49  # 시작 위치 (그리드 중앙)
        action_positions = [(ego_row, ego_col)]  # 시작 위치 포함
        
        # 각 action에 따라 위치 업데이트
        for i in range(len(episode)):
            if i < len(episode.actions):
                action_idx = episode.actions[i]
                action = Action.from_index(action_idx)
                
                # Action에 따른 이동 벡터 가져오기
                move_vector = action.get_movement_vector()
                
                # 위치 업데이트 (주의: 좌표계가 180도 회전됨)
                # 원래: 상단이 -row, 우측이 +col
                # 180도 회전: 상단이 +row, 우측이 -col
                ego_row += move_vector[1]  # y축 이동 (상하)
                ego_col -= move_vector[0]  # x축 이동 (좌우, 부호 반전)
                
                action_positions.append((ego_row, ego_col))
        
        # Expert와 Action 경로의 평균 거리 오차 계산
        avg_distance_error = self._calculate_trajectory_distance_error(episode, action_positions)
        
        if avg_distance_error is not None:
            print(f"  - 평균 거리 오차: {avg_distance_error:.2f} 그리드 셀")
            # 거리 오차를 value로 변환
            final_value = self._distance_to_value(avg_distance_error)
            print(f"  - 최종 value: {final_value:.3f}")
        else:
            print("  - Expert 위치 정보 없음")
            final_value = 0.0
        
        return final_value
    
    def _calculate_trajectory_distance_error(self, episode: Episode, action_positions: List[Tuple[int, int]]) -> Optional[float]:
        """Expert 경로와 Action 경로 간의 평균 거리 오차 계산
        
        Args:
            episode: GT Episode (expert_ego_pos 포함)
            action_positions: Action으로 계산한 위치들
            
        Returns:
            평균 거리 오차 (그리드 셀 단위) 또는 None
        """
        total_distance_error = 0.0
        num_comparisons = 0
        
        for i in range(len(episode)):
            if i < len(action_positions) and hasattr(episode, 'expert_ego_pos') and i < len(episode.expert_ego_pos):
                if episode.expert_ego_pos[i] is not None:
                    # Action 기반 위치
                    action_pos = action_positions[i]
                    
                    # Expert 위치 (그리드 좌표)
                    expert_pos = episode.expert_ego_pos[i]
                    
                    # 유클리드 거리 계산
                    distance = np.sqrt((action_pos[0] - expert_pos[0])**2 + 
                                     (action_pos[1] - expert_pos[1])**2)
                    
                    total_distance_error += distance
                    num_comparisons += 1
        
        # 평균 거리 오차 계산
        if num_comparisons > 0:
            return total_distance_error / num_comparisons
        else:
            return None
    
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