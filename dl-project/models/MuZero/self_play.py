"""
Self-play 관리자
MuZero의 Self-play 게임을 실행하고 데이터를 수집
"""
import torch
import numpy as np
from typing import List, Dict, Optional
from .mcts import MCTS
from .environment import BaseEnvironment
from .action import Action
from .replay_buffer import ReplayBuffer, Episode


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
            
            # 각 스텝에 대해 MCTS 실행
            for step_idx in range(len(episode)):
                # observation 가져오기 (dict 형태)
                state_dict = episode.states[step_idx]
                
                # observation tensor 추출
                if isinstance(state_dict, dict) and 'obstacles' in state_dict:
                    observation_tensor = state_dict['obstacles']
                    
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
                    print(f"  - 진행률: {step_idx}/{len(episode)} 스텝")
            
            enriched_episodes.append(episode)
            print(f"  - Episode {ep_idx + 1} 완료: {len(episode)} 스텝, "
                  f"평균 value: {np.mean(episode.values):.3f}")
        
        return enriched_episodes 