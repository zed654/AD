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
from .replay_buffer import ReplayBuffer


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
        
        # 게임이 끝날 때까지 플레이
        step = 0
        while not self.environment.is_terminal():
            if render:
                print(f"\n=== Step {step} ===")
            
            # MCTS 실행 - 시뮬레이션 통해 액션 선택
            mcts_result = self.mcts.run(
                observation,
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
            
            # 히스토리 저장
            game_history.store(
                observation=observation.cpu(),
                action=action,
                reward=reward,
                policy=mcts_result['policy_targets'],
                value=mcts_result['root_value']
            )
            
            if render:
                action_obj = Action.from_index(action)
                print(f"Action: {action_obj}")
                print(f"Reward: {reward:.2f}")
                print(f"Value: {mcts_result['root_value']:.2f}")
                print(f"Done: {done}")
                if done:
                    print(f"Done reason: {info['done_reason']}")
            
            # 다음 상태로 전환
            observation = next_observation
            step += 1
            
            # 최대 스텝 제한 (안전장치)
            if step >= self.config.max_game_length:
                break
        
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