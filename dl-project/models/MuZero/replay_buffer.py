"""
MuZero Replay Buffer
경험을 저장하고 학습을 위한 샘플을 제공
"""
import numpy as np
from typing import List, Optional


class Episode:
    """하나의 에피소드(게임) 데이터를 저장"""
    
    def __init__(self):
        self.states = []      # 상태 목록
        self.actions = []     # 액션 목록  
        self.rewards = []     # 보상 목록
        self.policies = []    # MCTS 정책 목록
        self.values = []      # MCTS 가치 목록
        self.dones = []       # 종료 여부 목록
        
    def add_step(self, state, action, reward, policy=None, value=None, done=False):
        """한 스텝의 데이터 추가"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.policies.append(policy)
        self.values.append(value)
        self.dones.append(done)
        
    def __len__(self):
        return len(self.states)
    
    def is_empty(self):
        return len(self.states) == 0


class ReplayBuffer:
    """에피소드를 저장하고 학습용 배치를 생성"""
    
    def __init__(self, capacity: int):
        """
        Args:
            capacity: 최대 에피소드 저장 개수
        """
        self.capacity = capacity
        self.buffer = []
        self.episode_count = 0
        
    def save_episode(self, episode: Episode):
        """에피소드 저장"""
        if episode.is_empty():
            return
            
        if len(self.buffer) >= self.capacity:
            # FIFO: 가장 오래된 에피소드 제거
            self.buffer.pop(0)
            
        self.buffer.append(episode)
        self.episode_count += 1
        
    def save_game(self, game_history):
        """GameHistory를 Episode로 변환하여 저장
        
        Args:
            game_history: SelfPlayManager의 GameHistory 객체
        """
        episode = Episode()
        
        # GameHistory 데이터를 Episode 형식으로 변환
        for i in range(len(game_history.observations)):
            done = (i == len(game_history.observations) - 1)  # 마지막 스텝인지 확인
            
            episode.add_step(
                state=game_history.observations[i],
                action=game_history.actions[i],
                reward=game_history.rewards[i],
                policy=game_history.policies[i],
                value=game_history.values[i],
                done=done
            )
        
        self.save_episode(episode)
        
    def get_batch(self, batch_size: int) -> Optional[List[Episode]]:
        """학습용 배치 샘플링"""
        if len(self.buffer) < batch_size:
            return None
        
        # 랜덤하게 에피소드 선택
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def size(self) -> int:
        """현재 버퍼 크기"""
        return len(self.buffer)
    
    def total_episodes(self) -> int:
        """지금까지 저장된 총 에피소드 수"""
        return self.episode_count
    
    def clear(self):
        """버퍼 초기화"""
        self.buffer = [] 