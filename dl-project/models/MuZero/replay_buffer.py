"""
MuZero Replay Buffer
경험을 저장하고 학습을 위한 샘플을 제공
"""
import numpy as np
from typing import List, Optional


class Episode:
    """하나의 에피소드(게임) 데이터를 저장"""
    
    def __init__(self, source_file=None, start_frame=None, end_frame=None):
        self.states = []      # 상태 목록
        self.actions = []     # 액션 목록 (self-play로 선택된 action)
        self.expert_ego_pos = []  # 전문가 ego 위치 목록 (GT 그리드 좌표 (row, col) 튜플)
        self.rewards = []     # 보상 목록
        self.target_rewards = []  # 목표 보상 목록 (n-step 누적 보상)
        self.pi = []    # MCTS 방문분포 목록 (π)
        self.policies = []    # 정책 신경망 출력 목록 (p)
        self.values = []      # MCTS 가치 목록
        self.target_values = []  # 목표 가치 목록 (n-step 누적 가치)
        self.dones = []       # 종료 여부 목록
        self.legal_mask = []  # 합법 액션 마스크 (9개 원소의 0/1 배열)

        # Episode 출처 정보
        self.source_file = source_file      # JSONL 파일명
        self.start_frame = start_frame      # 시작 프레임 번호
        self.end_frame = end_frame          # 종료 프레임 번호
        
    def add_step(self, state, action, reward, pi=None, value=None, done=False, expert_ego_pos=None, legal_mask=None):
        """한 스텝의 데이터 추가"""
        self.states.append(state)
        self.actions.append(action)
        self.expert_ego_pos.append(expert_ego_pos)
        self.rewards.append(reward)
        self.pi.append(pi)
        self.values.append(value)
        self.dones.append(done)
        self.legal_mask.append(legal_mask)
        
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
        
        # Episode 출처 정보 표시
        # if print_log and hasattr(episode, 'source_file') and episode.source_file:
            # print(f"Episode 추가: {episode.source_file} (프레임 {episode.start_frame}-{episode.end_frame}, 길이: {len(episode)})")
        
    # def save_game(self, game_history):
    #     """GameHistory를 Episode로 변환하여 저장
        
    #     Args:
    #         game_history: SelfPlayManager의 GameHistory 객체
    #     """
    #     episode = Episode()
        
    #     # GameHistory 데이터를 Episode 형식으로 변환
    #     for i in range(len(game_history.observations)):
    #         done = (i == len(game_history.observations) - 1)  # 마지막 스텝인지 확인
            
    #         episode.add_step(
    #             state=game_history.observations[i],
    #             action=game_history.actions[i],
    #             reward=game_history.rewards[i],
    #             pi=game_history.pi[i],
    #             value=game_history.values[i],
    #             done=done
    #         )
        
    #     self.save_episode(episode)
        
    def get_batch(self, batch_size: int) -> Optional[List[Episode]]:
        """학습용 배치 샘플링 (sample_episodes 사용)"""
        if len(self.buffer) < batch_size:
            return None
        
        return self.sample_episodes(batch_size)
    
    def size(self) -> int:
        """현재 버퍼 크기"""
        return len(self.buffer)
    
    def total_episodes(self) -> int:
        """지금까지 저장된 총 에피소드 수"""
        return self.episode_count
    
    def print_buffer_summary(self, print_file_info: bool = True):
        """Replay Buffer 내용 요약 출력"""
        if not self.buffer:
            print("Replay Buffer가 비어있습니다.")
            return
            
        print(f"\n=== Replay Buffer 상태 (총 {len(self.buffer)}개 Episode) ===")
        
        # 파일별 Episode 수 집계
        file_counts = {}
        frame_ranges = {}
        
        for i, episode in enumerate(self.buffer):
            if hasattr(episode, 'source_file') and episode.source_file:
                source_file = episode.source_file
                if source_file not in file_counts:
                    file_counts[source_file] = 0
                    frame_ranges[source_file] = []
                    
                file_counts[source_file] += 1
                if hasattr(episode, 'start_frame') and hasattr(episode, 'end_frame'):
                    frame_ranges[source_file].append((episode.start_frame, episode.end_frame))
        
        # 파일별 정보 출력
        for source_file, count in file_counts.items():
            print(f"  📁 {source_file}: {count}개 Episode")
            if source_file in frame_ranges and frame_ranges[source_file]:
                ranges = frame_ranges[source_file]
                min_frame = min(r[0] for r in ranges)
                max_frame = max(r[1] for r in ranges)
                # print(f"     프레임 범위: {min_frame} ~ {max_frame}")
        
        print(f"=== Buffer 활용도: {len(self.buffer)}/{self.capacity} ({len(self.buffer)/self.capacity*100:.1f}%) ===")
    
    def clear(self):
        """버퍼 초기화"""
        self.buffer = []
    
    def sample_episodes(self, num_episodes: int) -> List[Episode]:
        """랜덤하게 에피소드들을 샘플링
        
        Args:
            num_episodes: 샘플링할 에피소드 수
            
        Returns:
            샘플링된 Episode 리스트
        """
        if not self.buffer:
            return []
        
        num_samples = min(num_episodes, len(self.buffer))
        indices = np.random.choice(len(self.buffer), num_samples, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        """저장된 에피소드 수"""
        return len(self.buffer) 