"""MuZero 공통 유틸리티 함수"""
import numpy as np
from typing import List, Dict
from .self_play import GameHistory


def calculate_game_statistics(games: List[GameHistory]) -> Dict[str, float]:
    """게임 통계 계산
    
    Args:
        games: GameHistory 리스트
        
    Returns:
        통계 딕셔너리
    """
    if not games:
        return {
            'num_games': 0,
            'avg_length': 0,
            'std_length': 0,
            'avg_reward': 0,
            'std_reward': 0,
            'max_reward': 0,
            'min_reward': 0
        }
    
    game_lengths = [game.get_length() for game in games]
    total_rewards = [sum(game.rewards) for game in games]
    
    stats = {
        'num_games': len(games),
        'avg_length': np.mean(game_lengths),
        'std_length': np.std(game_lengths),
        'avg_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'max_reward': np.max(total_rewards),
        'min_reward': np.min(total_rewards)
    }
    
    return stats


def print_game_statistics(stats: Dict[str, float], title: str = "게임 통계"):
    """통계 출력
    
    Args:
        stats: calculate_game_statistics로 생성된 통계
        title: 출력 제목
    """
    print(f"\n{title} ({stats['num_games']}게임):")
    print(f"- 평균 길이: {stats['avg_length']:.1f} (±{stats['std_length']:.1f})")
    print(f"- 평균 보상: {stats['avg_reward']:.2f} (±{stats['std_reward']:.2f})")
    print(f"- 최대 보상: {stats['max_reward']:.2f}")
    print(f"- 최소 보상: {stats['min_reward']:.2f}")
    
    
def print_episode_summary(episode_idx: int, game: GameHistory, show_actions: bool = True):
    """에피소드 요약 출력
    
    Args:
        episode_idx: 에피소드 번호 (0-based)
        game: GameHistory 객체
        show_actions: 액션 분포 출력 여부
    """
    from .action import Action
    import json
    
    print(f"\n📊 Episode {episode_idx + 1} 결과:")
    print(f"   - 총 스텝: {game.get_length()}")
    print(f"   - 총 보상: {sum(game.rewards):.2f}")
    
    if show_actions:
        # 액션 통계
        action_counts = {}
        for action_idx in game.actions:
            action = Action.from_index(action_idx)
            key = str(action)
            action_counts[key] = action_counts.get(key, 0) + 1
        print(f"   - 액션 분포: {json.dumps(action_counts, indent=6)}")