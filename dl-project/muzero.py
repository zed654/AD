import torch
import numpy as np
from pathlib import Path
import datetime
import argparse

# MuZero 관련 import
from models.MuZero.models import MuZeroFullyConnectedNetwork
from models.MuZero.config import MuZeroConfig
from models.MuZero.mcts import MCTS
from models.MuZero.action import Action
from models.MuZero.environment import DrivingEnvironment
from models.MuZero.self_play import SelfPlayManager
from models.MuZero.replay_buffer import ReplayBuffer
from models.MuZero.train import train_network

def get_network(config):
    """설정에 따라 적절한 MuZero 네트워크를 생성합니다."""
    if config.network == "fullyconnected":
        return MuZeroFullyConnectedNetwork(
            config.observation_shape,
            config.stacked_observations,
            len(config.action_space),
            config.encoding_size,
            config.fc_reward_layers,
            config.fc_value_layers,
            config.fc_policy_layers,
            config.fc_representation_layers,
            config.fc_dynamics_layers,
            config.support_size,
        )
    else:
        raise NotImplementedError(
            '지원되는 네트워크 설정은 "fullyconnected"입니다.'
        )

def main(debug_mode=False):
    """MuZero Self-play 예제"""
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 설정 생성
    config = MuZeroConfig()
    
    # 네트워크 생성
    network = get_network(config).to(device)
    
    # 환경 생성 (config 사용)
    environment = DrivingEnvironment(config)
    
    # Self-play 매니저 생성
    self_play_manager = SelfPlayManager(
        network=network,
        environment=environment,
        config=config,
        device=device
    )
    
    if debug_mode:
        print("\n=== Single Game Self-play (with rendering) ===")
        game_history = self_play_manager.play_game(temperature=1.0, render=True)
        print(f"\n게임 요약:")
        print(f"- 총 스텝: {game_history.get_length()}")
        print(f"- 총 보상: {sum(game_history.rewards):.2f}")
        print(f"- 평균 가치: {np.mean(game_history.values):.2f}")
    
    # 여러 게임 실행 (메인 데이터 수집)
    print("\n=== Multiple Games Self-play ===")
    games = self_play_manager.play_games(num_games=config.num_selfplay_games, temperature=1.0)
    
    # 통계 출력
    game_lengths = [game.get_length() for game in games]
    total_rewards = [sum(game.rewards) for game in games]
    
    print(f"\n전체 게임 통계 ({config.num_selfplay_games}게임):")
    print(f"- 평균 길이: {np.mean(game_lengths):.1f} (±{np.std(game_lengths):.1f})")
    print(f"- 평균 보상: {np.mean(total_rewards):.2f} (±{np.std(total_rewards):.2f})")
    print(f"- 최대 보상: {np.max(total_rewards):.2f}")
    print(f"- 최소 보상: {np.min(total_rewards):.2f}")
    
    # Replay Buffer에 게임 데이터 저장
    print("\n=== Replay Buffer 데이터 수집 ===")
    replay_buffer = ReplayBuffer(capacity=100)
    
    # 디버그 모드에서 단일 게임도 저장
    if debug_mode:
        replay_buffer.save_game(game_history)
        print(f"단일 게임 저장 완료")
    
    # 여러 게임 저장
    for game in games:
        replay_buffer.save_game(game)
    print(f"{len(games)}개 게임 저장 완료")
    
    print(f"\nReplay Buffer 상태:")
    print(f"  - 저장된 에피소드: {replay_buffer.size()}")
    print(f"  - 총 누적 에피소드: {replay_buffer.total_episodes()}")
    
    # Replay Buffer에서 배치 샘플링 테스트
    print("\n=== 배치 샘플링 테스트 ===")
    batch = replay_buffer.get_batch(batch_size=2)
    if batch:
        print(f"샘플링된 배치 크기: {len(batch)}")
        for i, episode in enumerate(batch):
            print(f"  - Episode {i+1}: {len(episode)} 스텝, 총 보상: {sum(episode.rewards):.2f}")
            
            # 첫 번째 스텝의 상세 정보 출력
            if len(episode) > 0:
                print(f"    첫 번째 스텝:")
                print(f"      - Action: {Action.from_index(episode.actions[0])}")
                print(f"      - Reward: {episode.rewards[0]:.2f}")
                print(f"      - Value: {episode.values[0]:.2f}")
                print(f"      - Policy shape: {episode.policies[0].shape}")
    else:
        print("배치 샘플링 실패 (데이터 부족)")

    # 모델 학습 수행
    print("\n=== MuZero 모델 학습 ===")
    train_network(network, replay_buffer, config, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MuZero Self-play")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with visualization")
    args = parser.parse_args()
    
    main(debug_mode=args.debug)
