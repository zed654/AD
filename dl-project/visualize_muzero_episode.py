#!/usr/bin/env python3
"""
MuZero Episode 시각화 스크립트 - 프로젝트 루트에서 실행
"""

import sys
import os
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''

import matplotlib

# GUI 가용성 확인
GUI_AVAILABLE = False
if 'DISPLAY' in os.environ and os.environ['DISPLAY']:
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        root.destroy()
        matplotlib.use('TkAgg')
        GUI_AVAILABLE = True
    except:
        pass

if not GUI_AVAILABLE:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
import argparse
from pathlib import Path
import torch
from typing import List, Dict, Tuple, Optional
import time

# MuZero 모듈 import
from models.MuZero.expert_data_loader import ExpertDataLoader
from models.MuZero.config import MuZeroConfig
from models.MuZero.replay_buffer import Episode
from models.MuZero.action import Action
from models.MuZero.data_parser import (
    get_grid_value_info, 
    GRID_VALUE_EMPTY, GRID_VALUE_PEDESTRIAN, GRID_VALUE_OTHER_OBJECT,
    GRID_VALUE_VEHICLE, GRID_VALUE_EGO
)


def get_grid_color(value: float) -> str:
    """Grid 값에 따른 색상을 반환합니다."""
    if value >= 0.9:  # 자차
        return 'green'
    elif value >= 0.5:  # 차량
        return 'blue'
    elif value >= 0.2:  # 보행자
        return 'red'
    elif value > 0:  # 기타
        return 'gray'
    else:
        return 'white'


class EpisodeInteractiveVisualizer:
    """Episode 데이터의 성능 최적화된 인터렉티브 시각화 클래스"""
    
    def __init__(self, episodes: List[Episode], config: MuZeroConfig, 
                 interval: int = 200, debug: bool = False):
        """
        Args:
            episodes: Episode 리스트
            config: MuZero 설정
            interval: 애니메이션 프레임 간격 (밀리초)
            debug: 디버그 모드
        """
        # ===== 1. 기본 설정 =====
        self.episodes = episodes
        self.config = config
        self.interval = interval
        self.debug = debug
        
        # Grid 설정
        self.grid_rows = config.observation_shape[1]
        self.grid_cols = config.observation_shape[2]
        self.cell_size = config.cell_size
        
        # ===== 2. Episode 및 스텝 관리 =====
        self.current_episode = 0
        self.current_step = 0
        self.total_episodes = len(episodes)
        
        # ===== 3. matplotlib 그래픽 요소 =====
        self.fig = None
        self.ax_grid = None  # 왼쪽: Grid observation
        self.ax_info = None  # 오른쪽: 정보 표시
        self.anim = None
        self.playing = False
        
        # ===== 4. 그래픽 요소 저장 =====
        self.artists = {
            'grid_image': None,      # Grid 이미지
            'info_text': None,       # Episode 정보 텍스트
            'step_text': None,       # 스텝별 정보 텍스트
            'policy_bars': None,     # Policy 막대 그래프
            'colorbar': None,        # Grid 값 색상 바
        }
        
        # ===== 5. 캐시 =====
        self.observation_cache = {}  # numpy 변환 캐시
        
        # ===== 6. 전처리 실행 =====
        self.preprocess_episodes()
    
    def preprocess_episodes(self):
        """모든 Episode의 observation을 numpy로 변환하여 캐시"""
        print("Preprocessing episodes for better performance...")
        start_time = time.time()
        
        for ep_idx, episode in enumerate(self.episodes):
            for step_idx, state in enumerate(episode.states):
                cache_key = (ep_idx, step_idx)
                
                # state가 dict인 경우 obstacles 추출 (이전 이름 grid도 지원)
                if isinstance(state, dict):
                    obs = state.get('obstacles', state.get('grid'))  # obstacles 우선, 없으면 grid
                else:
                    # 기존 tensor 형태 (하위 호환성)
                    obs = state
                
                # Torch tensor를 numpy로 변환
                if hasattr(obs, 'numpy'):
                    obs_np = obs.numpy()
                else:
                    obs_np = obs
                
                # 3D인 경우 첫 번째 채널만
                if len(obs_np.shape) == 3:
                    obs_np = obs_np[0]
                
                self.observation_cache[cache_key] = obs_np
        
        elapsed = time.time() - start_time
        print(f"Preprocessing completed in {elapsed:.2f} seconds")
    
    def on_key(self, event):
        """키보드 이벤트 핸들러"""
        if event.key == ' ':  # 스페이스바: 재생/일시정지
            self.playing = not self.playing
            if self.playing and self.anim:
                self.anim.resume()
                print("▶ Playing")
            elif self.anim:
                self.anim.pause()
                print("⏸ Paused")
                
        elif event.key == 'right':
            # 다음 스텝
            current_episode = self.episodes[self.current_episode]
            if self.current_step < len(current_episode) - 1:
                self.current_step += 1
            else:
                # 다음 Episode로
                if self.current_episode < self.total_episodes - 1:
                    self.current_episode += 1
                    self.current_step = 0
            self.update_frame()
            if not self.playing:
                plt.draw()
                
        elif event.key == 'left':
            # 이전 스텝
            if self.current_step > 0:
                self.current_step -= 1
            else:
                # 이전 Episode로
                if self.current_episode > 0:
                    self.current_episode -= 1
                    self.current_step = len(self.episodes[self.current_episode]) - 1
            self.update_frame()
            if not self.playing:
                plt.draw()
                
        elif event.key == 'up':
            # 다음 Episode
            if self.current_episode < self.total_episodes - 1:
                self.current_episode += 1
                self.current_step = 0
                self.update_frame()
                if not self.playing:
                    plt.draw()
                    
        elif event.key == 'down':
            # 이전 Episode
            if self.current_episode > 0:
                self.current_episode -= 1
                self.current_step = 0
                self.update_frame()
                if not self.playing:
                    plt.draw()
                    
        elif event.key == 'escape' or event.key == 'q':
            if self.anim:
                self.anim.pause()
            plt.close(self.fig)
    
    def init_graphics(self):
        """그래픽 요소를 초기화합니다."""
        # Grid view 설정
        self.ax_grid.set_aspect('equal')
        self.ax_grid.set_xlabel('X (cells)')
        self.ax_grid.set_ylabel('Y (cells)')
        
        # 초기 Grid 이미지
        empty_grid = np.zeros((self.grid_rows, self.grid_cols))
        self.artists['grid_image'] = self.ax_grid.imshow(
            empty_grid, cmap='hot', origin='lower', vmin=-1, vmax=1
        )
        
        # Colorbar
        self.artists['colorbar'] = plt.colorbar(
            self.artists['grid_image'], ax=self.ax_grid
        )
        self.artists['colorbar'].set_label('Grid Value')
        
        # 색상 범례 추가 (hot colormap 기준)
        from matplotlib.cm import hot
        legend_elements = []
        for value in [GRID_VALUE_EMPTY, GRID_VALUE_PEDESTRIAN, GRID_VALUE_OTHER_OBJECT, 
                      GRID_VALUE_VEHICLE, GRID_VALUE_EGO]:
            name, _ = get_grid_value_info(value)
            color = hot(value)
            legend_elements.append(
                patches.Patch(color=color, label=f'{name} ({value})')
            )
        self.ax_grid.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        # 자차 마커는 Episode 데이터에 이미 포함되어 있으므로 별도로 그리지 않음
        # Episode의 grid 데이터에서 1.0 값이 ego 위치를 나타냄
        
        # Grid 선
        for i in range(self.grid_rows + 1):
            self.ax_grid.axhline(y=i-0.5, color='gray', linewidth=0.5, alpha=0.3)
        for i in range(self.grid_cols + 1):
            self.ax_grid.axvline(x=i-0.5, color='gray', linewidth=0.5, alpha=0.3)
        
        # 정보 표시 영역 초기화
        self.ax_info.axis('off')
        
        # 텍스트 초기화
        self.artists['info_text'] = self.ax_info.text(
            0.05, 0.95, '', transform=self.ax_info.transAxes,
            verticalalignment='top', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8)
        )
        
        self.artists['step_text'] = self.ax_info.text(
            0.05, 0.5, '', transform=self.ax_info.transAxes,
            verticalalignment='top', fontsize=10, family='monospace'
        )
    
    def update_frame(self):
        """현재 프레임 업데이트"""
        # 현재 Episode와 스텝
        episode = self.episodes[self.current_episode]
        
        # ===== 1. Grid 업데이트 =====
        cache_key = (self.current_episode, self.current_step)
        grid = self.observation_cache.get(cache_key)
        
        if grid is not None:
            self.artists['grid_image'].set_data(grid)
            
            # Grid 제목 업데이트
            self.ax_grid.set_title(
                f'Episode {self.current_episode + 1}/{self.total_episodes}, '
                f'Step {self.current_step + 1}/{len(episode)}'
            )
        
        # ===== 2. Episode 정보 업데이트 =====
        info_text = f"Episode {self.current_episode + 1} / {self.total_episodes}\n"
        info_text += f"Total Steps: {len(episode)}\n"
        info_text += f"Current Step: {self.current_step + 1}\n"
        
        # Episode 총 보상
        total_reward = sum(episode.rewards)
        info_text += f"\nTotal Reward: {total_reward:.2f}"
        
        self.artists['info_text'].set_text(info_text)
        
        # ===== 3. 스텝 정보 업데이트 =====
        step_text = "Step Information:\n" + "="*30 + "\n"
        
        # Action (Self-play)
        if self.current_step < len(episode.actions):
            action_idx = episode.actions[self.current_step]
            action = Action.from_index(action_idx)
            step_text += f"Action: {action_idx} ({action.position.value})\n"
        
        # Expert Ego Position (GT 그리드 좌표)
        if hasattr(episode, 'expert_ego_pos') and self.current_step < len(episode.expert_ego_pos):
            expert_pos = episode.expert_ego_pos[self.current_step]
            if expert_pos is not None:
                if isinstance(expert_pos, tuple) and len(expert_pos) == 2:
                    # 그리드 좌표 (row, col)
                    step_text += f"Expert Ego Pos: Grid({expert_pos[0]}, {expert_pos[1]})\n"
                else:
                    # 기타 형식
                    step_text += f"Expert Ego Pos: {expert_pos}\n"
        # 이전 버전 호환성 (expert_actions)
        elif hasattr(episode, 'expert_actions') and self.current_step < len(episode.expert_actions):
            expert_action = episode.expert_actions[self.current_step]
            if expert_action is not None:
                step_text += f"Expert Action: {expert_action}\n"
        
        # Reward
        if self.current_step < len(episode.rewards):
            reward = episode.rewards[self.current_step]
            step_text += f"Reward: {reward:.3f}\n"
        
        # Value
        if self.current_step < len(episode.values):
            value = episode.values[self.current_step]
            step_text += f"Value: {value:.3f}\n"
        
        # Done
        if self.current_step < len(episode.dones):
            done = episode.dones[self.current_step]
            step_text += f"Done: {done}\n"
        
        # Policy (Top 5)
        if self.current_step < len(episode.policies) and episode.policies[self.current_step] is not None:
            policy = episode.policies[self.current_step]
            step_text += f"\nPolicy (Top 5):\n"
            top_actions = np.argsort(policy)[-5:][::-1]
            for i, action_idx in enumerate(top_actions):
                action = Action.from_index(action_idx)
                prob = policy[action_idx]
                step_text += f"  {i+1}. Action {action_idx} ({action.position.value}): {prob:.3f}\n"
        else:
            step_text += f"\nPolicy: None (GT data)\n"
        
        # Grid 통계 및 객체 타입
        if grid is not None:
            unique_values, counts = np.unique(grid, return_counts=True)
            step_text += f"\nGrid Statistics:\n"
            for val, count in zip(unique_values, counts):
                if val != 0:  # 0이 아닌 값만 표시
                    name, _ = get_grid_value_info(val)
                    step_text += f"  {name} ({val}): {count} cells\n"
        
        # 추가 정보 표시 (state가 dict인 경우)
        if self.current_step < len(episode.states) and isinstance(episode.states[self.current_step], dict):
            state = episode.states[self.current_step]
            
            # Ego 위치
            if 'ego_pos' in state:
                ego_x, ego_y = state['ego_pos']
                step_text += f"\nEgo Position: ({ego_x:.2f}, {ego_y:.2f})\n"
            
            # 객체 수
            if 'objects' in state and state['objects']:
                step_text += f"Objects: {len(state['objects'])} detected\n"
            
            # 방향 정보
            if 'near_direction' in state and state['near_direction']:
                step_text += f"Near Direction: {state['near_direction']}\n"
            if 'destination_direction' in state and state['destination_direction']:
                step_text += f"Dest Direction: {state['destination_direction']}\n"
            
            # Static Environment 정보
            if 'static_environment' in state and state['static_environment']:
                static_env = state['static_environment']
                step_text += f"\nStatic Environment:\n"
                for env_type, grid in static_env.items():
                    if isinstance(grid, np.ndarray):
                        count = np.sum(grid > 0)
                        if count > 0:
                            step_text += f"  {env_type}: {count} cells\n"
        
        self.artists['step_text'].set_text(step_text)
        
        # ===== 4. Policy 막대 그래프 업데이트 =====
        if self.artists['policy_bars'] is not None:
            self.artists['policy_bars'].remove()
        
        if self.current_step < len(episode.policies) and episode.policies[self.current_step] is not None:
            policy = episode.policies[self.current_step]
            # 상위 9개 액션만 표시
            top_actions = np.argsort(policy)[-9:]
            top_probs = policy[top_actions]
            
            # 막대 그래프
            y_pos = np.linspace(0.3, 0.05, len(top_actions))
            bars = self.ax_info.barh(
                y_pos, top_probs, height=0.02,
                transform=self.ax_info.transAxes
            )
            
            # 막대 색상
            for i, (bar, action_idx) in enumerate(zip(bars, top_actions)):
                if action_idx == episode.actions[self.current_step]:
                    bar.set_color('green')  # 실제 선택된 액션
                else:
                    bar.set_color('lightblue')
            
            self.artists['policy_bars'] = bars
        
        # ===== 5. 제목 업데이트 =====
        status = "[Playing]" if self.playing else "[Paused]"
        self.fig.suptitle(
            f'MuZero Episode Visualization {status}\n'
            f'(Space: Play/Pause, ←/→: Step, ↑/↓: Episode, ESC/Q: Quit)',
            fontsize=14
        )
    
    def animate_update(self, _):
        """애니메이션 업데이트 함수"""
        if self.playing:
            # 다음 스텝으로
            current_episode = self.episodes[self.current_episode]
            if self.current_step < len(current_episode) - 1:
                self.current_step += 1
            else:
                # 다음 Episode로 (순환)
                self.current_episode = (self.current_episode + 1) % self.total_episodes
                self.current_step = 0
            
            self.update_frame()
        return []
    
    def show(self, start_episode: int = 0, start_step: int = 0):
        """시각화 시작"""
        self.current_episode = start_episode
        self.current_step = start_step
        
        # Figure 생성
        self.fig, (self.ax_grid, self.ax_info) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 키보드 이벤트 연결
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Layout 조정
        plt.tight_layout()
        
        # 그래픽 초기화
        self.init_graphics()
        
        # 첫 프레임 표시
        self.update_frame()
        
        # 애니메이션 생성
        self.anim = FuncAnimation(
            self.fig, self.animate_update,
            interval=self.interval,
            repeat=True,
            blit=False,
            cache_frame_data=False  # 경고 방지
        )
        
        # 일시정지 상태로 시작
        self.anim.pause()
        
        try:
            plt.show()
        except Exception as e:
            print(f"GUI display error: {e}")
            print("Use --save option to save as files instead.")


def save_episode_frames(episodes: List[Episode], config: MuZeroConfig, 
                       save_dir: Path, num_frames: int = 5):
    """Episode의 주요 프레임들을 파일로 저장"""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for ep_idx, episode in enumerate(episodes):
        print(f"\nSaving Episode {ep_idx + 1}...")
        
        # Episode 전체를 한 그림에 표시
        num_steps = min(num_frames, len(episode))
        fig, axes = plt.subplots(2, num_steps, figsize=(4*num_steps, 8))
        
        if num_steps == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle(f'Episode {ep_idx + 1} (Total: {len(episode)} steps, Reward: {sum(episode.rewards):.2f})', 
                     fontsize=16)
        
        # 균등하게 분포된 스텝 선택
        step_indices = np.linspace(0, len(episode)-1, num_steps, dtype=int)
        
        for i, step_idx in enumerate(step_indices):
            # 상단: Grid
            ax_grid = axes[0, i]
            state = episode.states[step_idx]
            
            # state가 dict인 경우 obstacles 추출 (이전 이름 grid도 지원)
            if isinstance(state, dict):
                obs = state.get('obstacles', state.get('grid'))  # obstacles 우선, 없으면 grid
            else:
                obs = state
                
            if hasattr(obs, 'numpy'):
                grid = obs.numpy()
            else:
                grid = obs
            if len(grid.shape) == 3:
                grid = grid[0]
            
            im = ax_grid.imshow(grid, cmap='hot', origin='lower', vmin=0, vmax=1)
            ax_grid.set_title(f'Step {step_idx}')
            ax_grid.set_xlabel('X')
            ax_grid.set_ylabel('Y')
            
            # 색상 범례 추가
            from matplotlib.cm import hot
            legend_elements = []
            for value in [GRID_VALUE_EMPTY, GRID_VALUE_PEDESTRIAN, GRID_VALUE_OTHER_OBJECT,
                          GRID_VALUE_VEHICLE, GRID_VALUE_EGO]:
                name, _ = get_grid_value_info(value)
                color = hot(value)
                legend_elements.append(
                    patches.Patch(color=color, label=f'{name} ({value})')
                )
            ax_grid.legend(handles=legend_elements, loc='upper right', fontsize=6)
            
            # Episode 데이터에 이미 ego 위치(GRID_VALUE_EGO)가 포함되어 있음
            # grid에서 값이 GRID_VALUE_EGO인 위치가 ego vehicle
            ego_positions = np.where(grid == GRID_VALUE_EGO)
            if len(ego_positions[0]) > 0:
                ego_row, ego_col = ego_positions[0][0], ego_positions[1][0]
                ax_grid.plot(ego_col, ego_row, 'g*', markersize=15, label='Ego')
                # 그리드 중심점 표시
                ax_grid.axhline(y=ego_row, color='green', alpha=0.3, linestyle='--')
                ax_grid.axvline(x=ego_col, color='green', alpha=0.3, linestyle='--')
            
            # 하단: 정보
            ax_info = axes[1, i]
            ax_info.axis('off')
            
            info_text = f"Step: {step_idx}\n"
            if step_idx < len(episode.actions):
                action_idx = episode.actions[step_idx]
                action = Action.from_index(action_idx)
                info_text += f"Action: {action_idx}\n({action.position.value})\n"
                
                # Expert Ego Position 표시
                if hasattr(episode, 'expert_ego_pos') and step_idx < len(episode.expert_ego_pos):
                    expert_pos = episode.expert_ego_pos[step_idx]
                    if expert_pos is not None:
                        if isinstance(expert_pos, tuple) and len(expert_pos) == 2:
                            info_text += f"Expert Pos: G({expert_pos[0]}, {expert_pos[1]})\n"
                        else:
                            info_text += f"Expert Pos: {expert_pos}\n"
                # 이전 버전 호환성
                elif hasattr(episode, 'expert_actions') and step_idx < len(episode.expert_actions):
                    expert_action = episode.expert_actions[step_idx]
                    if expert_action is not None:
                        info_text += f"Expert: {expert_action}\n"
                
                info_text += f"Reward: {episode.rewards[step_idx]:.3f}\n"
                
                # Policy top 3
                if step_idx < len(episode.policies) and episode.policies[step_idx] is not None:
                    policy = episode.policies[step_idx]
                    top_actions = np.argsort(policy)[-3:][::-1]
                    info_text += f"\nTop Actions:\n"
                    for j, a in enumerate(top_actions):
                        info_text += f"{j+1}. {a}: {policy[a]:.3f}\n"
            
            ax_info.text(0.1, 0.9, info_text, transform=ax_info.transAxes,
                        verticalalignment='top', fontsize=10, family='monospace')
        
        plt.tight_layout()
        save_path = save_dir / f"episode_{ep_idx + 1}.png"
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {save_path}")
    
    print(f"\n✅ All episodes saved to: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='MuZero Episode 시각화')
    parser.add_argument('--num-episodes', type=int, default=5,
                       help='로드할 Episode 수 (기본값: 5)')
    parser.add_argument('--episode-length', type=int, default=50,
                       help='Episode 길이 (기본값: 50)')
    parser.add_argument('--overlap', type=int, default=25,
                       help='Episode 간 겹침 (기본값: 25)')
    parser.add_argument('--interval', type=int, default=200,
                       help='애니메이션 프레임 간격 (ms, 기본값: 200)')
    parser.add_argument('--data-path', type=str, default=None,
                       help='GT 데이터 파일 경로')
    parser.add_argument('--debug', action='store_true',
                       help='디버그 정보 출력')
    parser.add_argument('--save', action='store_true',
                       help='파일로 저장 (GUI 대신)')
    parser.add_argument('--save-dir', type=str, default='visualizations/muzero_episodes',
                       help='저장 디렉토리')
    parser.add_argument('--num-frames', type=int, default=5,
                       help='Episode당 저장할 프레임 수')
    
    args = parser.parse_args()
    
    # MuZero 설정 로드
    config = MuZeroConfig()
    
    # 데이터 경로 오버라이드
    if args.data_path:
        config.gt_data_path = Path(args.data_path)
    
    # Episode 로드
    print("Loading expert data and creating episodes...")
    data_loader = ExpertDataLoader(config)
    episodes = data_loader.create_episodes_from_json(
        num_episodes=args.num_episodes,
        episode_length=args.episode_length,
        overlap=args.overlap
    )
    
    if not episodes:
        print("No episodes created!")
        return
    
    print(f"\nLoaded {len(episodes)} episodes")
    for i, ep in enumerate(episodes):
        print(f"  Episode {i+1}: {len(ep)} steps, "
              f"total reward: {sum(ep.rewards):.2f}")
    
    # 시각화
    if args.save or not GUI_AVAILABLE:
        # 파일로 저장
        if not GUI_AVAILABLE and not args.save:
            print("\n⚠️  GUI backend not available. Saving to files instead.")
        save_dir = Path(args.save_dir)
        save_episode_frames(episodes, config, save_dir, args.num_frames)
    else:
        # 인터랙티브 모드 (GUI 환경에서만)
        print("\n=== MuZero Episode Interactive Visualization ===")
        print("Controls: Space(Play/Pause), ←/→(Step), ↑/↓(Episode), ESC/Q(Quit)")
        
        visualizer = EpisodeInteractiveVisualizer(
            episodes, config, args.interval, args.debug
        )
        visualizer.show()


if __name__ == '__main__':
    main()