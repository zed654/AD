"""
MuZero 평가 결과 시각화
GT vs Predicted 경로를 비교하여 시각화
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
from typing import List, Dict, Optional, Tuple
import argparse
from pathlib import Path
import pickle

# MuZero 관련 import
from models.MuZero.replay_buffer import Episode
from models.MuZero.action import Action

# rosetta import
import sys
sys.path.append(str(Path(__file__).parent / 'data' / 'db3_parser'))
from rosetta import (
    get_grid_value_info, get_object_color, get_ldm_segmentation_color,
    GRID_VALUE_EMPTY
)


class EvaluationVisualizer:
    """평가 결과 시각화 클래스"""
    
    def __init__(self, episodes: List[Episode], grid_size: int = 99):
        """
        Args:
            episodes: 평가된 Episode 리스트 (GT + MCTS 결과)
            grid_size: 그리드 크기
        """
        self.episodes = episodes
        self.grid_size = grid_size
        self.center = grid_size // 2  # 49
        
        # 색상 설정
        self.gt_color = 'blue'
        self.predicted_color = 'red'
        self.ego_color = 'yellow'
        
    def visualize_episode_comparison(self, episode_idx: int, save_path: Optional[str] = None):
        """단일 Episode의 GT vs Predicted 경로 비교 시각화
        
        Args:
            episode_idx: 시각화할 Episode 인덱스
            save_path: 저장 경로 (None이면 인터랙티브 모드)
        """
        if episode_idx >= len(self.episodes):
            print(f"Error: Episode {episode_idx} not found")
            return
            
        episode = self.episodes[episode_idx]
        
        # Figure 설정
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 왼쪽: 전체 경로 비교
        self._plot_full_trajectory(ax1, episode)
        
        # 오른쪽: 애니메이션 준비
        self.current_step = 0
        self.ax_anim = ax2
        self.episode = episode
        
        # 애니메이션 초기화
        self._init_animation()
        
        if save_path:
            # 정적 이미지로 저장
            self._plot_step(ax2, episode, len(episode) - 1)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        else:
            # 인터랙티브 애니메이션
            self.anim = FuncAnimation(
                fig, self._animate_step, frames=len(episode),
                interval=200, repeat=True, blit=False
            )
            
            # 키보드 이벤트 연결
            fig.canvas.mpl_connect('key_press_event', self._on_key_press)
            
            plt.tight_layout()
            plt.show()
    
    def _plot_full_trajectory(self, ax, episode):
        """전체 경로를 한 번에 표시"""
        ax.set_title("Full Trajectory Comparison", fontsize=14)
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        
        # GT 경로 (파란색)
        gt_xs = []
        gt_ys = []
        for step_idx in range(len(episode)):
            if hasattr(episode, 'expert_ego_pos') and step_idx < len(episode.expert_ego_pos):
                expert_pos = episode.expert_ego_pos[step_idx]
                if expert_pos:
                    # 그리드 좌표를 미터로 변환
                    x_meters = (expert_pos[0] - self.center) * 0.2
                    y_meters = (self.center - expert_pos[1]) * 0.2
                    gt_xs.append(x_meters)
                    gt_ys.append(y_meters)
        
        # Predicted 경로 (빨간색)
        pred_xs = []
        pred_ys = []
        for step_idx in range(len(episode)):
            if 'simulated_ego_pos' in episode.states[step_idx]:
                sim_pos = episode.states[step_idx]['simulated_ego_pos']
                if sim_pos:
                    # 그리드 좌표를 미터로 변환
                    x_meters = (sim_pos[0] - self.center) * 0.2
                    y_meters = (self.center - sim_pos[1]) * 0.2
                    pred_xs.append(x_meters)
                    pred_ys.append(y_meters)
        
        # 경로 그리기
        if gt_xs:
            ax.plot(gt_xs, gt_ys, 'b-', linewidth=2, label='GT (Expert)', alpha=0.7)
            ax.plot(gt_xs[0], gt_ys[0], 'go', markersize=10, label='Start')
            ax.plot(gt_xs[-1], gt_ys[-1], 'ro', markersize=10, label='GT End')
        
        if pred_xs:
            ax.plot(pred_xs, pred_ys, 'r--', linewidth=2, label='Predicted (MCTS)', alpha=0.7)
            ax.plot(pred_xs[-1], pred_ys[-1], 'r^', markersize=10, label='Predicted End')
        
        # 거리 오차 표시
        if gt_xs and pred_xs and len(gt_xs) == len(pred_xs):
            final_error = np.sqrt((gt_xs[-1] - pred_xs[-1])**2 + (gt_ys[-1] - pred_ys[-1])**2)
            ax.text(0.02, 0.98, f"Final Error: {final_error:.2f}m", 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.legend()
    
    def _init_animation(self):
        """애니메이션 초기화"""
        self.ax_anim.clear()
        self.ax_anim.set_title("Step-by-Step Comparison", fontsize=14)
        self.ax_anim.set_xlim(0, self.grid_size)
        self.ax_anim.set_ylim(0, self.grid_size)
        self.ax_anim.set_aspect('equal')
        self.ax_anim.invert_yaxis()
        self.ax_anim.grid(True, alpha=0.3)
        
        # 중심 표시
        self.ax_anim.axhline(y=self.center, color='gray', linestyle='--', alpha=0.5)
        self.ax_anim.axvline(x=self.center, color='gray', linestyle='--', alpha=0.5)
    
    def _plot_step(self, ax, episode, step_idx):
        """특정 스텝의 상태 표시"""
        ax.clear()
        self._init_animation()
        
        # 현재 스텝 정보
        ax.set_title(f"Step {step_idx + 1}/{len(episode)}", fontsize=14)
        
        # 환경 그리드 표시 (배경)
        if step_idx < len(episode.states):
            state = episode.states[step_idx]
            
            # obstacles (Channel 0)
            if 'obstacles' in state:
                obstacles = state['obstacles']
                for row in range(self.grid_size):
                    for col in range(self.grid_size):
                        value = obstacles[row, col]
                        if value != GRID_VALUE_EMPTY:
                            info = get_grid_value_info(value)
                            if info:
                                color = get_object_color(info['name'])
                                ax.add_patch(Rectangle((col-0.5, row-0.5), 1, 1, 
                                           facecolor=color, alpha=0.5))
        
        # GT 경로 (지금까지)
        gt_xs = []
        gt_ys = []
        for i in range(min(step_idx + 1, len(episode))):
            if hasattr(episode, 'expert_ego_pos') and i < len(episode.expert_ego_pos):
                expert_pos = episode.expert_ego_pos[i]
                if expert_pos:
                    gt_xs.append(expert_pos[1])  # col
                    gt_ys.append(expert_pos[0])  # row
        
        if gt_xs:
            ax.plot(gt_xs, gt_ys, 'b-', linewidth=2, alpha=0.7, label='GT Path')
            ax.plot(gt_xs[-1], gt_ys[-1], 'bo', markersize=10, label='GT Position')
        
        # Predicted 경로 (지금까지)
        pred_xs = []
        pred_ys = []
        for i in range(min(step_idx + 1, len(episode))):
            if 'simulated_ego_pos' in episode.states[i]:
                sim_pos = episode.states[i]['simulated_ego_pos']
                if sim_pos:
                    pred_xs.append(sim_pos[1])  # col
                    pred_ys.append(sim_pos[0])  # row
        
        if pred_xs:
            ax.plot(pred_xs, pred_ys, 'r--', linewidth=2, alpha=0.7, label='Predicted Path')
            ax.plot(pred_xs[-1], pred_ys[-1], 'r^', markersize=10, label='Predicted Position')
        
        # 현재 액션 표시
        if step_idx < len(episode.actions):
            action_idx = episode.actions[step_idx]
            action = Action.from_index(action_idx)
            ax.text(0.02, 0.02, f"Action: {action}", 
                   transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Value 표시
        if step_idx < len(episode.values):
            value = episode.values[step_idx]
            ax.text(0.98, 0.02, f"Value: {value:.3f}", 
                   transform=ax.transAxes, horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.legend(loc='upper right')
    
    def _animate_step(self, frame):
        """애니메이션 프레임 업데이트"""
        self._plot_step(self.ax_anim, self.episode, frame)
        return []
    
    def _on_key_press(self, event):
        """키보드 이벤트 처리"""
        if event.key == 'q':
            plt.close('all')
        elif event.key == ' ':  # 스페이스바로 일시정지/재생
            if hasattr(self, 'anim'):
                if self.anim.event_source is None:
                    return
                if self.anim.event_source.interval == 0:
                    self.anim.event_source.interval = 200
                else:
                    self.anim.event_source.interval = 0


def save_evaluation_results(episodes: List[Episode], save_path: str):
    """평가 결과를 파일로 저장"""
    with open(save_path, 'wb') as f:
        pickle.dump(episodes, f)
    print(f"Evaluation results saved to {save_path}")


def load_evaluation_results(load_path: str) -> List[Episode]:
    """저장된 평가 결과 로드"""
    with open(load_path, 'rb') as f:
        episodes = pickle.load(f)
    print(f"Loaded {len(episodes)} episodes from {load_path}")
    return episodes


def main():
    parser = argparse.ArgumentParser(description="MuZero 평가 결과 시각화")
    parser.add_argument("--results", type=str, required=True,
                       help="평가 결과 파일 경로 (.pkl)")
    parser.add_argument("--episode", type=int, default=0,
                       help="시각화할 에피소드 인덱스")
    parser.add_argument("--save", type=str, default=None,
                       help="이미지 저장 경로")
    parser.add_argument("--all", action="store_true",
                       help="모든 에피소드 시각화")
    
    args = parser.parse_args()
    
    # 평가 결과 로드
    episodes = load_evaluation_results(args.results)
    
    # 시각화
    visualizer = EvaluationVisualizer(episodes)
    
    if args.all:
        # 모든 에피소드 시각화
        for i in range(len(episodes)):
            save_path = None
            if args.save:
                save_path = f"{args.save}_episode_{i}.png"
            visualizer.visualize_episode_comparison(i, save_path)
            if not args.save:
                input("Press Enter for next episode...")
    else:
        # 단일 에피소드 시각화
        visualizer.visualize_episode_comparison(args.episode, args.save)


if __name__ == "__main__":
    main()