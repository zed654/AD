#!/usr/bin/env python3
"""
MuZero Episode 시각화 스크립트 - 프로젝트 루트에서 실행
"""

import sys
import os
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''

import matplotlib

# 안전한 백엔드 설정
GUI_AVAILABLE = False
if 'DISPLAY' in os.environ and os.environ['DISPLAY']:
    # GUI 환경에서는 안전한 백엔드 시도
    try:
        # Qt 백엔드 시도 (가장 안정적)
        matplotlib.use('Qt5Agg')
        GUI_AVAILABLE = True
        print("Using Qt5Agg backend")
    except:
        try:
            # TkAgg 백엔드 시도
            matplotlib.use('TkAgg')
            GUI_AVAILABLE = True
            print("Using TkAgg backend")
        except:
            # 모든 GUI 백엔드 실패시 Agg 사용
            matplotlib.use('Agg')
            print("GUI backends failed, using Agg (save-only mode)")
else:
    # DISPLAY 없으면 Agg 사용
    matplotlib.use('Agg')
    print("No DISPLAY detected, using Agg backend")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import CheckButtons
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
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'data' / 'db3_parser'))
from rosetta import (
    get_grid_value_info,
    GRID_VALUE_EMPTY, GRID_VALUE_PEDESTRIAN, GRID_VALUE_OTHER_OBJECT,
    GRID_VALUE_VEHICLE,
    GRID_VALUE_CLIFF, GRID_VALUE_STATIC_OBSTACLE, GRID_VALUE_MISSING_POINTS,
    get_object_name, get_ldm_segmentation_name,
    get_object_color, get_ldm_segmentation_color,
    OBJECT_CLASS_IDS, LDM_SEGMENTATION_IDS,
    OBJECT_COLORS, LDM_SEGMENTATION_COLORS
)


def get_grid_color(value: float, channel: int = None) -> str:
    """Grid 값과 채널에 따른 색상을 반환합니다.
    
    Args:
        value: Grid 값
        channel: 채널 번호 (0: obstacles, 1: static_environment, 2: ground_class_id)
    """
    # 부동소수점 비교를 위한 작은 오차 허용
    epsilon = 0.01
    
    # 빈 공간
    if abs(value - GRID_VALUE_EMPTY) < epsilon:  # 0.0
        return 'white'
    
    # Channel 0: 동적 객체
    if channel == 0:
        if abs(value - GRID_VALUE_VEHICLE) < epsilon:  # 0.6: 차량
            return 'red'
        elif abs(value - GRID_VALUE_OTHER_OBJECT) < epsilon:  # 0.4: 기타 객체
            return 'orange'
        elif abs(value - GRID_VALUE_PEDESTRIAN) < epsilon:  # 0.3: 보행자
            return 'magenta'
    
    # Channel 1: 정적 환경
    elif channel == 1:
        if abs(value - GRID_VALUE_MISSING_POINTS) < epsilon:  # 0.9: Missing points
            return 'purple'
        elif abs(value - GRID_VALUE_STATIC_OBSTACLE) < epsilon:  # 0.6: Static obstacles
            return 'yellow'
        elif abs(value - GRID_VALUE_CLIFF) < epsilon:  # 0.3: Cliff
            return 'cyan'
    
    # Channel 2: 지면 semantic class ID
    elif channel == 2:
        class_id = int(value)
        if class_id == 0:  # Empty/Background
            return 'lightgray'
        else:
            # LDM segmentation 색상 가져오기
            color_hex = get_ldm_segmentation_color(class_id)
            if color_hex:
                # hex 색상을 matplotlib 색상 이름으로 변환하거나 그대로 반환
                return color_hex
            else:
                return 'gray'
    
    # 채널 정보가 없으면 값으로만 추측 (하위 호환성)
    elif channel is None:
        # 고유한 값들
        if abs(value - GRID_VALUE_MISSING_POINTS) < epsilon:  # 0.9
            return 'purple'
        elif abs(value - GRID_VALUE_OTHER_OBJECT) < epsilon:  # 0.4
            return 'orange'
        # 중복되는 값들 - 기본적으로 동적 객체로 가정
        elif abs(value - 0.6) < epsilon:  # Vehicle or Static Obstacle
            return 'red'  # 차량으로 가정
        elif abs(value - 0.3) < epsilon:  # Pedestrian or Cliff
            return 'magenta'  # 보행자로 가정
    
    return 'gray'  # 정의되지 않은 값


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
        self.ax_grid = None  # 왼쪽: 통합 그리드 (Obstacles + Static + Class ID)
        self.ax_info = None  # 오른쪽: 정보 표시
        self.anim = None
        self.playing = False
        
        # ===== 4. 그래픽 요소 저장 =====
        self.artists = {
            'grid_image': None,      # 통합 그리드 이미지
            'info_text': None,       # Episode 정보 텍스트
            'step_text': None,       # 스텝별 정보 텍스트
            'policy_bars': None,     # Policy 막대 그래프
            'colorbar': None,        # Grid 값 색상 바
        }
        
        # ===== 5. 캐시 =====
        self.observation_cache = {}  # numpy 변환 캐시
        
        # ===== 6. 채널 토글 상태 =====
        self.channel_visibility = {
            'obstacles': True,          # Channel 0: 동적 객체
            'static_environment': True, # Channel 1: 정적 환경
            'ground_class_id': True,    # Channel 2: 지면 class_id
            'ego': True,                # Ego 위치 표시
            'directions': True          # 방향 벡터 표시
        }
        
        # ===== 7. 전처리 실행 =====
        self.preprocess_episodes()
    
    def preprocess_episodes(self):
        """모든 Episode의 observation을 numpy로 변환하여 캐시"""
        print("Preprocessing episodes for better performance...")
        start_time = time.time()
        
        for ep_idx, episode in enumerate(self.episodes):
            for step_idx, state in enumerate(episode.states):
                cache_key = (ep_idx, step_idx)
                
                # 첫 번째 Episode의 첫 번째 스텝에서 state 구조 확인
                if ep_idx == 0 and step_idx == 0:
                    print(f"[DEBUG] State type: {type(state)}")
                    if isinstance(state, dict):
                        print(f"[DEBUG] State keys: {state.keys()}")
                    elif hasattr(state, 'shape'):
                        print(f"[DEBUG] State shape: {state.shape}")
                
                # state가 dict인 경우 obstacles와 static_environment 추출
                if isinstance(state, dict):
                    obstacles = state.get('obstacles', state.get('grid'))  # obstacles 우선, 없으면 grid
                    static_env = state.get('static_environment')
                    ground_class_id = state.get('ground_class_id')  # Channel 2: Ground semantic class ID
                    
                    # class_id_np 초기화
                    class_id_np = None
                    
                    # numpy로 변환
                    if hasattr(obstacles, 'numpy'):
                        obstacles_np = obstacles.numpy()
                    else:
                        obstacles_np = obstacles
                    
                    # ground_class_id 처리
                    if ground_class_id is not None and not isinstance(ground_class_id, dict):
                        if hasattr(ground_class_id, 'numpy'):
                            class_id_np = ground_class_id.numpy()
                        else:
                            class_id_np = ground_class_id
                        
                        # Shape 처리
                        if len(class_id_np.shape) == 4:
                            class_id_np = class_id_np[0, 0]
                        elif len(class_id_np.shape) == 3:
                            class_id_np = class_id_np[0]
                        
                        # 첫 번째 Episode의 첫 번째 스텝에서 ground_class_id 값 확인
                        if ep_idx == 0 and step_idx == 0:
                            print(f"[DEBUG] class_id_np shape: {class_id_np.shape}")
                            unique_values = np.unique(class_id_np)
                            print(f"[DEBUG] ground_class_id unique values: {unique_values}")
                            print(f"[DEBUG] ground_class_id value counts:")
                            for val in unique_values:
                                count = np.sum(class_id_np == val)
                                print(f"  {val}: {count} pixels")
                            
                            # obstacles와 비교
                            print(f"\n[DEBUG] obstacles_np shape: {obstacles_np.shape}")
                            obstacles_unique = np.unique(obstacles_np)
                            print(f"[DEBUG] obstacles unique values: {obstacles_unique}")
                            
                            # 두 배열이 같은지 확인
                            if np.array_equal(class_id_np, obstacles_np):
                                print("[ERROR] class_id_np와 obstacles_np가 완전히 같습니다!")
                            else:
                                print("[OK] class_id_np와 obstacles_np가 다릅니다.")
                                # 몇 개의 픽셀이 다른지 확인
                                diff_count = np.sum(class_id_np != obstacles_np)
                                print(f"  다른 픽셀 수: {diff_count}/{class_id_np.size} ({diff_count/class_id_np.size*100:.1f}%)")
                    
                    # 세 채널 모두 추출 (obstacles, static_environment, ground_class_id)
                    # Shape 처리 - obstacles
                    if len(obstacles_np.shape) == 4:
                        obstacles_np = obstacles_np[0, 0]
                    elif len(obstacles_np.shape) == 3:
                        obstacles_np = obstacles_np[0]
                    
                    # static_environment 처리
                    static_env_np = None
                    if static_env is not None and not isinstance(static_env, dict):
                        if hasattr(static_env, 'numpy'):
                            static_env_np = static_env.numpy()
                        else:
                            static_env_np = static_env
                        
                        # Shape 처리
                        if len(static_env_np.shape) == 4:
                            static_env_np = static_env_np[0, 0]
                        elif len(static_env_np.shape) == 3:
                            static_env_np = static_env_np[0]
                else:
                    # 기존 tensor 형태 (하위 호환성)
                    obs = state
                    # Torch tensor를 numpy로 변환
                    if hasattr(obs, 'numpy'):
                        obs_full = obs.numpy()
                    else:
                        obs_full = obs
                    
                    # 3채널 observation 처리
                    # Shape: (batch, channel, height, width) 또는 (channel, height, width)
                    if len(obs_full.shape) == 4:
                        # batch dimension 제거
                        obs_full = obs_full[0]  # shape: (3, 99, 99)
                    
                    # class_id_np 초기화
                    class_id_np = None
                    
                    if len(obs_full.shape) == 3 and obs_full.shape[0] >= 3:
                        # 6채널 데이터: channel 0 = obstacles, channel 1 = static_env, channel 2 = ground_class_id
                        # channel 3,4 = directions, channel 5 = simulated_ego_pos
                        obstacles_np = obs_full[0]  # 첫 번째 채널 (동적 장애물)
                        static_env_np = obs_full[1]  # 두 번째 채널 (정적 환경)
                        class_id_np = obs_full[2]   # 세 번째 채널 (ground_class_id)
                        
                        # 첫 번째 Episode의 첫 번째 스텝에서만 디버그 출력
                        if ep_idx == 0 and step_idx == 0:
                            print(f"[DEBUG] 3채널 observation shape: {obs_full.shape}")
                            print(f"[DEBUG] class_id_np unique values: {np.unique(class_id_np)}")
                        
                        # 병합하지 않고 별도로 저장
                        obs_np = obstacles_np
                        static_np = static_env_np
                    else:
                        # 1채널만 있는 경우 (이전 버전 호환성)
                        if len(obs_full.shape) == 3:
                            obs_np = obs_full[0]
                        else:
                            obs_np = obs_full
                        class_id_np = None
                
                # 3개 채널 모두 별도로 저장 (obstacles, static_environment, ground_class_id)
                if isinstance(state, dict):
                    self.observation_cache[cache_key] = {
                        'obstacles': obstacles_np,
                        'static_environment': static_env_np,  # 병합하지 않고 원본 그대로 저장
                        'ground_class_id': class_id_np
                    }
                else:
                    # 이전 버전 호환성 - 3채널 텐서 형태
                    if 'static_np' in locals():
                        self.observation_cache[cache_key] = {
                            'obstacles': obs_np,
                            'static_environment': static_np,
                            'ground_class_id': class_id_np
                        }
                    else:
                        self.observation_cache[cache_key] = (obs_np, class_id_np)
        
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
                    
        elif event.key == '1':
            # Toggle Channel 0 (Dynamic Objects)
            self.channel_visibility['obstacles'] = not self.channel_visibility['obstacles']
            self.update_frame()
            if not self.playing:
                plt.draw()
        
        elif event.key == '2':
            # Toggle Channel 1 (Static Environment)
            self.channel_visibility['static_environment'] = not self.channel_visibility['static_environment']
            self.update_frame()
            if not self.playing:
                plt.draw()
        
        elif event.key == '3':
            # Toggle Channel 2 (Ground Class ID)
            self.channel_visibility['ground_class_id'] = not self.channel_visibility['ground_class_id']
            self.update_frame()
            if not self.playing:
                plt.draw()
        
        elif event.key == '4':
            # Toggle Ego position
            self.channel_visibility['ego'] = not self.channel_visibility['ego']
            self.update_frame()
            if not self.playing:
                plt.draw()
        
        elif event.key == '5':
            # Toggle Direction vectors
            self.channel_visibility['directions'] = not self.channel_visibility['directions']
            self.update_frame()
            if not self.playing:
                plt.draw()
                
        elif event.key == 'escape' or event.key == 'q':
            if self.anim:
                self.anim.pause()
            plt.close(self.fig)
    
    def setup_toggle_buttons(self):
        """채널별 토글 버튼 설정"""
        self.ax_toggle.set_title('Channel Visibility', fontsize=10)
        
        # 토글 버튼 생성
        labels = ['Dynamic Objects (Ch0)', 'Static Environment (Ch1)', 'Ground Class ID (Ch2)', 'Ego Position', 'Direction Vectors']
        visibility = [self.channel_visibility['obstacles'], 
                      self.channel_visibility['static_environment'],
                      self.channel_visibility['ground_class_id'],
                      self.channel_visibility['ego'],
                      self.channel_visibility['directions']]
        
        self.check_buttons = CheckButtons(self.ax_toggle, labels, visibility)
        
        # 버튼 크기 조정 (더 작게) - matplotlib 버전 호환성 처리
        try:
            if hasattr(self.check_buttons, 'rectangles'):
                for rect in self.check_buttons.rectangles:
                    rect.set_width(0.08)
                    rect.set_height(0.08)
        except AttributeError:
            pass  # matplotlib 버전에 따라 rectangles 속성이 없을 수 있음

        # X 표시 크기 조정
        try:
            if hasattr(self.check_buttons, 'lines'):
                for line in self.check_buttons.lines:
                    for l in line:
                        l.set_markersize(4)  # X 표시 크기 축소
        except AttributeError:
            pass  # matplotlib 버전에 따라 lines 속성이 없을 수 있음
        
        # 버튼 클릭 이벤트 연결
        def on_toggle(label):
            if 'Ch0' in label:
                self.channel_visibility['obstacles'] = not self.channel_visibility['obstacles']
            elif 'Ch1' in label:
                self.channel_visibility['static_environment'] = not self.channel_visibility['static_environment']
            elif 'Ch2' in label:
                self.channel_visibility['ground_class_id'] = not self.channel_visibility['ground_class_id']
            elif 'Ego' in label:
                self.channel_visibility['ego'] = not self.channel_visibility['ego']
            elif 'Direction' in label:
                self.channel_visibility['directions'] = not self.channel_visibility['directions']
            self.update_frame()
        
        self.check_buttons.on_clicked(on_toggle)
    
    def init_graphics(self):
        """그래픽 요소를 초기화합니다."""
        # Grid view 설정
        self.ax_grid.set_aspect('equal')
        self.ax_grid.set_xlabel('Column Index (y coordinate)')
        self.ax_grid.set_ylabel('Row Index (x coordinate)')
        self.ax_grid.set_title('Ego-Centric Grid View - (0,0) at bottom-left')
        
        # 마우스 좌표 표시 형식 커스터마이징
        def format_coord(x, y):
            col = int(x + 0.5)
            row = int(y + 0.5)
            if 0 <= col < self.grid_cols and 0 <= row < self.grid_rows:
                return f'x={row}, y={col}'  # row를 x로, col을 y로 표시
            else:
                return f'x={row}, y={col}'
        self.ax_grid.format_coord = format_coord
        
        # 초기 Grid 이미지 - RGB 이미지로 초기화
        empty_rgb = np.ones((self.grid_rows, self.grid_cols, 3))  # 흰색 배경
        self.artists['grid_image'] = self.ax_grid.imshow(
            empty_rgb, origin='lower', interpolation='nearest'
        )
        
        # Ego 마커 초기화 (Grid가 아닌 별도 표시)
        self.artists['ego_marker'] = self.ax_grid.plot(
            [], [], 'g*', markersize=10, label='Ego', zorder=100  # 크기 절반으로, 최상위 레이어
        )[0]
        
        # 방향 벡터 화살표 초기화
        self.artists['near_arrow'] = None
        self.artists['dest_arrow'] = None
        
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
            0.05, 0.7, '', transform=self.ax_info.transAxes,
            verticalalignment='top', fontsize=10, family='monospace'
        )
    
    def _create_integrated_grid(self, obstacles_grid, static_env_grid, class_id_grid, ego_pos=None):
        """
        토글 상태에 따라 각 채널을 개별적으로 시각화
        Channel 0 (obstacles): 동적 객체만 표시
        Channel 1 (static_environment): 정적 환경만 표시  
        Channel 2 (ground_class_id): 지면 색상만 표시
        ego_pos: (row, col) 형태의 Ego 위치
        """
        import matplotlib.colors as mcolors
        
        # RGB 이미지 초기화 (white background)
        rgb_image = np.ones((obstacles_grid.shape[0], obstacles_grid.shape[1], 3))
        
        # 디버그: 첫 번째 호출 시 데이터 분포 출력
        if not hasattr(self, '_debug_printed'):
            self._debug_printed = True
            print(f"\n[DEBUG] Creating integrated grid:")
            print(f"  obstacles_grid shape: {obstacles_grid.shape}")
            print(f"  obstacles unique values: {np.unique(obstacles_grid)}")
            if static_env_grid is not None:
                print(f"  static_env_grid shape: {static_env_grid.shape}")
                print(f"  static_env unique values: {np.unique(static_env_grid)}")
            if class_id_grid is not None:
                print(f"  class_id_grid shape: {class_id_grid.shape}")
                print(f"  ground_class_id unique values: {np.unique(class_id_grid)}")
        
        # 우선순위 기반 시각화: Channel 0 > Channel 1 > Channel 2
        # 각 채널을 순서대로 그리되, 이미 그려진 부분은 덮어쓰지 않음
        
        # 먼저 모든 픽셀이 비어있는지 추적
        drawn_mask = np.zeros((obstacles_grid.shape[0], obstacles_grid.shape[1]), dtype=bool)
        
        # 1. Channel 0 (Obstacles) - 가장 높은 우선순위
        if hasattr(self, 'channel_visibility') and self.channel_visibility.get('obstacles', True):
            unique_values = np.unique(obstacles_grid)
            for value in unique_values:
                if value == GRID_VALUE_EMPTY:
                    continue
                mask = (obstacles_grid == value) & (~drawn_mask)
                if np.any(mask):
                    color_name = get_grid_color(value, channel=0)
                    rgb_image[mask] = mcolors.to_rgb(color_name)
                    drawn_mask |= mask
        
        # 2. Channel 1 (Static Environment) - 중간 우선순위
        if static_env_grid is not None and hasattr(self, 'channel_visibility') and self.channel_visibility.get('static_environment', True):
            unique_values = np.unique(static_env_grid)
            for value in unique_values:
                if value == GRID_VALUE_EMPTY:
                    continue
                mask = (static_env_grid == value) & (~drawn_mask)
                if np.any(mask):
                    color_name = get_grid_color(value, channel=1)
                    rgb_image[mask] = mcolors.to_rgb(color_name)
                    drawn_mask |= mask
        
        # 3. Channel 2 (Ground Class ID) - 가장 낮은 우선순위
        if class_id_grid is not None and hasattr(self, 'channel_visibility') and self.channel_visibility.get('ground_class_id', True):
            unique_class_ids = np.unique(class_id_grid)
            for ground_class_id in unique_class_ids:
                mask = (class_id_grid == ground_class_id) & (~drawn_mask)
                if np.any(mask):
                    color_str = get_grid_color(ground_class_id, channel=2)
                    if color_str.startswith('#'):
                        color = mcolors.to_rgb(color_str)
                    else:
                        color = mcolors.to_rgb(color_str)
                    rgb_image[mask] = color
                    drawn_mask |= mask
        
        # 4. Ego 위치 표시 (최우선 순위)
        if hasattr(self, 'channel_visibility') and self.channel_visibility.get('ego', True) and ego_pos is not None:
            ego_row, ego_col = ego_pos
            
            if 0 <= ego_row < obstacles_grid.shape[0] and 0 <= ego_col < obstacles_grid.shape[1]:
                # Ego를 녹색으로 표시
                rgb_image[ego_row, ego_col] = mcolors.to_rgb('green')
                # 주변 픽셀도 약간 표시해서 더 잘 보이게
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        r, c = ego_row + dr, ego_col + dc
                        if 0 <= r < obstacles_grid.shape[0] and 0 <= c < obstacles_grid.shape[1]:
                            if dr == 0 and dc == 0:
                                continue  # 중심은 이미 처리함
                            # 주변은 연한 녹색으로
                            rgb_image[r, c] = 0.5 * rgb_image[r, c] + 0.5 * np.array(mcolors.to_rgb('green'))
        
        # 색상 범례 추가를 위한 정보 저장
        self._legend_info = []
        
        # 표시된 채널에 따라 범례 항목 추가
        if hasattr(self, 'channel_visibility') and self.channel_visibility.get('obstacles', True):
            # Channel 0의 값들 (Ego 제외)
            for value in [GRID_VALUE_VEHICLE, 
                         GRID_VALUE_OTHER_OBJECT, GRID_VALUE_PEDESTRIAN]:
                color = get_grid_color(value, channel=0)
                name, _ = get_grid_value_info(value)
                self._legend_info.append((color, name, value))
                
        if hasattr(self, 'channel_visibility') and self.channel_visibility.get('static_environment', True):
            # Channel 1의 값들
            for value in [GRID_VALUE_CLIFF, GRID_VALUE_STATIC_OBSTACLE, 
                         GRID_VALUE_MISSING_POINTS]:
                color = get_grid_color(value, channel=1)
                name, _ = get_grid_value_info(value)
                self._legend_info.append((color, name, value))
                
        if hasattr(self, 'channel_visibility') and self.channel_visibility.get('ground_class_id', True) and class_id_grid is not None:
            # Channel 2의 주요 클래스들 (상위 5개만 표시)
            unique_class_ids, counts = np.unique(class_id_grid, return_counts=True)
            # 카운트 순으로 정렬
            sorted_indices = np.argsort(counts)[::-1]
            shown_count = 0
            for idx in sorted_indices:
                ground_class_id = unique_class_ids[idx]
                count = counts[idx]
                
                # 0(Background)은 생략, 너무 적은 픽셀도 생략
                if ground_class_id == 0 or count < 10:
                    continue
                    
                color_str = get_grid_color(ground_class_id, channel=2)
                name = get_ldm_segmentation_name(int(ground_class_id))
                
                # 색상이 hex인 경우 matplotlib 색상으로 변환
                if color_str.startswith('#'):
                    # hex 색상은 그대로 사용
                    self._legend_info.append((color_str, f"{name}", ground_class_id))
                else:
                    self._legend_info.append((color_str, f"{name}", ground_class_id))
                
                shown_count += 1
                if shown_count >= 5:  # 최대 5개만 표시
                    break
        
        # Ego 범례 추가
        if hasattr(self, 'channel_visibility') and self.channel_visibility.get('ego', True):
            self._legend_info.append(('green', 'Ego', None))
        
        return rgb_image
    
    def update_frame(self):
        """현재 프레임 업데이트"""
        # 현재 Episode와 스텝
        episode = self.episodes[self.current_episode]
        
        # ===== 1. Grid 업데이트 =====
        cache_key = (self.current_episode, self.current_step)
        cached_data = self.observation_cache.get(cache_key)
        
        if cached_data is not None:
            if isinstance(cached_data, dict):
                # 새로운 형식: 3개 채널 별도 저장
                obstacles_grid = cached_data['obstacles']
                static_env_grid = cached_data['static_environment']
                class_id_grid = cached_data['ground_class_id']
            elif isinstance(cached_data, tuple):
                # 이전 버전 호환성
                obstacles_grid, class_id_grid = cached_data
                static_env_grid = None
            else:
                # 더 오래된 버전 호환성
                obstacles_grid = cached_data
                static_env_grid = None
                class_id_grid = None
            
            # Ego 위치 계산
            ego_pos = None
            if self.channel_visibility['ego']:
                # 첫 번째 프레임에서는 ego가 중앙(49, 49)에 위치
                if self.current_step == 0:
                    ego_pos = (self.grid_rows // 2, self.grid_cols // 2)
                else:
                    # 이후 프레임에서는 expert_ego_pos 사용
                    if hasattr(episode, 'expert_ego_pos') and self.current_step < len(episode.expert_ego_pos):
                        expert_pos = episode.expert_ego_pos[self.current_step]
                        if expert_pos is not None and isinstance(expert_pos, tuple):
                            ego_pos = expert_pos  # (row, col) 형태
                        else:
                            ego_pos = (self.grid_rows // 2, self.grid_cols // 2)
                    else:
                        ego_pos = (self.grid_rows // 2, self.grid_cols // 2)
            
            # 통합된 그리드 생성 및 업데이트 - ego_pos도 전달
            integrated_rgb = self._create_integrated_grid(obstacles_grid, static_env_grid, class_id_grid, ego_pos)
            self.artists['grid_image'].set_data(integrated_rgb)
            
            # Ego 마커는 이제 그리드에 직접 그려지므로 별도 업데이트 불필요
            self.artists['ego_marker'].set_data([], [])
            
            # 방향 벡터 표시
            if self.channel_visibility['directions']:
                # 기존 화살표 제거
                if self.artists['near_arrow']:
                    self.artists['near_arrow'].remove()
                    self.artists['near_arrow'] = None
                if self.artists['dest_arrow']:
                    self.artists['dest_arrow'].remove()
                    self.artists['dest_arrow'] = None
                
                # 현재 상태에서 방향 벡터 가져오기
                state = episode.states[self.current_step]
                near_dir = state.get('near_destination_direction')
                dest_dir = state.get('final_destination_direction')
                
                # Ego 위치 계산 (방향 벡터용 - Ego 가시성과 무관)
                # 첫 번째 프레임에서는 ego가 중앙(49, 49)에 위치
                if self.current_step == 0:
                    arrow_ego_pos = (self.grid_rows // 2, self.grid_cols // 2)
                else:
                    # 이후 프레임에서는 expert_ego_pos 사용
                    if hasattr(episode, 'expert_ego_pos') and self.current_step < len(episode.expert_ego_pos):
                        expert_pos = episode.expert_ego_pos[self.current_step]
                        if expert_pos is not None and isinstance(expert_pos, tuple):
                            arrow_ego_pos = expert_pos  # (row, col) 형태
                        else:
                            arrow_ego_pos = (self.grid_rows // 2, self.grid_cols // 2)
                    else:
                        arrow_ego_pos = (self.grid_rows // 2, self.grid_cols // 2)
                
                ego_row, ego_col = arrow_ego_pos
                
                # Near destination 화살표
                if near_dir and near_dir.get('x') is not None and near_dir.get('y') is not None:
                    # 방향 벡터는 이미 차량 좌표계 기준
                    # x는 전방(양수가 앞), y는 좌우(양수가 오른쪽)
                    dx = near_dir['x'] * 10  # x는 전방 (row 방향 상방 +)
                    dy = near_dir['y'] * 10  # y는 우측방 (col 방향 우측 +)
                    # matplotlib 좌표계는 x축 좌우(우가 +), y축 상하(상이 +) 임. 그래서 거꾸로 넣어줘야함
                    self.artists['near_arrow'] = self.ax_grid.arrow(
                        ego_col, ego_row, dy, dx,
                        head_width=2, head_length=1, fc='blue', ec='blue',
                        alpha=0.7, zorder=90, label='Near Destination'
                    )
                
                # Final destination 화살표
                if dest_dir and dest_dir.get('x') is not None and dest_dir.get('y') is not None:
                    # 방향 벡터는 이미 차량 좌표계 기준
                    # x는 전방(양수가 앞), y는 좌우(양수가 오른쪽)
                    dx = dest_dir['x'] * 15  # x는 전방 (row 방향 상방 +), 더 길게
                    dy = dest_dir['y'] * 15  # y는 우측방 (col 방향 우측 +), 더 길게
                    # matplotlib 좌표계는 x축 좌우(우가 +), y축 상하(상이 +) 임. 그래서 거꾸로 넣어줘야함
                    self.artists['dest_arrow'] = self.ax_grid.arrow(
                        ego_col, ego_row, dy, dx,
                        head_width=2, head_length=1, fc='red', ec='red',
                        alpha=0.5, zorder=89, label='Final Destination'
                    )
            else:
                # 방향 벡터 숨기기
                if self.artists['near_arrow']:
                    self.artists['near_arrow'].remove()
                    self.artists['near_arrow'] = None
                if self.artists['dest_arrow']:
                    self.artists['dest_arrow'].remove()
                    self.artists['dest_arrow'] = None
            
            # 범례 업데이트 (채널 가시성에 따라)
            legend_elements = []
            
            # Channel 0: 동적 객체
            if self.channel_visibility.get('obstacles', True):
                for value in [GRID_VALUE_VEHICLE, GRID_VALUE_OTHER_OBJECT, GRID_VALUE_PEDESTRIAN]:
                    color = get_grid_color(value, channel=0)
                    name, _ = get_grid_value_info(value)
                    legend_elements.append(patches.Patch(color=color, label=f'{name} ({value})'))
            
            # Channel 1: 정적 환경
            if self.channel_visibility.get('static_environment', True):
                for value in [GRID_VALUE_CLIFF, GRID_VALUE_STATIC_OBSTACLE, GRID_VALUE_MISSING_POINTS]:
                    color = get_grid_color(value, channel=1)
                    name, _ = get_grid_value_info(value)
                    legend_elements.append(patches.Patch(color=color, label=f'{name} ({value})'))
            
            # Channel 2: 지면 클래스 (상위 5개만)
            if self.channel_visibility.get('ground_class_id', True) and class_id_grid is not None:
                unique_class_ids, counts = np.unique(class_id_grid, return_counts=True)
                sorted_indices = np.argsort(counts)[::-1]
                shown_count = 0
                for idx in sorted_indices[:5]:  # 최대 5개
                    ground_class_id = unique_class_ids[idx]
                    if ground_class_id == 0 or counts[idx] < 10:  # Background 또는 너무 적은 픽셀 제외
                        continue
                    color_str = get_grid_color(ground_class_id, channel=2)
                    name = get_ldm_segmentation_name(int(ground_class_id))
                    legend_elements.append(patches.Patch(color=color_str, label=f'{name}'))
                    shown_count += 1
                    if shown_count >= 5:
                        break
            
            # Ego
            if self.channel_visibility.get('ego', True):
                legend_elements.append(patches.Patch(color='green', label='Ego'))
            
            # 방향 벡터
            if self.channel_visibility.get('directions', True):
                state = episode.states[self.current_step]
                near_dir = state.get('near_destination_direction')
                dest_dir = state.get('final_destination_direction')
                if near_dir:
                    legend_elements.append(patches.Patch(color='blue', label='Near Destination'))
                if dest_dir:
                    legend_elements.append(patches.Patch(color='red', label='Final Destination'))
            
            # 범례 업데이트
            if legend_elements:
                self.ax_grid.legend(handles=legend_elements, loc='upper right', fontsize=8)
            else:
                # 범례가 없으면 기존 범례 제거
                if self.ax_grid.get_legend():
                    self.ax_grid.get_legend().remove()
        
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
        
        # Value (target_values 우선 표시, 없으면 values)
        if hasattr(episode, 'target_values') and self.current_step < len(episode.target_values):
            value = episode.target_values[self.current_step]
            step_text += f"Target Value: {value:.3f}\n"
        elif self.current_step < len(episode.values):
            value = episode.values[self.current_step]
            step_text += f"Value: {value:.3f}\n"
        
        # Done
        if self.current_step < len(episode.dones):
            done = episode.dones[self.current_step]
            step_text += f"Done: {done}\n"
        
        # Policy (Top 5)
        if self.current_step < len(episode.pi) and episode.pi[self.current_step] is not None:
            pi = episode.pi[self.current_step]
            step_text += f"\nPolicy (Top 5):\n"
            top_actions = np.argsort(pi)[-5:][::-1]
            for i, action_idx in enumerate(top_actions):
                action = Action.from_index(action_idx)
                prob = pi[action_idx]
                step_text += f"  {i+1}. Action {action_idx} ({action.position.value}): {prob:.3f}\n"
        else:
            step_text += f"\nPolicy: None (GT data)\n"
        
        # Grid 통계 - 통합된 정보 표시
        if obstacles_grid is not None:
            step_text += f"\n[Obstacles/Static Layer]:\n"
            unique_values, counts = np.unique(obstacles_grid, return_counts=True)
            for val, count in zip(unique_values, counts):
                if val != 0:  # 0이 아닌 값만 표시
                    name, _ = get_grid_value_info(val)
                    step_text += f"  {name} ({val}): {count} cells\n"
        
        # Class ID 통계
        if class_id_grid is not None:
            step_text += f"\n[Class ID Layer]:\n"
            unique_class_ids, counts = np.unique(class_id_grid, return_counts=True)
            # 주요 클래스만 표시 (너무 많으면 상위 10개만)
            sorted_indices = np.argsort(counts)[::-1]
            shown_count = 0
            for idx in sorted_indices:
                ground_class_id = unique_class_ids[idx]
                count = counts[idx]
                if shown_count >= 10 and count < 100:  # 10개 이상 표시했고 카운트가 작으면 중단
                    remaining = len(sorted_indices) - shown_count
                    if remaining > 0:
                        step_text += f"  ... and {remaining} more classes\n"
                    break
                
                if ground_class_id == 0:
                    if count > 1000:  # 배경이 너무 많으면 간략히
                        step_text += f"  Background: {count} cells\n"
                elif 0 < ground_class_id < 100:
                    # LDM semantic class
                    name = get_ldm_segmentation_name(int(ground_class_id))
                    step_text += f"  {name}: {count} cells\n"
                else:
                    # Object class
                    name = get_object_name(int(ground_class_id))
                    step_text += f"  {name}: {count} cells\n"
                shown_count += 1
        
        # 추가 정보 표시 (state가 dict인 경우)
        if self.current_step < len(episode.states) and isinstance(episode.states[self.current_step], dict):
            state = episode.states[self.current_step]
            
            # Ego 위치 (expert_ego_pos로 대체)
            if hasattr(episode, 'expert_ego_pos') and self.current_step < len(episode.expert_ego_pos):
                expert_pos = episode.expert_ego_pos[self.current_step]
                if expert_pos is not None and isinstance(expert_pos, tuple):
                    step_text += f"\nEgo Position (grid): ({expert_pos[0]}, {expert_pos[1]})\n"
            
            # Simulated Ego Position (state에 저장된 경우)
            if 'simulated_ego_pos' in state and state['simulated_ego_pos']:
                sim_pos = state['simulated_ego_pos']
                step_text += f"Simulated Ego Pos: ({sim_pos[0]}, {sim_pos[1]})\n"
            
            
            # 방향 정보 (수치값 포함)
            if 'near_destination_direction' in state and state['near_destination_direction']:
                near_dir = state['near_destination_direction']
                if isinstance(near_dir, dict) and 'x' in near_dir and 'y' in near_dir:
                    step_text += f"Near Direction: x={near_dir['x']:.3f}, y={near_dir['y']:.3f}\n"
                else:
                    step_text += f"Near Direction: {near_dir}\n"
            
            if 'final_destination_direction' in state and state['final_destination_direction']:
                dest_dir = state['final_destination_direction']
                if isinstance(dest_dir, dict) and 'x' in dest_dir and 'y' in dest_dir:
                    step_text += f"Final Direction: x={dest_dir['x']:.3f}, y={dest_dir['y']:.3f}\n"
                else:
                    step_text += f"Final Direction: {dest_dir}\n"
            
            # Static Environment 정보
            if 'static_environment' in state and state['static_environment'] is not None:
                static_env = state['static_environment']
                if hasattr(static_env, 'numpy'):
                    static_env = static_env.numpy()
                
                if isinstance(static_env, np.ndarray):
                    # 2D array인 경우
                    cliff_count = np.sum(static_env == GRID_VALUE_CLIFF)
                    static_obs_count = np.sum(static_env == GRID_VALUE_STATIC_OBSTACLE)
                    missing_count = np.sum(static_env == GRID_VALUE_MISSING_POINTS)
                    
                    step_text += f"\nStatic Environment:\n"
                    if cliff_count > 0:
                        step_text += f"  Cliff: {cliff_count} cells\n"
                    if static_obs_count > 0:
                        step_text += f"  Static Obstacles: {static_obs_count} cells\n"
                    if missing_count > 0:
                        step_text += f"  Missing Points: {missing_count} cells\n"
        
        self.artists['step_text'].set_text(step_text)
        
        # ===== 4. Policy 막대 그래프 업데이트 =====
        if self.artists['policy_bars'] is not None:
            self.artists['policy_bars'].remove()
        
        if self.current_step < len(episode.pi) and episode.pi[self.current_step] is not None:
            pi = episode.pi[self.current_step]
            # 상위 9개 액션만 표시
            top_actions = np.argsort(pi)[-9:]
            top_probs = pi[top_actions]
            
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
        # 채널 가시성 상태 표시
        visibility_status = f"Ch0: {'ON' if self.channel_visibility['obstacles'] else 'OFF'}, " \
                           f"Ch1: {'ON' if self.channel_visibility['static_environment'] else 'OFF'}, " \
                           f"Ch2: {'ON' if self.channel_visibility['ground_class_id'] else 'OFF'}, " \
                           f"Ego: {'ON' if self.channel_visibility['ego'] else 'OFF'}, " \
                           f"Dir: {'ON' if self.channel_visibility['directions'] else 'OFF'}"
        
        self.fig.suptitle(
            f'MuZero Episode Visualization {status} - {visibility_status}\n'
            f'(Space: Play/Pause, ←/→: Step, ↑/↓: Episode, 1/2/3/4/5: Toggle Channels/Ego/Dir, ESC/Q: Quit)',
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
        
        # Figure 생성 - 토글 버튼을 위한 공간 추가
        self.fig = plt.figure(figsize=(22, 10))
        gs = self.fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.15])
        
        self.ax_grid = self.fig.add_subplot(gs[0, 0])
        self.ax_info = self.fig.add_subplot(gs[0, 1])
        self.ax_toggle = self.fig.add_subplot(gs[0, 2])
        
        # 토글 버튼 생성
        self.setup_toggle_buttons()
        
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
        
        # Episode 전체를 한 그림에 표시 - 2행으로 변경 (integrated grid, info)
        num_steps = min(num_frames, len(episode))
        fig, axes = plt.subplots(2, num_steps, figsize=(4*num_steps, 8))
        
        if num_steps == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle(f'Episode {ep_idx + 1} (Total: {len(episode)} steps, Reward: {sum(episode.rewards):.2f})', 
                     fontsize=16)
        
        # 균등하게 분포된 스텝 선택
        step_indices = np.linspace(0, len(episode)-1, num_steps, dtype=int)
        
        for i, step_idx in enumerate(step_indices):
            # 상단: 통합 그리드
            ax_grid = axes[0, i]
            # 하단: 정보
            ax_info = axes[1, i]
            
            state = episode.states[step_idx]
            
            # state가 dict인 경우 obstacles와 static_environment 병합
            if isinstance(state, dict):
                obstacles = state.get('obstacles', state.get('grid'))
                static_env = state.get('static_environment')
                ground_class_id = state.get('ground_class_id')  # Channel 2: Class ID
                
                # numpy로 변환
                if hasattr(obstacles, 'numpy'):
                    obstacles_np = obstacles.numpy()
                else:
                    obstacles_np = obstacles
                    
                if len(obstacles_np.shape) == 3:
                    obstacles_np = obstacles_np[0]
                
                # static_environment 처리
                static_env_np = None
                if static_env is not None and not isinstance(static_env, dict):
                    if hasattr(static_env, 'numpy'):
                        static_env_np = static_env.numpy()
                    else:
                        static_env_np = static_env
                        
                    if len(static_env_np.shape) == 3:
                        static_env_np = static_env_np[0]
                
                # ground_class_id 처리
                class_id_np = None
                if ground_class_id is not None and not isinstance(ground_class_id, dict):
                    if hasattr(ground_class_id, 'numpy'):
                        class_id_np = ground_class_id.numpy()
                    else:
                        class_id_np = ground_class_id
                    if len(class_id_np.shape) == 3:
                        class_id_np = class_id_np[0]
            else:
                # 기존 tensor 형태 - 3채널 observation
                if hasattr(state, 'numpy'):
                    obs_full = state.numpy()
                else:
                    obs_full = state
                
                # 3채널 데이터 처리
                if len(obs_full.shape) == 3 and obs_full.shape[0] >= 3:
                    # Channel 0: obstacles, Channel 1: static_env, Channel 2: ground_class_id
                    obstacles_np = obs_full[0]
                    static_env_np = obs_full[1]
                    class_id_np = obs_full[2]
                else:
                    # 1채널만 있는 경우
                    obstacles_np = obs_full[0] if len(obs_full.shape) == 3 else obs_full
                    static_env_np = None
                    class_id_np = None
            
            # 통합 그리드 생성 - 각 채널을 개별적으로 전달
            visualizer = EpisodeInteractiveVisualizer([], config, 0, False)
            # 모든 채널을 표시하도록 설정 (Ego, Directions 포함)
            visualizer.channel_visibility = {'obstacles': True, 'static_environment': True, 'ground_class_id': True, 'ego': True, 'directions': True}
            
            # Ego 위치 계산
            if step_idx == 0:
                # 첫 프레임에서는 중앙
                ego_pos = (config.observation_shape[1] // 2, config.observation_shape[2] // 2)  # (row, col)
            else:
                # 이후 프레임에서는 expert_ego_pos 사용
                if hasattr(episode, 'expert_ego_pos') and step_idx < len(episode.expert_ego_pos):
                    expert_pos = episode.expert_ego_pos[step_idx]
                    if expert_pos is not None and isinstance(expert_pos, tuple):
                        ego_pos = expert_pos  # (row, col) 형태
                    else:
                        ego_pos = (config.observation_shape[1] // 2, config.observation_shape[2] // 2)
                else:
                    ego_pos = (config.observation_shape[1] // 2, config.observation_shape[2] // 2)
            
            integrated_rgb = visualizer._create_integrated_grid(obstacles_np, static_env_np, class_id_np, ego_pos)
            
            # 통합 그리드 시각화
            ax_grid.imshow(integrated_rgb, origin='lower')
            ax_grid.set_title(f'Step {step_idx}: Integrated Grid')
            ax_grid.set_xlabel('Column Index')
            ax_grid.set_ylabel('Row Index')
            
            # 색상 범례 추가
            if i == 0 and hasattr(visualizer, '_legend_info'):
                legend_elements = []
                for color, label, value in visualizer._legend_info:
                    legend_elements.append(
                        patches.Patch(color=color, label=f'{label} ({value})')
                    )
                ax_grid.legend(handles=legend_elements, loc='upper right', fontsize=6)
            
            # Ego 위치는 이미 integrated_rgb에 포함되어 있음
            # 그리드 중심점 표시 (Episode 시작점)
            center_row, center_col = config.observation_shape[1] // 2, config.observation_shape[2] // 2
            ax_grid.axhline(y=center_row, color='gray', alpha=0.3, linestyle='--')
            ax_grid.axvline(x=center_col, color='gray', alpha=0.3, linestyle='--')
            
            # 그리드 선
            for j in range(0, obstacles_np.shape[0], 10):
                ax_grid.axhline(y=j, color='gray', alpha=0.3, linewidth=0.5)
            for j in range(0, obstacles_np.shape[1], 10):
                ax_grid.axvline(x=j, color='gray', alpha=0.3, linewidth=0.5)
            
            # 하단: 정보
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
                if step_idx < len(episode.pi) and episode.pi[step_idx] is not None:
                    pi = episode.pi[step_idx]
                    top_actions = np.argsort(pi)[-3:][::-1]
                    info_text += f"\nTop Actions:\n"
                    for j, a in enumerate(top_actions):
                        info_text += f"{j+1}. {a}: {pi[a]:.3f}\n"
            
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
    
    # MuZero 설정 로드하여 기본값 설정
    config = MuZeroConfig()
    
    parser.add_argument('--num-episodes', type=int, default=5,
                       help='로드할 Episode 수 (기본값: 5)')
    parser.add_argument('--episode-step-length', type=int, default=config.max_episode_step_length,
                       help=f'Episode 스텝 길이 (기본값: {config.max_episode_step_length})')
    parser.add_argument('--overlap', type=int, default=config.gt_episode_overlap,
                       help=f'Episode 간 겹침 (기본값: {config.gt_episode_overlap})')
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
    
    # 데이터 경로 오버라이드
    if args.data_path:
        config.gt_data_path = Path(args.data_path)
    
    # Episode 로드
    print("Loading expert data and creating episodes...")
    data_loader = ExpertDataLoader(config)
    episodes = data_loader.create_episodes_from_json(
        num_episodes=args.num_episodes,
        episode_step_length=args.episode_step_length,
        overlap=args.overlap
    )
    
    if not episodes:
        print("No episodes created!")
        return
    
    print(f"\nLoaded {len(episodes)} episodes")
    for i, ep in enumerate(episodes):
        print(f"  Episode {i+1}: {len(ep)} steps, "
              f"total reward: {sum(ep.rewards):.2f}")
        # 첫 Episode의 expert_ego_pos 디버그
        if i == 0 and hasattr(ep, 'expert_ego_pos'):
            print(f"\n[DEBUG] Episode 1 expert_ego_pos in main:")
            print(f"  Total positions: {len(ep.expert_ego_pos)}")
            if len(ep.expert_ego_pos) > 0:
                print(f"  First: {ep.expert_ego_pos[0]}")
                print(f"  Last: {ep.expert_ego_pos[-1]}")
                # 처음 3개 출력
                for j in range(min(3, len(ep.expert_ego_pos))):
                    print(f"    Step {j}: {ep.expert_ego_pos[j]}")
    
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