#!/usr/bin/env python3
"""
GT(Ground Truth) 데이터 시각화 스크립트 - 성능 최적화 버전
db3_extractor.py로 추출한 JSONL 파일의 GT 데이터를 빠르게 시각화합니다.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection, PatchCollection
import numpy as np
import argparse
from pathlib import Path
import cv2
from typing import List, Dict, Tuple, Optional
import math
from collections import defaultdict
import time


def load_jsonl_data(filepath: Path) -> List[Dict]:
    """JSONL 파일에서 데이터를 로드합니다."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def preprocess_frame_data(frame_data: Dict, ego_info: Dict, view_range: float) -> Dict:
    """프레임 데이터를 사전 처리하여 캐시합니다."""
    cache = {
        'objects': [],
        'ldm_points': [],
        'trajectory_points': []
    }
    
    # 자차 정보가 없으면 기본값 사용
    if not ego_info:
        ego_info = {'x': 0, 'y': 0, 'yaw': 0}
    
    ego_x, ego_y, ego_yaw = ego_info['x'], ego_info['y'], ego_info['yaw']
    cos_yaw = np.cos(-ego_yaw)
    sin_yaw = np.sin(-ego_yaw)
    
    # 객체 처리
    objects = frame_data.get('input_environment', {}).get('_objects', [])
    for obj in objects:
        global_center = obj.get('_global_center', {})
        global_x = global_center.get('_x', 0)
        global_y = global_center.get('_y', 0)
        
        # 자차 좌표계로 변환
        dx = global_x - ego_x
        dy = global_y - ego_y
        ego_x_obj = dx * cos_yaw - dy * sin_yaw
        ego_y_obj = dx * sin_yaw + dy * cos_yaw
        
        # 화면 좌표계로 변환
        screen_x = -ego_y_obj
        screen_y = ego_x_obj
        
        # view_range 내의 객체만 처리
        if abs(screen_x) <= view_range/2 and abs(screen_y) <= view_range/2:
            obj_data = {
                'screen_x': screen_x,
                'screen_y': screen_y,
                'class_id': obj.get('_class_id', -1),
                'id': obj.get('_id', ''),
                'velocity': obj.get('_global_velocity', 0),
                'global_yaw': obj.get('_global_yaw_heading', 0),
                'global_points': [],
                'global_x': global_x,
                'global_y': global_y
            }
            
            # 포인트 처리
            for point in obj.get('_global_points', []):
                pt_x = point.get('_x', 0)
                pt_y = point.get('_y', 0)
                pt_dx = pt_x - ego_x
                pt_dy = pt_y - ego_y
                pt_ego_x = pt_dx * cos_yaw - pt_dy * sin_yaw
                pt_ego_y = pt_dx * sin_yaw + pt_dy * cos_yaw
                pt_screen_x = -pt_ego_y
                pt_screen_y = pt_ego_x
                
                if abs(pt_screen_x) <= view_range/2 and abs(pt_screen_y) <= view_range/2:
                    obj_data['global_points'].append((pt_screen_x, pt_screen_y))
            
            cache['objects'].append(obj_data)
    
    # LDM 포인트 처리
    ldm_points = frame_data.get('input_environment', {}).get('_ldm_points', [])
    for point in ldm_points:
        global_x = point.get('x', 0)
        global_y = point.get('y', 0)
        
        dx = global_x - ego_x
        dy = global_y - ego_y
        ego_x_pt = dx * cos_yaw - dy * sin_yaw
        ego_y_pt = dx * sin_yaw + dy * cos_yaw
        screen_x = -ego_y_pt
        screen_y = ego_x_pt
        
        if abs(screen_x) < view_range/2 and abs(screen_y) < view_range/2:
            cache['ldm_points'].append({
                'x': screen_x,
                'y': screen_y,
                'obstacles_type': point.get('obstacles_type', 0),
                'class_id': point.get('class_id', 0)
            })
    
    # GT trajectory 처리
    gt_trajectory = frame_data.get('gt_trajectory', [])
    for gt in gt_trajectory:
        pose = gt.get('_pose', {}).get('_pose', {})
        position = pose.get('_position', {})
        gt_x = position.get('_x', 0)
        gt_y = position.get('_y', 0)
        
        dx = gt_x - ego_x
        dy = gt_y - ego_y
        ego_x_traj = dx * cos_yaw - dy * sin_yaw
        ego_y_traj = dx * sin_yaw + dy * cos_yaw
        screen_x = -ego_y_traj
        screen_y = ego_x_traj
        
        cache['trajectory_points'].append((screen_x, screen_y))
    
    return cache


def extract_objects_info(frame_data: Dict) -> Tuple[List[Dict], Dict, int, List[Dict], Optional[Dict], Optional[Dict], List[Dict]]:
    """프레임 데이터에서 객체 정보를 추출합니다."""
    objects = frame_data.get('input_environment', {}).get('_objects', [])
    ego_trajectory = frame_data.get('gt_trajectory', [])
    
    # 자차 위치 (첫 번째 trajectory 데이터의 pose)
    ego_info = {}
    if ego_trajectory and len(ego_trajectory) > 0:
        pose = ego_trajectory[0].get('_pose', {}).get('_pose', {})
        position = pose.get('_position', {})
        orientation = pose.get('_orientation', {})
        
        # 쿼터니언에서 yaw 계산
        x, y, z, w = orientation.get('_x', 0), orientation.get('_y', 0), orientation.get('_z', 0), orientation.get('_w', 1)
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        
        # 속도 정보 추출
        linear_vel = ego_trajectory[0].get('_linear_vel', 0.0)
        
        ego_info = {
            'x': position.get('_x', 0),
            'y': position.get('_y', 0),
            'yaw': yaw,
            'velocity': linear_vel
        }
    
    # 타임스탬프 추출
    timestamp = 0
    header = frame_data.get('input_environment', {}).get('_header', {})
    stamp = header.get('_stamp', {})
    if stamp:
        timestamp = stamp.get('_sec', 0) * 1_000_000_000 + stamp.get('_nanosec', 0)
    
    # 방위벡터 추출
    near_direction = frame_data.get('input_environment', {}).get('near_direction_vector')
    destination_direction = frame_data.get('input_environment', {}).get('destination_direction_vector')
    
    # LDM 포인트 추출
    ldm_points = frame_data.get('input_environment', {}).get('_ldm_points', [])
    
    return objects, ego_info, timestamp, ego_trajectory, near_direction, destination_direction, ldm_points


def get_object_color(class_id: int) -> str:
    """클래스 ID에 따른 색상을 반환합니다."""
    color_map = {
        100: 'red',      # 보행자
        202: 'blue',     # 차량  
        203: 'orange',   # 자전거
        204: 'purple',   # 트럭
        205: 'green',    # 버스
        300: 'gray',     # 기타
        0: 'darkred',    # Unknown
        -1: 'black',     # 미분류
    }
    return color_map.get(class_id, 'black')


def get_ldm_color(obstacles_type: int) -> Optional[str]:
    """LDM obstacles_type에 따른 색상을 반환합니다."""
    color_map = {
        1: 'lime',       # cliff - 연두색
        2: 'yellow',     # static_obstacles - 노란색
        3: 'purple'      # missing_points - 보라색
    }
    if obstacles_type >= 1:
        return color_map.get(obstacles_type, 'gray')
    return None


def get_ldm_segmentation_color(class_id: int) -> str:
    """LDM class_id (segmentation)에 따른 색상을 반환합니다."""
    color_map = {
        0: '#000000',   # Background
        1: '#3D3DF5',   # Blocks
        2: '#CC3366',   # Colored Road
        3: '#F078F0',   # Cement Asphalt
        4: '#008000',   # Crosswalk
        5: '#733380',   # Speed Bumps
        6: '#FF6A4D',   # Manhole
        7: '#5986B3',   # Curbs
        8: '#00FF00',   # Grass
        9: '#CC9933',   # Soil/Gravel
        10: '#34D1B7',  # Grating
        11: '#FFCC33',  # Guide Blocks
        12: '#DBDB8D',  # Rock
        13: '#FF007C',  # Unidentified
        14: '#942701',  # Wood
        15: '#AEC7E8',  # Iron
        16: '#A7FFD8',  # Artificial Turf
        17: '#00498C',  # Water
        18: '#000000',  # Mat
        19: '#F7ECE6',  # Unlabeled
        20: '#6000C0',  # Restricted
        21: '#FFFFFF',  # Lane Straight White
        22: '#F59331',  # Lane Straight Yellow
        23: '#DBDB8D',  # Lane Dashed White
        24: '#000000',  # Lane Dashed Yellow
        25: '#FCE7A7',  # Guide Blocks Stop
        99: '#808080'   # 기타/미분류
    }
    return color_map.get(class_id, '#808080')


def find_matching_image(timestamp: int, img_dir: Path, db3_name: str = None) -> Optional[Path]:
    """타임스탬프를 기반으로 매칭되는 이미지를 찾습니다."""
    if not img_dir or not img_dir.exists():
        return None
    
    # 모든 이미지 파일 가져오기
    if db3_name:
        pattern = f"cam_f_*_{db3_name}.jpg"
    else:
        pattern = "cam_f_*.jpg"
    
    all_images = list(img_dir.glob(pattern))
    if not all_images:
        return None
    
    # 각 이미지의 타임스탬프 추출
    image_timestamps = []
    for img_path in all_images:
        # 파일명에서 타임스탬프 추출: cam_f_{timestamp}_{db3_name}.jpg
        parts = img_path.stem.split('_')
        if len(parts) >= 3 and parts[2].isdigit():
            img_timestamp = int(parts[2])
            image_timestamps.append((img_timestamp, img_path))
    
    if not image_timestamps:
        return None
    
    # 가장 가까운 타임스탬프 찾기
    image_timestamps.sort(key=lambda x: abs(x[0] - timestamp))
    closest_timestamp, closest_path = image_timestamps[0]
    
    # 최대 허용 차이: 1초 (1,000,000,000 나노초)
    max_diff = 1_000_000_000
    if abs(closest_timestamp - timestamp) <= max_diff:
        return closest_path
    
    return None


class FastInteractiveVisualizer:
    """성능 최적화된 인터렉티브 시각화 클래스"""
    def __init__(self, data: List[Dict], img_dir: Optional[Path], 
                 view_range: float, db3_name: Optional[str] = None, debug: bool = False,
                 interval: int = 100):
        self.data = data
        self.img_dir = img_dir
        self.view_range = view_range
        self.db3_name = db3_name
        self.debug = debug
        self.interval = interval
        self.current_frame = 0
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.anim = None
        self.playing = False
        self.prev_objects = {}
        self.ldm_mode = 1
        
        # 캐시 및 성능 최적화 변수
        self.frame_cache = {}
        self.image_cache = {}
        
        # 그래픽 요소 저장
        self.artists = {
            'ego_rect': None,
            'ego_center': None,
            'ego_arrow': None,
            'objects': [],
            'object_points': None,
            'object_labels': [],
            'ldm_scatter': None,
            'trajectory_circles': [],
            'near_arrow': None,
            'dest_arrow': None,
            'info_text': None,
            'ego_info_text': None,
            'camera_image': None
        }
        
        # 전처리
        self.preprocess_all_frames()
        
    def preprocess_all_frames(self):
        """모든 프레임 데이터를 사전 처리하여 캐시합니다."""
        print("Preprocessing frames for better performance...")
        start_time = time.time()
        
        for i, frame_data in enumerate(self.data):
            objects, ego_info, timestamp, gt_trajectory, near_direction, destination_direction, ldm_points = extract_objects_info(frame_data)
            
            # 프레임 캐시에 필요한 정보 저장
            self.frame_cache[i] = {
                'ego_info': ego_info,
                'timestamp': timestamp,
                'near_direction': near_direction,
                'destination_direction': destination_direction,
                'preprocessed': preprocess_frame_data(frame_data, ego_info, self.view_range),
                'raw_objects': objects  # 속도 벡터 계산용
            }
            
            # 이미지 경로 캐싱
            if self.img_dir:
                image_path = find_matching_image(timestamp, self.img_dir, self.db3_name)
                if image_path:
                    self.image_cache[i] = image_path
        
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
            # 다음 프레임
            if self.current_frame < len(self.data) - 1:
                self.current_frame += 1
                self.update_frame_fast()
        elif event.key == 'left':
            # 이전 프레임
            if self.current_frame > 0:
                self.current_frame -= 1
                self.update_frame_fast()
        elif event.key == 'up':
            # 속도 증가 (interval 감소)
            self.interval = max(10, int(self.interval * 0.8))
            if self.anim:
                self.anim.interval = self.interval
            print(f"Playback speed: {1000/self.interval:.1f} FPS (interval: {self.interval}ms)")
        elif event.key == 'down':
            # 속도 감소 (interval 증가)
            self.interval = min(1000, int(self.interval * 1.25))
            if self.anim:
                self.anim.interval = self.interval
            print(f"Playback speed: {1000/self.interval:.1f} FPS (interval: {self.interval}ms)")
        elif event.key == '1':  # LDM obstacles_type 모드
            self.ldm_mode = 1
            self.update_frame_fast()
            print("LDM Mode: Obstacles Type")
        elif event.key == '2':  # LDM class_id 모드
            self.ldm_mode = 2
            self.update_frame_fast()
            print("LDM Mode: Class ID")
        elif event.key == '3':  # LDM both 모드
            self.ldm_mode = 3
            self.update_frame_fast()
            print("LDM Mode: Both (Priority: Obstacles)")
        elif event.key == 'escape' or event.key == 'q':
            if self.anim:
                self.anim.pause()
            plt.close(self.fig)
            
    def init_graphics(self):
        """그래픽 요소를 초기화합니다."""
        # Bird's eye view 설정
        self.ax1.set_xlim(-self.view_range/2, self.view_range/2)
        self.ax1.set_ylim(-self.view_range/2, self.view_range/2)
        self.ax1.set_aspect('equal')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_xlabel('Y (m) →')
        self.ax1.set_ylabel('X (m) →')
        
        # 자차 그래픽 초기화
        self.artists['ego_rect'] = patches.Rectangle(
            (-0.3, -0.3), 0.6, 0.6,
            angle=0, linewidth=2, edgecolor='black', facecolor='yellow'
        )
        self.ax1.add_patch(self.artists['ego_rect'])
        self.artists['ego_center'], = self.ax1.plot(0, 0, 'ko', markersize=6)
        
        # 자차 화살표
        self.artists['ego_arrow'] = patches.FancyArrow(
            0, 0, 0, 0.5, head_width=0.15, head_length=0.1,
            fc='black', ec='black', linewidth=2
        )
        self.ax1.add_patch(self.artists['ego_arrow'])
        
        # 텍스트 초기화
        self.artists['info_text'] = self.fig.text(0.02, 0.02, '', fontsize=10, 
                                                 bbox=dict(boxstyle="round,pad=0.3", 
                                                          facecolor="yellow", alpha=0.5))
        self.artists['ego_info_text'] = self.ax1.text(0.02, 0.98, '', transform=self.ax1.transAxes, 
                                                      verticalalignment='top', 
                                                      bbox=dict(boxstyle="round,pad=0.3", 
                                                               facecolor="white", alpha=0.8))
        
        # Camera view 초기화
        self.artists['camera_image'] = self.ax2.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
        self.ax2.axis('off')
        self.ax2.set_title('Camera View')
        
    def update_frame_fast(self):
        """최적화된 프레임 업데이트"""
        # 캐시된 데이터 가져오기
        cache_data = self.frame_cache[self.current_frame]
        ego_info = cache_data['ego_info']
        preprocessed = cache_data['preprocessed']
        
        # LDM 포인트 업데이트 (scatter plot 사용)
        ldm_x = []
        ldm_y = []
        ldm_colors = []
        
        for ldm_pt in preprocessed['ldm_points']:
            color = None
            if self.ldm_mode == 1:
                color = get_ldm_color(ldm_pt['obstacles_type'])
            elif self.ldm_mode == 2:
                color = get_ldm_segmentation_color(ldm_pt['class_id'])
            elif self.ldm_mode == 3:
                if ldm_pt['obstacles_type'] >= 1:
                    color = get_ldm_color(ldm_pt['obstacles_type'])
                else:
                    color = get_ldm_segmentation_color(ldm_pt['class_id'])
            
            if color and (self.ldm_mode != 1 or ldm_pt['obstacles_type'] >= 1):
                ldm_x.append(ldm_pt['x'])
                ldm_y.append(ldm_pt['y'])
                ldm_colors.append(color)
        
        # LDM scatter plot 업데이트
        if self.artists['ldm_scatter']:
            self.artists['ldm_scatter'].remove()
        if ldm_x:
            self.artists['ldm_scatter'] = self.ax1.scatter(ldm_x, ldm_y, c=ldm_colors, s=4, alpha=0.8)
        
        # GT trajectory 업데이트
        for circle in self.artists['trajectory_circles']:
            circle.remove()
        self.artists['trajectory_circles'] = []
        
        trajectory_points = preprocessed['trajectory_points']
        for i, (x, y) in enumerate(trajectory_points):
            color_ratio = i / max(len(trajectory_points) - 1, 1)
            green_color = (0.3 * (1 - color_ratio) + 0.0 * color_ratio,
                          0.8 * (1 - color_ratio) + 0.5 * color_ratio,
                          0.3 * (1 - color_ratio) + 0.0 * color_ratio)
            circle = plt.Circle((x, y), 0.1, color=green_color, alpha=0.8)
            self.ax1.add_patch(circle)
            self.artists['trajectory_circles'].append(circle)
        
        # 객체 업데이트
        # 기존 객체 그래픽 제거
        for artist in self.artists['objects']:
            artist.remove()
        self.artists['objects'] = []
        
        for label in self.artists['object_labels']:
            label.remove()
        self.artists['object_labels'] = []
        
        if self.artists['object_points']:
            self.artists['object_points'].remove()
            self.artists['object_points'] = None
        
        # 모든 객체 포인트를 한번에 그리기
        all_points_x = []
        all_points_y = []
        all_colors = []
        
        for obj in preprocessed['objects']:
            color = get_object_color(obj['class_id'])
            
            # 포인트 수집
            for pt_x, pt_y in obj['global_points']:
                all_points_x.append(pt_x)
                all_points_y.append(pt_y)
                all_colors.append(color)
            
            # 객체 중심점
            if obj['global_points']:
                circle = self.ax1.plot(obj['screen_x'], obj['screen_y'], 'o', 
                                     color=color, markersize=8)[0]
            else:
                circle = self.ax1.plot(obj['screen_x'], obj['screen_y'], 'o', 
                                     color=color, markersize=8, fillstyle='none', 
                                     markeredgewidth=2)[0]
            self.artists['objects'].append(circle)
            
            # 속도 벡터
            if obj['velocity'] > 0.1:
                # 이전 프레임과의 위치 차이로 실제 이동 방향 계산
                actual_heading = None
                if self.prev_objects and obj['id'] in self.prev_objects:
                    prev_obj = self.prev_objects[obj['id']]
                    prev_pos = prev_obj.get('_global_center', {})
                    dx = obj['global_x'] - prev_pos.get('_x', obj['global_x'])
                    dy = obj['global_y'] - prev_pos.get('_y', obj['global_y'])
                    if abs(dx) > 0.01 or abs(dy) > 0.01:
                        actual_heading = math.atan2(dy, dx)
                
                if actual_heading is not None:
                    visualize_scale_factor = 3
                    global_vel_x = obj['velocity'] * np.cos(actual_heading) * visualize_scale_factor
                    global_vel_y = obj['velocity'] * np.sin(actual_heading) * visualize_scale_factor
                else:
                    yaw = obj['global_yaw']
                    visualize_scale_factor = 3
                    global_vel_x = obj['velocity'] * np.cos(yaw) * visualize_scale_factor
                    global_vel_y = obj['velocity'] * np.sin(yaw) * visualize_scale_factor
                
                # 자차 좌표계로 변환
                if ego_info:
                    cos_yaw = np.cos(-ego_info['yaw'])
                    sin_yaw = np.sin(-ego_info['yaw'])
                    ego_vel_x = global_vel_x * cos_yaw - global_vel_y * sin_yaw
                    ego_vel_y = global_vel_x * sin_yaw + global_vel_y * cos_yaw
                else:
                    ego_vel_x = global_vel_x
                    ego_vel_y = global_vel_y
                
                dx = -ego_vel_y
                dy = ego_vel_x
                
                arrow = patches.FancyArrowPatch(
                    (obj['screen_x'], obj['screen_y']),
                    (obj['screen_x'] + dx, obj['screen_y'] + dy),
                    arrowstyle='->', mutation_scale=20,
                    color=color, alpha=0.7, linewidth=2
                )
                self.ax1.add_patch(arrow)
                self.artists['objects'].append(arrow)
            
            # 객체 정보 텍스트
            info = f"ID:{obj['id']}"
            if obj['velocity'] > 0.1:
                info += f"\nv:{obj['velocity']:.1f}m/s"
                info += f"\nyaw:{np.degrees(obj['global_yaw']):.1f}°"
            text = self.ax1.text(obj['screen_x'] + 1, obj['screen_y'] + 1, info, 
                               fontsize=8, color=color,
                               bbox=dict(boxstyle="round,pad=0.2", 
                                       facecolor="white", alpha=0.7))
            self.artists['object_labels'].append(text)
        
        # 모든 객체 포인트를 한번에 그리기
        if all_points_x:
            self.artists['object_points'] = self.ax1.scatter(
                all_points_x, all_points_y, c=all_colors, s=9, alpha=0.8
            )
        
        # 현재 프레임의 객체 정보를 다음 프레임을 위해 저장
        self.prev_objects = {}
        for obj in cache_data['raw_objects']:
            obj_id = obj.get('_id', '')
            if obj_id:
                self.prev_objects[obj_id] = obj
        
        # 방위벡터 업데이트
        if self.artists['near_arrow']:
            self.artists['near_arrow'].remove()
            self.artists['near_arrow'] = None
        if self.artists['dest_arrow']:
            self.artists['dest_arrow'].remove()
            self.artists['dest_arrow'] = None
        
        near_direction = cache_data['near_direction']
        destination_direction = cache_data['destination_direction']
        
        if near_direction and destination_direction and ego_info:
            # 근접 방위벡터
            if near_direction.get('x') is not None and near_direction.get('y') is not None:
                dx_near = near_direction['x']
                dy_near = near_direction['y']
                
                cos_yaw = np.cos(-ego_info['yaw'])
                sin_yaw = np.sin(-ego_info['yaw'])
                ego_dx_near = dx_near * cos_yaw - dy_near * sin_yaw
                ego_dy_near = dx_near * sin_yaw + dy_near * cos_yaw
                
                screen_x_near = -ego_dy_near * 5
                screen_y_near = ego_dx_near * 5
                
                self.artists['near_arrow'] = patches.FancyArrowPatch(
                    (0, 0), (screen_x_near, screen_y_near),
                    arrowstyle='->', mutation_scale=30,
                    color='darkgreen', alpha=0.8, linewidth=1
                )
                self.ax1.add_patch(self.artists['near_arrow'])
            
            # 목적지 방위벡터
            if destination_direction.get('x') is not None and destination_direction.get('y') is not None:
                dx_dest = destination_direction['x']
                dy_dest = destination_direction['y']
                
                cos_yaw = np.cos(-ego_info['yaw'])
                sin_yaw = np.sin(-ego_info['yaw'])
                ego_dx_dest = dx_dest * cos_yaw - dy_dest * sin_yaw
                ego_dy_dest = dx_dest * sin_yaw + dy_dest * cos_yaw
                
                screen_x_dest = -ego_dy_dest * 8
                screen_y_dest = ego_dx_dest * 8
                
                self.artists['dest_arrow'] = patches.FancyArrowPatch(
                    (0, 0), (screen_x_dest, screen_y_dest),
                    arrowstyle='->', mutation_scale=25,
                    color='lightgreen', alpha=0.6, linewidth=1
                )
                self.ax1.add_patch(self.artists['dest_arrow'])
        
        # 자차 정보 텍스트 업데이트
        if ego_info:
            info_text = f"Ego Position: ({ego_info['x']:.1f}, {ego_info['y']:.1f}), Yaw: {np.degrees(ego_info['yaw']):.1f}°"
            if 'velocity' in ego_info and ego_info['velocity'] > 0:
                info_text += f", Velocity: {ego_info['velocity']:.1f} m/s"
            self.artists['ego_info_text'].set_text(info_text)
        
        # Camera view 업데이트
        if self.current_frame in self.image_cache:
            image_path = self.image_cache[self.current_frame]
            img = cv2.imread(str(image_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.artists['camera_image'].set_data(img)
        else:
            # 빈 이미지
            self.artists['camera_image'].set_data(np.zeros((480, 640, 3), dtype=np.uint8))
        
        # 제목 업데이트
        status = "[Playing]" if self.playing else "[Paused]"
        fps = 1000 / self.interval
        self.fig.suptitle(f'Frame {self.current_frame} / {len(self.data)-1} {status} [{fps:.1f} FPS]\n'
                          f'(Space: 재생/일시정지, ←/→: 프레임, ↑/↓: 속도, ESC/Q: 종료)', fontsize=14)
        
        # 정보 텍스트 업데이트
        info_text = f"Frame: {self.current_frame} / {len(self.data)-1}\nObjects: {len(preprocessed['objects'])}"
        if ego_info:
            info_text += f"\nEgo: ({ego_info['x']:.1f}, {ego_info['y']:.1f})"
        if self.current_frame in self.image_cache:
            info_text += f"\nImage: {self.image_cache[self.current_frame].name}"
        else:
            info_text += f"\nImage: Not found"
        self.artists['info_text'].set_text(info_text)
        
        # LDM 모드에 따른 타이틀 업데이트
        ldm_mode_text = {1: "Obstacles Type", 2: "Class ID", 3: "Both (Priority: Obstacles)"}
        self.ax1.set_title(f'Bird\'s Eye View (Ego-centric, Forward: Up) - LDM Mode: {ldm_mode_text.get(self.ldm_mode, "Unknown")}')
        
    def animate_update(self, frame_idx):
        """애니메이션 업데이트 함수"""
        if self.playing:
            self.current_frame = (self.current_frame + 1) % len(self.data)
            self.update_frame_fast()
        return []
    
    def show(self, start_frame: int = 0):
        """시각화 시작 (애니메이션 모드)"""
        self.current_frame = start_frame
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 키보드 이벤트 연결
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # tight_layout을 먼저 호출
        plt.tight_layout()
        
        # 그래픽 초기화
        self.init_graphics()
        
        # 첫 프레임 표시
        self.update_frame_fast()
        
        # 애니메이션 생성
        self.anim = FuncAnimation(self.fig, self.animate_update, 
                                frames=len(self.data),
                                interval=self.interval, 
                                repeat=True,
                                blit=False)
        
        # 일시정지 상태로 시작
        self.anim.pause()
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='GT 데이터 시각화 - 성능 최적화 버전')
    parser.add_argument('jsonl_file', type=str, nargs='?', default=None,
                       help='시각화할 JSONL 파일 경로 (미지정시 results 디렉토리의 모든 파일 재생)')
    parser.add_argument('--frame', type=int, default=None,
                       help='특정 프레임만 시각화 (미지정시 애니메이션)')
    parser.add_argument('--img-dir', type=str, default=None,
                       help='이미지 디렉토리 경로 (기본값: results/img)')
    parser.add_argument('--view-range', type=float, default=30.0,
                       help='Bird view 표시 범위 (m)')
    parser.add_argument('--interval', type=int, default=100,
                       help='애니메이션 프레임 간격 (ms)')
    parser.add_argument('--no-trajectory', action='store_true',
                       help='객체 궤적 표시 안함')
    parser.add_argument('--debug', action='store_true',
                       help='디버그 정보 출력')
    
    args = parser.parse_args()
    
    # 파일 경로 처리
    results_dir = Path(__file__).parent / 'results'
    
    if args.jsonl_file:
        # 특정 파일이 지정된 경우
        jsonl_path = Path(args.jsonl_file)
        if not jsonl_path.exists():
            # results 디렉토리에서 찾기
            results_path = results_dir / args.jsonl_file
            if results_path.exists():
                jsonl_path = results_path
            else:
                print(f"File not found: {args.jsonl_file}")
                return
        jsonl_files = [jsonl_path]
    else:
        # 파일이 지정되지 않은 경우: results 디렉토리의 모든 JSONL 파일
        jsonl_files = sorted(results_dir.glob('IO_GT_matching_*.jsonl'))
        if not jsonl_files:
            print(f"No JSONL files found in results directory: {results_dir}")
            return
    
    # 이미지 디렉토리 설정
    img_dir = None
    if args.img_dir:
        img_dir = Path(args.img_dir)
    else:
        img_dir = results_dir / 'img'
    
    if not img_dir.exists():
        print(f"경고: 이미지 디렉토리가 없습니다: {img_dir}")
        img_dir = None
    
    # 단일 파일 처리 (여러 파일 재생은 간단히 하기 위해 첫 번째 파일만)
    jsonl_path = jsonl_files[0]
    
    # 데이터 로드
    print(f"Loading data: {jsonl_path}")
    data = load_jsonl_data(jsonl_path)
    print(f"Total {len(data)} frames loaded")
    
    # db3 이름 추출
    db3_name = None
    if 'IO_GT_matching_' in jsonl_path.name:
        db3_name = jsonl_path.name.replace('IO_GT_matching_', '').replace('.jsonl', '')
    
    # 시각화
    print("\n=== 성능 최적화 버전 실행 중 ===")
    print("Controls: Space(Play/Pause), ←/→(Frame), ↑/↓(Speed), 1/2/3(LDM Mode), ESC/Q(Quit)")
    
    visualizer = FastInteractiveVisualizer(data, img_dir, args.view_range, db3_name, args.debug, args.interval)
    visualizer.show(start_frame=args.frame if args.frame is not None else 0)


if __name__ == '__main__':
    main()