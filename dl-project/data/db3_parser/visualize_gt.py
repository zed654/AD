#!/usr/bin/env python3
"""
GT(Ground Truth) 데이터 시각화 스크립트
db3_extractor.py로 추출한 JSONL 파일의 GT 데이터를 시각화합니다.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
import argparse
from pathlib import Path
import cv2
from typing import List, Dict, Tuple, Optional
import math


def load_jsonl_data(filepath: Path) -> List[Dict]:
    """JSONL 파일에서 데이터를 로드합니다."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def extract_objects_info(frame_data: Dict) -> Tuple[List[Dict], Dict, int, List[Dict], Optional[Dict], Optional[Dict]]:
    """프레임 데이터에서 객체 정보를 추출합니다."""
    objects = frame_data.get('input_environment', {}).get('_objects', [])
    # gt_trajectory로 변경 (output_GT_trajectory가 아님)
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
        
        # 속도 정보 추출 (_linear_vel 필드가 있으면 사용)
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
    
    return objects, ego_info, timestamp, ego_trajectory, near_direction, destination_direction


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


def plot_bird_view(ax, objects: List[Dict], ego_info: Dict, 
                   view_range: float = 30.0, show_trajectory: bool = True,
                   gt_trajectory: List[Dict] = None, near_direction: Optional[Dict] = None,
                   destination_direction: Optional[Dict] = None, prev_objects: Dict = None):
    """Bird's eye view로 장면을 시각화합니다."""
    ax.clear()
    
    # 자차 중심으로 좌표 설정 (전진 방향이 위쪽)
    ax.set_xlim(-view_range/2, view_range/2)
    ax.set_ylim(-view_range/2, view_range/2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Y (m) →')  # Y축이 오른쪽
    ax.set_ylabel('X (m) →')  # X축이 위쪽 (전진 방향)
    ax.set_title('Bird\'s Eye View (Ego-centric, Forward: Up)')
    
    # GT trajectory 그리기 (자차 보다 먼저 그려서 배경에 위치)
    if gt_trajectory and show_trajectory and ego_info:
        trajectory_points = []
        for gt in gt_trajectory:
            pose = gt.get('_pose', {}).get('_pose', {})
            position = pose.get('_position', {})
            gt_x = position.get('_x', 0)
            gt_y = position.get('_y', 0)
            
            # 자차 중심 좌표계로 변환
            dx = gt_x - ego_info['x']
            dy = gt_y - ego_info['y']
            cos_yaw = np.cos(-ego_info['yaw'])
            sin_yaw = np.sin(-ego_info['yaw'])
            # 좌표 변환 (x: 전진, y: 우측)
            ego_x = dx * cos_yaw - dy * sin_yaw
            ego_y = dx * sin_yaw + dy * cos_yaw
            
            # 화면 좌표계로 변환 (x가 위쪽, y가 오른쪽)
            screen_x = -ego_y  # 우측이 화면의 오른쪽
            screen_y = ego_x   # 전진이 화면의 위쪽
            
            trajectory_points.append((screen_x, screen_y))
        
        # 초록색 그라데이션으로 trajectory 그리기
        if len(trajectory_points) > 0:
            for i, (x, y) in enumerate(trajectory_points):
                # 연한 초록(0.3, 0.8, 0.3)에서 진한 초록(0.0, 0.5, 0.0)으로
                color_ratio = i / max(len(trajectory_points) - 1, 1)
                green_color = (0.3 * (1 - color_ratio) + 0.0 * color_ratio,
                              0.8 * (1 - color_ratio) + 0.5 * color_ratio,
                              0.3 * (1 - color_ratio) + 0.0 * color_ratio)
                
                # 원으로 그리기 (크기 절반으로 축소)
                circle = plt.Circle((x, y), 0.1, color=green_color, alpha=0.8)
                ax.add_patch(circle)
                
                # 마지막 포인트에 레이블 추가
                if len(trajectory_points) > 0 and i == len(trajectory_points) - 1:
                    ax.text(x + 0.5, y + 0.5, 'Future Pose', fontsize=6, color='darkgreen',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
                # if i == 0:
                #     ax.text(x + 1, y + 1, 'Start', fontsize=6, color='darkgreen',
                #            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
                # elif i == len(trajectory_points) - 1:
                #     ax.text(x + 0.5, y + 0.5, 'Future Pose', fontsize=6, color='darkgreen',
                #            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    
    # 자차는 항상 원점에 표시 (60cm x 60cm)
    ego_rect = patches.Rectangle(
        (-0.3, -0.3), 0.6, 0.6,  # 60cm x 60cm
        angle=0,  # 자차 중심 좌표계이므로 회전 없음
        linewidth=2, edgecolor='black', facecolor='yellow'
    )
    ax.add_patch(ego_rect)
    ax.plot(0, 0, 'ko', markersize=6)
    
    # 전진 방향 표시 (화살표)
    ax.arrow(0, 0, 0, 0.5, head_width=0.15, head_length=0.1, 
             fc='black', ec='black', linewidth=2)
    
    # 방위벡터 시각화 (자차 중심에서 표시)
    if near_direction and destination_direction:
        # 근접 방위벡터 (진한 초록색, 길이 5m)
        if near_direction.get('x') is not None and near_direction.get('y') is not None:
            # 글로벌 좌표계의 방위벡터를 자차 좌표계로 변환
            dx_near = near_direction['x']
            dy_near = near_direction['y']
            
            # 자차 yaw를 고려한 좌표 변환
            if ego_info:
                cos_yaw = np.cos(-ego_info['yaw'])
                sin_yaw = np.sin(-ego_info['yaw'])
                ego_dx_near = dx_near * cos_yaw - dy_near * sin_yaw
                ego_dy_near = dx_near * sin_yaw + dy_near * cos_yaw
            else:
                ego_dx_near = dx_near
                ego_dy_near = dy_near
            
            # 화면 좌표계로 변환 (x가 위쪽, y가 오른쪽)
            screen_x_near = -ego_dy_near * 5  # 5m 길이로 스케일
            screen_y_near = ego_dx_near * 5
            
            # 진한 초록색 화살표
            ax.arrow(0, 0, screen_x_near, screen_y_near, 
                    head_width=0.3, head_length=0.2, 
                    fc='darkgreen', ec='darkgreen', 
                    linewidth=1, alpha=0.8)
            
            # 라벨 추가 (화살표 끝 위쪽에 표시)
            ax.text(screen_x_near, screen_y_near + 1.0, 
                   'Near\n(20m)', fontsize=9, color='darkgreen',
                   ha='center', va='bottom',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
        
        # 목적지 방위벡터 (연한 초록색, 길이 8m)
        if destination_direction.get('x') is not None and destination_direction.get('y') is not None:
            # 글로벌 좌표계의 방위벡터를 자차 좌표계로 변환
            dx_dest = destination_direction['x']
            dy_dest = destination_direction['y']
            
            # 자차 yaw를 고려한 좌표 변환
            if ego_info:
                cos_yaw = np.cos(-ego_info['yaw'])
                sin_yaw = np.sin(-ego_info['yaw'])
                ego_dx_dest = dx_dest * cos_yaw - dy_dest * sin_yaw
                ego_dy_dest = dx_dest * sin_yaw + dy_dest * cos_yaw
            else:
                ego_dx_dest = dx_dest
                ego_dy_dest = dy_dest
            
            # 화면 좌표계로 변환 (x가 위쪽, y가 오른쪽)
            screen_x_dest = -ego_dy_dest * 8  # 8m 길이로 스케일
            screen_y_dest = ego_dx_dest * 8
            
            # 연한 초록색 화살표
            ax.arrow(0, 0, screen_x_dest, screen_y_dest, 
                    head_width=0.25, head_length=0.15, 
                    fc='lightgreen', ec='lightgreen', 
                    linewidth=1, alpha=0.6)
            
            # 라벨 추가 (화살표 끝 위쪽에 표시)
            ax.text(screen_x_dest, screen_y_dest + 1.0, 
                   'Final Dest', fontsize=9, color='green',
                   ha='center', va='bottom',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    
    # 자차의 실제 위치 정보 표시
    if ego_info:
        info_text = f"Ego Position: ({ego_info['x']:.1f}, {ego_info['y']:.1f}), Yaw: {np.degrees(ego_info['yaw']):.1f}°"
        if 'velocity' in ego_info and ego_info['velocity'] > 0:
            info_text += f", Velocity: {ego_info['velocity']:.1f} m/s"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 객체들 그리기 (자차 중심 좌표계로 변환)
    for obj in objects:
        # 글로벌 센터 사용
        global_center = obj.get('_global_center', {})
        
        # 글로벌 좌표 추출 (속도 벡터 계산에도 필요)
        global_x = global_center.get('_x', 0)
        global_y = global_center.get('_y', 0)
        
        # 글로벌 좌표를 자차 중심으로 변환
        if ego_info:
            # 자차 위치를 기준으로 상대 좌표 계산
            dx = global_x - ego_info['x']
            dy = global_y - ego_info['y']
            # 자차 yaw를 고려한 좌표 변환
            cos_yaw = np.cos(-ego_info['yaw'])
            sin_yaw = np.sin(-ego_info['yaw'])
            ego_x = dx * cos_yaw - dy * sin_yaw
            ego_y = dx * sin_yaw + dy * cos_yaw
            # 화면 좌표계로 변환
            x = -ego_y  # 우측이 화면의 오른쪽
            y = ego_x   # 전진이 화면의 위쪽
        else:
            x, y = global_x, global_y
        
        # view_range 내의 객체만 표시
        if abs(x) > view_range/2 or abs(y) > view_range/2:
            continue
            
        class_id = obj.get('_class_id', -1)
        color = get_object_color(class_id)
        
        # 글로벌 포인트들 그리기 (작은 원으로)
        global_points = obj.get('_global_points', [])
        for point in global_points:
            pt_x = point.get('_x', 0)
            pt_y = point.get('_y', 0)
            
            if ego_info:
                # 자차 중심 좌표계로 변환
                pt_dx = pt_x - ego_info['x']
                pt_dy = pt_y - ego_info['y']
                pt_ego_x = pt_dx * cos_yaw - pt_dy * sin_yaw
                pt_ego_y = pt_dx * sin_yaw + pt_dy * cos_yaw
                # 화면 좌표계로 변환
                pt_screen_x = -pt_ego_y
                pt_screen_y = pt_ego_x
            else:
                pt_screen_x, pt_screen_y = pt_x, pt_y
            
            # view_range 내의 포인트만 표시
            if abs(pt_screen_x) <= view_range/2 and abs(pt_screen_y) <= view_range/2:
                ax.plot(pt_screen_x, pt_screen_y, '.', color=color, markersize=3)
        
        # 객체 중심점 (포인트가 있으면 채워진 원, 없으면 빈 원)
        if global_points:
            ax.plot(x, y, 'o', color=color, markersize=8)  # 채워진 원
        else:
            ax.plot(x, y, 'o', color=color, markersize=8, fillstyle='none', markeredgewidth=2)  # 빈 원
        
        # 객체 정보 표시
        obj_id = obj.get('_id', '')
        velocity = obj.get('_global_velocity', 0)
        global_yaw = obj.get('_global_yaw_heading', 0)
        info = f"ID:{obj_id}"
        if velocity > 0.1:
            info += f"\nv:{velocity:.1f}m/s"
            info += f"\nyaw:{np.degrees(global_yaw):.1f}°"
        ax.text(x + 1, y + 1, info, fontsize=8, color=color,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
        
        # 속도 벡터 표시
        if velocity > 0.1:  # 0.1 m/s 이상일 때만 표시
            # 실제 속도 벡터가 있는지 확인 (옵션 1: 직접 속도 벡터 사용)
            global_velocity_vector = obj.get('_global_velocity_vector', None)
            if global_velocity_vector and isinstance(global_velocity_vector, dict):
                # 실제 속도 벡터가 있으면 사용
                global_vel_x = global_velocity_vector.get('_x', 0) * 3
                global_vel_y = global_velocity_vector.get('_y', 0) * 3
            else:
                # 이전 프레임과의 위치 차이로 실제 이동 방향 계산 (더 정확함)
                actual_heading = None
                if prev_objects and obj_id in prev_objects:
                    prev_obj = prev_objects[obj_id]
                    prev_pos = prev_obj.get('_global_center', {})
                    dx = global_x - prev_pos.get('_x', global_x)
                    dy = global_y - prev_pos.get('_y', global_y)
                    if abs(dx) > 0.01 or abs(dy) > 0.01:  # 최소 이동 거리
                        actual_heading = math.atan2(dy, dx)
                
                if actual_heading is not None:
                    # 실제 이동 방향 사용
                    visualize_scale_factor = 3
                    global_vel_x = velocity * np.cos(actual_heading) * visualize_scale_factor
                    global_vel_y = velocity * np.sin(actual_heading) * visualize_scale_factor
                else:
                    # 이전 위치가 없으면 global_yaw_heading 사용 (차선책)
                    yaw = obj.get('_global_yaw_heading', 0)
                    # 글로벌 좌표계에서 속도 벡터 계산
                    visualize_scale_factor = 3
                    global_vel_x = velocity * np.cos(yaw) * visualize_scale_factor
                    global_vel_y = velocity * np.sin(yaw) * visualize_scale_factor
            
            if ego_info:
                # 자차 좌표계로 변환
                cos_yaw = np.cos(-ego_info['yaw'])
                sin_yaw = np.sin(-ego_info['yaw'])
                ego_vel_x = global_vel_x * cos_yaw - global_vel_y * sin_yaw
                ego_vel_y = global_vel_x * sin_yaw + global_vel_y * cos_yaw
            else:
                ego_vel_x = global_vel_x
                ego_vel_y = global_vel_y
            
            # 화면 좌표계로 변환
            dx = -ego_vel_y  # 우측이 화면의 오른쪽
            dy = ego_vel_x   # 전진이 화면의 위쪽
            ax.arrow(x, y, dx, dy, head_width=0.5, head_length=0.3, 
                    fc=color, ec=color, alpha=0.7, linewidth=2)
    
    # 범례 추가 (포인트 있음/없음 구분)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Object (with points)',
               markerfacecolor='gray', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Object (no points)',
               markerfacecolor='none', markeredgecolor='gray', markeredgewidth=2, markersize=8),
        Line2D([0], [0], marker='.', color='gray', label='Point cloud',
               markersize=3, linestyle='None')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)


def plot_camera_view(ax, image_path: Optional[Path]):
    """카메라 이미지를 표시합니다."""
    ax.clear()
    if image_path and image_path.exists():
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title('Camera View')
    else:
        ax.text(0.5, 0.5, 'No Image Available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Camera View')
        ax.axis('off')


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


class InteractiveVisualizer:
    """키보드 입력을 처리하는 인터렉티브 시각화 클래스"""
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
        self.playing = False  # 애니메이션 일시정지 상태로 시작
        self.prev_objects = {}  # 이전 프레임의 객체 정보 저장
        
    def on_key(self, event):
        """키보드 이벤트 핸들러"""
        if event.key == ' ':  # 스페이스바: 재생/일시정지
            self.playing = not self.playing
            if self.playing and self.anim:
                self.anim.resume()
            elif self.anim:
                self.anim.pause()
        elif event.key == 'right':
            # 다음 프레임
            if self.current_frame < len(self.data) - 1:
                self.current_frame += 1
                self.update_frame()
                if self.anim and not self.playing:
                    # 일시정지 상태에서 프레임 이동 시 애니메이션 프레임도 업데이트
                    self.anim.frame_seq = self.anim.new_frame_seq()
        elif event.key == 'left':
            # 이전 프레임
            if self.current_frame > 0:
                self.current_frame -= 1
                self.update_frame()
                if self.anim and not self.playing:
                    # 일시정지 상태에서 프레임 이동 시 애니메이션 프레임도 업데이트
                    self.anim.frame_seq = self.anim.new_frame_seq()
        elif event.key == 'up':
            # 속도 증가 (interval 감소)
            self.interval = max(10, int(self.interval * 0.8))  # 최소 10ms
            if self.anim:
                self.anim.interval = self.interval
            print(f"재생 속도: {1000/self.interval:.1f} FPS (interval: {self.interval}ms)")
        elif event.key == 'down':
            # 속도 감소 (interval 증가)
            self.interval = min(1000, int(self.interval * 1.25))  # 최대 1000ms
            if self.anim:
                self.anim.interval = self.interval
            print(f"재생 속도: {1000/self.interval:.1f} FPS (interval: {self.interval}ms)")
        elif event.key == 'escape' or event.key == 'q':
            if self.anim:
                self.anim.pause()
            plt.close(self.fig)
            
    def update_frame(self):
        """현재 프레임 업데이트"""
        frame_data = self.data[self.current_frame]
        objects, ego_info, timestamp, gt_trajectory, near_direction, destination_direction = extract_objects_info(frame_data)
        
        # 타임스탬프 기반으로 이미지 찾기
        image_path = None
        if self.img_dir:
            image_path = find_matching_image(timestamp, self.img_dir, self.db3_name)
        
        # Bird's eye view
        plot_bird_view(self.ax1, objects, ego_info, self.view_range, 
                      gt_trajectory=gt_trajectory, 
                      near_direction=near_direction,
                      destination_direction=destination_direction,
                      prev_objects=self.prev_objects)
        
        # 현재 프레임의 객체 정보를 다음 프레임을 위해 저장
        self.prev_objects = {}
        for obj in objects:
            obj_id = obj.get('_id', '')
            if obj_id:
                self.prev_objects[obj_id] = obj
        
        # Camera view
        plot_camera_view(self.ax2, image_path)
        
        # 제목 업데이트
        status = "[Playing]" if self.playing else "[Paused]"
        fps = 1000 / self.interval
        self.fig.suptitle(f'Frame {self.current_frame} / {len(self.data)-1} {status} [{fps:.1f} FPS]\n'
                          f'(Space: 재생/일시정지, ←/→: 프레임, ↑/↓: 속도, ESC/Q: 종료)', fontsize=14)
        
        # 추가 정보 표시
        info_text = f"Frame: {self.current_frame} / {len(self.data)-1}\nObjects: {len(objects)}"
        if ego_info:
            info_text += f"\nEgo: ({ego_info['x']:.1f}, {ego_info['y']:.1f})"
        if image_path:
            info_text += f"\nImage: {image_path.name}"
        else:
            info_text += f"\nImage: Not found"
            if self.debug:
                info_text += f" (timestamp: {timestamp})"
        
        # 기존 텍스트 삭제 후 새로 그리기
        for txt in self.fig.texts:
            if txt.get_position()[1] < 0.1:  # 하단 텍스트만 삭제
                txt.remove()
        self.fig.text(0.02, 0.02, info_text, fontsize=10, 
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
        
        # 화면 업데이트
        self.fig.canvas.draw()
        
    def animate_update(self, frame_idx):
        """애니메이션 업데이트 함수"""
        if self.playing:
            self.current_frame = frame_idx
            self.update_frame()
        return [self.ax1, self.ax2]
    
    def show(self, start_frame: int = 0):
        """시각화 시작 (애니메이션 모드)"""
        self.current_frame = start_frame
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 키보드 이벤트 연결
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # tight_layout을 먼저 호출
        plt.tight_layout()
        
        # 첫 프레임 표시
        self.update_frame()
        
        # 애니메이션 생성
        self.anim = FuncAnimation(self.fig, self.animate_update, 
                                frames=len(self.data),
                                interval=self.interval, 
                                repeat=True,
                                blit=False)
        
        # 일시정지 상태로 시작
        self.anim.pause()
        
        plt.show()


def visualize_single_frame(data: List[Dict], frame_idx: int, 
                          img_dir: Optional[Path] = None,
                          view_range: float = 100.0,
                          db3_name: str = None,
                          debug: bool = False,
                          interval: int = 100):
    """단일 프레임을 시각화합니다. (키보드 이벤트 지원)"""
    if frame_idx >= len(data):
        print(f"프레임 인덱스 {frame_idx}가 범위를 벗어났습니다. (최대: {len(data)-1})")
        return
    
    # 인터렉티브 시각화 사용 (애니메이션 모드)
    visualizer = InteractiveVisualizer(data, img_dir, view_range, db3_name, debug, interval)
    visualizer.playing = False  # 특정 프레임 지정 시 일시정지 상태로 시작
    visualizer.show(start_frame=frame_idx)


def animate_frames(data: List[Dict], img_dir: Optional[Path] = None,
                  view_range: float = 100.0, interval: int = 100,
                  db3_name: str = None):
    """모든 프레임을 애니메이션으로 재생합니다."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 애니메이션 상태 관리
    anim_state = {'playing': True, 'current_frame': 0}
    
    def on_key(event):
        """키보드 이벤트 핸들러 (애니메이션 모드)"""
        if event.key == ' ':  # 스페이스바: 재생/일시정지
            anim_state['playing'] = not anim_state['playing']
            if anim_state['playing']:
                anim.resume()
            else:
                anim.pause()
        elif event.key == 'right' and not anim_state['playing']:
            # 일시정지 상태에서 다음 프레임
            anim_state['current_frame'] = (anim_state['current_frame'] + 1) % len(data)
            update(anim_state['current_frame'])
            fig.canvas.draw()
        elif event.key == 'left' and not anim_state['playing']:
            # 일시정지 상태에서 이전 프레임
            anim_state['current_frame'] = (anim_state['current_frame'] - 1) % len(data)
            update(anim_state['current_frame'])
            fig.canvas.draw()
        elif event.key == 'escape' or event.key == 'q':
            plt.close(fig)
    
    def update(frame_idx):
        anim_state['current_frame'] = frame_idx
        frame_data = data[frame_idx]
        objects, ego_info, timestamp, gt_trajectory, near_direction, destination_direction = extract_objects_info(frame_data)
        
        # Bird's eye view
        plot_bird_view(ax1, objects, ego_info, view_range, 
                      gt_trajectory=gt_trajectory,
                      near_direction=near_direction,
                      destination_direction=destination_direction)
        
        # Camera view
        image_path = None
        if img_dir:
            image_path = find_matching_image(timestamp, img_dir, db3_name)
        plot_camera_view(ax2, image_path)
        
        # 제목 업데이트
        status = "Playing" if anim_state['playing'] else "Paused"
        fig.suptitle(f'Frame {frame_idx} / {len(data)-1} [{status}] '
                    f'(Space: 재생/일시정지, ←/→: 프레임 이동, ESC/Q: 종료)', 
                    fontsize=16)
        
        # 추가 정보 표시
        info_text = f"Objects: {len(objects)}"
        if image_path:
            info_text += f" | Image: Found"
        else:
            info_text += f" | Image: Not found"
        
        # 기존 텍스트 삭제 후 새로 그리기
        for txt in fig.texts:
            if txt.get_position()[1] < 0.1:  # 하단 텍스트만
                txt.remove()
        fig.text(0.5, 0.01, info_text, fontsize=10, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
    
    # 키보드 이벤트 연결
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    anim = FuncAnimation(fig, update, frames=len(data), 
                        interval=interval, repeat=True)
    plt.tight_layout()
    plt.show()
    
    return anim


def main():
    parser = argparse.ArgumentParser(description='GT 데이터 시각화')
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
                print(f"파일을 찾을 수 없습니다: {args.jsonl_file}")
                return
        jsonl_files = [jsonl_path]
    else:
        # 파일이 지정되지 않은 경우: results 디렉토리의 모든 JSONL 파일
        jsonl_files = sorted(results_dir.glob('IO_GT_matching_*.jsonl'))
        if not jsonl_files:
            print(f"results 디렉토리에 JSONL 파일이 없습니다: {results_dir}")
            return
        print(f"총 {len(jsonl_files)}개의 JSONL 파일을 찾았습니다:")
    
    # 여러 파일 순차 재생을 위한 새로운 클래스
    class MultiFileVisualizer:
        """여러 JSONL 파일을 순차적으로 재생하는 시각화 클래스"""
        def __init__(self, jsonl_files: List[Path], img_dir: Optional[Path], 
                     view_range: float, interval: int, debug: bool = False):
            self.jsonl_files = jsonl_files
            self.img_dir = img_dir
            self.view_range = view_range
            self.interval = interval
            self.debug = debug
            self.current_file_idx = 0
            self.all_data = []
            self.file_boundaries = []  # 각 파일의 시작 프레임 인덱스
            self.db3_names = []
            
            # 모든 파일 로드
            total_frames = 0
            for i, jsonl_path in enumerate(jsonl_files):
                print(f"\n[{i+1}/{len(jsonl_files)}] 로드 중: {jsonl_path.name}")
                data = load_jsonl_data(jsonl_path)
                self.all_data.extend(data)
                self.file_boundaries.append(total_frames)
                total_frames += len(data)
                
                # db3 이름 추출
                db3_name = None
                if 'IO_GT_matching_' in jsonl_path.name:
                    db3_name = jsonl_path.name.replace('IO_GT_matching_', '').replace('.jsonl', '')
                self.db3_names.append(db3_name)
                
                print(f"  - {len(data)}개 프레임 로드 완료")
            
            print(f"\n총 {total_frames}개 프레임 로드 완료")
            
        def get_current_db3_name(self, frame_idx: int) -> str:
            """현재 프레임이 속한 파일의 db3 이름 반환"""
            for i in range(len(self.file_boundaries) - 1, -1, -1):
                if frame_idx >= self.file_boundaries[i]:
                    return self.db3_names[i]
            return self.db3_names[0]
        
        def get_current_file_info(self, frame_idx: int) -> Tuple[int, str]:
            """현재 프레임이 속한 파일 정보 반환"""
            for i in range(len(self.file_boundaries) - 1, -1, -1):
                if frame_idx >= self.file_boundaries[i]:
                    return i, self.jsonl_files[i].name
            return 0, self.jsonl_files[0].name
        
        def show(self):
            """시각화 시작"""
            visualizer = InteractiveVisualizer(self.all_data, self.img_dir, 
                                             self.view_range, None, self.debug, self.interval)
            # 파일 정보를 표시하도록 update_frame 메서드 오버라이드
            original_update_frame = visualizer.update_frame
            
            def new_update_frame():
                original_update_frame()
                # 현재 파일 정보 추가
                file_idx, file_name = self.get_current_file_info(visualizer.current_frame)
                db3_name = self.get_current_db3_name(visualizer.current_frame)
                visualizer.db3_name = db3_name  # 이미지 매칭을 위해 업데이트
                
                # 파일 정보 표시
                file_info = f"File [{file_idx+1}/{len(self.jsonl_files)}]: {file_name}"
                visualizer.fig.text(0.5, 0.96, file_info, fontsize=12, ha='center',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            
            visualizer.update_frame = new_update_frame
            visualizer.show()
    
    # 이미지 디렉토리 설정
    img_dir = None
    if args.img_dir:
        img_dir = Path(args.img_dir)
    else:
        img_dir = results_dir / 'img'
    
    if not img_dir.exists():
        print(f"경고: 이미지 디렉토리가 없습니다: {img_dir}")
        img_dir = None
    
    # 여러 파일인 경우
    if len(jsonl_files) > 1:
        for i, f in enumerate(jsonl_files):
            print(f"  [{i+1}] {f.name}")
        
        # 멀티 파일 시각화
        multi_visualizer = MultiFileVisualizer(jsonl_files, img_dir, 
                                             args.view_range, args.interval, args.debug)
        multi_visualizer.show()
    else:
        # 단일 파일 처리 (기존 로직)
        jsonl_path = jsonl_files[0]
        
        # 데이터 로드
        print(f"데이터 로드 중: {jsonl_path}")
        data = load_jsonl_data(jsonl_path)
        print(f"총 {len(data)}개 프레임 로드 완료")
        
        # db3 이름 추출 (타임스탬프 매칭용)
        db3_name = None
        if 'IO_GT_matching_' in jsonl_path.name:
            db3_name = jsonl_path.name.replace('IO_GT_matching_', '').replace('.jsonl', '')
        
        # 디버그 모드일 때 첫 프레임 정보 출력
        if args.debug and len(data) > 0:
            first_frame = data[0]
            print("\n[디버그] 첫 프레임 데이터 구조:")
            print(f"- input_environment keys: {list(first_frame.get('input_environment', {}).keys())}")
            print(f"- gt_trajectory 길이: {len(first_frame.get('gt_trajectory', []))}")
            if first_frame.get('input_environment', {}).get('_header', {}).get('_stamp'):
                stamp = first_frame['input_environment']['_header']['_stamp']
                timestamp = stamp.get('_sec', 0) * 1_000_000_000 + stamp.get('_nanosec', 0)
                print(f"- 첫 프레임 timestamp: {timestamp}")
                
                # 이미지 매칭 테스트
                if img_dir:
                    test_img = find_matching_image(timestamp, img_dir, db3_name)
                    if test_img:
                        print(f"- 매칭된 이미지: {test_img.name}")
                    else:
                        print(f"- 매칭된 이미지 없음")
        
        # 이미지 디렉토리 정보 출력
        if img_dir and args.debug:
            img_files = list(img_dir.glob('*.jpg'))
            print(f"\n이미지 디렉토리: {img_dir}")
            print(f"이미지 파일 개수: {len(img_files)}")
            if img_files:
                print(f"예시 이미지 파일: {img_files[0].name}")
        
        # 시각화
        if args.frame is not None:
            # 단일 프레임 시각화 (일시정지 상태로 시작)
            visualize_single_frame(data, args.frame, img_dir, args.view_range, db3_name, args.debug, args.interval)
        else:
            # 애니메이션 모드 (기본 일시정지 상태로 시작)
            print("시각화 준비 완료")
            print("조작법: Space(재생/일시정지), ←/→(프레임 이동), ↑/↓(속도 조절), ESC/Q(종료)")
            visualizer = InteractiveVisualizer(data, img_dir, args.view_range, db3_name, args.debug, args.interval)
            visualizer.show()


if __name__ == '__main__':
    main()