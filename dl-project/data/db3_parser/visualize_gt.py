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


def load_jsonl_data(filepath: Path) -> List[Dict]:
    """JSONL 파일에서 데이터를 로드합니다."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def extract_objects_info(frame_data: Dict) -> Tuple[List[Dict], Dict]:
    """프레임 데이터에서 객체 정보를 추출합니다."""
    objects = frame_data.get('input_environment', {}).get('_objects', [])
    ego_pos = frame_data.get('output_GT_trajectory', {}).get('_waypoints', [])
    
    # 자차 위치 (첫 번째 웨이포인트)
    ego_info = {}
    if ego_pos:
        ego_info = {
            'x': ego_pos[0].get('_x', 0),
            'y': ego_pos[0].get('_y', 0),
            'yaw': ego_pos[0].get('_yaw', 0)
        }
    
    return objects, ego_info


def get_object_color(class_id: int) -> str:
    """클래스 ID에 따른 색상을 반환합니다."""
    color_map = {
        100: 'red',      # 차량
        202: 'blue',     # 보행자
        203: 'orange',   # 자전거
        204: 'purple',   # 트럭
        205: 'green',    # 버스
        300: 'gray',     # 기타
    }
    return color_map.get(class_id, 'black')


def plot_bird_view(ax, objects: List[Dict], ego_info: Dict, 
                   view_range: float = 100.0, show_trajectory: bool = True):
    """Bird's eye view로 장면을 시각화합니다."""
    ax.clear()
    ax.set_xlim(-view_range/2, view_range/2)
    ax.set_ylim(-view_range/2, view_range/2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Bird\'s Eye View')
    
    # 자차 그리기
    if ego_info:
        ego_x, ego_y = ego_info['x'], ego_info['y']
        ego_rect = patches.Rectangle(
            (ego_x - 2, ego_y - 1), 4, 2,
            angle=np.degrees(ego_info['yaw']),
            linewidth=2, edgecolor='black', facecolor='yellow'
        )
        ax.add_patch(ego_rect)
        ax.plot(ego_x, ego_y, 'ko', markersize=8)
    
    # 객체들 그리기
    for obj in objects:
        center = obj.get('_global_center', {})
        x, y = center.get('_x', 0), center.get('_y', 0)
        
        # view_range 내의 객체만 표시
        if abs(x) > view_range or abs(y) > view_range:
            continue
            
        class_id = obj.get('_class_id', -1)
        color = get_object_color(class_id)
        
        # 객체 중심점
        ax.plot(x, y, 'o', color=color, markersize=6)
        
        # 객체 ID 표시
        obj_id = obj.get('_id', '')
        ax.text(x + 1, y + 1, f"ID:{obj_id}", fontsize=8, color=color)
        
        # 속도 표시
        velocity = obj.get('_global_velocity', 0)
        if velocity > 0.1:  # 0.1 m/s 이상일 때만 표시
            yaw = obj.get('_global_yaw_heading', 0)
            dx = velocity * np.cos(yaw) * 5  # 화살표 길이 스케일
            dy = velocity * np.sin(yaw) * 5
            ax.arrow(x, y, dx, dy, head_width=1, head_length=0.5, 
                    fc=color, ec=color, alpha=0.7)
        
        # 글로벌 포인트들 표시 (있는 경우)
        global_points = obj.get('_global_points', [])
        if global_points and show_trajectory:
            points_x = [p['_x'] for p in global_points]
            points_y = [p['_y'] for p in global_points]
            ax.plot(points_x, points_y, '.', color=color, markersize=2, alpha=0.5)


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


def visualize_single_frame(data: List[Dict], frame_idx: int, 
                          img_dir: Optional[Path] = None,
                          view_range: float = 100.0):
    """단일 프레임을 시각화합니다."""
    if frame_idx >= len(data):
        print(f"프레임 인덱스 {frame_idx}가 범위를 벗어났습니다. (최대: {len(data)-1})")
        return
    
    frame_data = data[frame_idx]
    objects, ego_info = extract_objects_info(frame_data)
    
    # 이미지 경로 찾기
    image_path = None
    if img_dir:
        matched_img = frame_data.get('matched_img_filename', '')
        if matched_img:
            image_path = img_dir / matched_img
    
    # 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Bird's eye view
    plot_bird_view(ax1, objects, ego_info, view_range)
    
    # Camera view
    plot_camera_view(ax2, image_path)
    
    # 전체 정보 표시
    fig.suptitle(f'Frame {frame_idx} / {len(data)-1}', fontsize=16)
    
    # 추가 정보 표시
    info_text = f"Objects: {len(objects)}"
    if ego_info:
        info_text += f"\nEgo: ({ego_info['x']:.1f}, {ego_info['y']:.1f})"
    plt.figtext(0.02, 0.02, info_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
    
    plt.tight_layout()
    plt.show()


def animate_frames(data: List[Dict], img_dir: Optional[Path] = None,
                  view_range: float = 100.0, interval: int = 100):
    """모든 프레임을 애니메이션으로 재생합니다."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    def update(frame_idx):
        frame_data = data[frame_idx]
        objects, ego_info = extract_objects_info(frame_data)
        
        # Bird's eye view
        plot_bird_view(ax1, objects, ego_info, view_range)
        
        # Camera view
        image_path = None
        if img_dir:
            matched_img = frame_data.get('matched_img_filename', '')
            if matched_img:
                image_path = img_dir / matched_img
        plot_camera_view(ax2, image_path)
        
        # 제목 업데이트
        fig.suptitle(f'Frame {frame_idx} / {len(data)-1}', fontsize=16)
    
    anim = FuncAnimation(fig, update, frames=len(data), 
                        interval=interval, repeat=True)
    plt.tight_layout()
    plt.show()
    
    return anim


def main():
    parser = argparse.ArgumentParser(description='GT 데이터 시각화')
    parser.add_argument('jsonl_file', type=str, 
                       help='시각화할 JSONL 파일 경로')
    parser.add_argument('--frame', type=int, default=None,
                       help='특정 프레임만 시각화 (미지정시 애니메이션)')
    parser.add_argument('--img-dir', type=str, default=None,
                       help='이미지 디렉토리 경로 (기본값: results/img)')
    parser.add_argument('--view-range', type=float, default=100.0,
                       help='Bird view 표시 범위 (m)')
    parser.add_argument('--interval', type=int, default=100,
                       help='애니메이션 프레임 간격 (ms)')
    parser.add_argument('--no-trajectory', action='store_true',
                       help='객체 궤적 표시 안함')
    
    args = parser.parse_args()
    
    # 파일 경로 처리
    jsonl_path = Path(args.jsonl_file)
    if not jsonl_path.exists():
        # results 디렉토리에서 찾기
        results_path = Path(__file__).parent / 'results' / args.jsonl_file
        if results_path.exists():
            jsonl_path = results_path
        else:
            print(f"파일을 찾을 수 없습니다: {args.jsonl_file}")
            return
    
    # 이미지 디렉토리 설정
    img_dir = None
    if args.img_dir:
        img_dir = Path(args.img_dir)
    else:
        img_dir = jsonl_path.parent / 'img'
    
    if not img_dir.exists():
        print(f"경고: 이미지 디렉토리가 없습니다: {img_dir}")
        img_dir = None
    
    # 데이터 로드
    print(f"데이터 로드 중: {jsonl_path}")
    data = load_jsonl_data(jsonl_path)
    print(f"총 {len(data)}개 프레임 로드 완료")
    
    # 시각화
    if args.frame is not None:
        # 단일 프레임 시각화
        visualize_single_frame(data, args.frame, img_dir, args.view_range)
    else:
        # 애니메이션
        print("애니메이션 재생 중... (종료: Ctrl+C)")
        animate_frames(data, img_dir, args.view_range, args.interval)


if __name__ == '__main__':
    main()