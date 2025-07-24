#!/usr/bin/env python3
"""
GT(Ground Truth) 데이터 시각화 스크립트 - 공통 파서 사용 버전
db3_extractor.py로 추출한 JSONL 파일의 GT 데이터를 빠르게 시각화합니다.
"""

import sys
import os

# matplotlib 백엔드 설정
import matplotlib
if '--save' in sys.argv:
    matplotlib.use('Agg')
elif 'DISPLAY' not in os.environ:
    # 헤드리스 환경에서는 Agg 백엔드 사용
    matplotlib.use('Agg')
    print("Warning: No display detected, using Agg backend")

# Qt 플랫폼 플러그인 충돌 방지를 위해 환경변수 설정 (OpenCV import 전에 설정)
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''

# MuZero 모듈을 import할 수 있도록 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
import argparse
from pathlib import Path
import cv2
from typing import List, Dict, Tuple, Optional
import time

# 🔥 공통 데이터 파서 사용! (직접 import로 __init__.py 우회)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models', 'MuZero'))
from data_parser import (
    GTDataParser, CoordinateTransformer,
    EgoInfo, ObjectInfo, LDMPoint, TrajectoryPoint,
    get_object_color, get_ldm_segmentation_color
)

# rosetta.py에서 추가 함수 import
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from rosetta import get_ldm_obstacles_type_color


def _get_object_color_old(class_id: int) -> str:  # 임시로 이름 변경 (공용 함수 사용)
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


def _get_ldm_color_old(obstacles_type: int) -> Optional[str]:  # 임시로 이름 변경 (공용 함수 사용)
    """LDM obstacles_type에 따른 색상을 반환합니다."""
    color_map = {
        1: 'lime',       # cliff - 연두색
        2: 'yellow',     # static_obstacles - 노란색
        3: 'purple'      # missing_points - 보라색
    }
    if obstacles_type >= 1:
        return color_map.get(obstacles_type, 'gray')
    return None


def _get_ldm_segmentation_color_old(class_id: int) -> str:  # 임시로 이름 변경 (공용 함수 사용)
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
    """공통 파서를 사용하는 성능 최적화된 인터렉티브 시각화 클래스"""
    def __init__(self, parser: GTDataParser, img_dir: Optional[Path], 
                 view_range: float, db3_name: Optional[str] = None, debug: bool = False,
                 interval: int = 100):
        # ===== 1. 기본 설정 =====
        self.parser = parser  # 🔥 공통 파서 사용! (JSONL 데이터를 파싱하는 객체)
        self.img_dir = img_dir  # 카메라 이미지가 저장된 디렉토리
        self.view_range = view_range  # Bird's eye view에서 보여줄 범위 (미터 단위, 기본 30m)
        self.db3_name = db3_name  # db3 파일 이름 (이미지 매칭용)
        self.debug = debug  # 디버그 모드 플래그
        self.interval = interval  # 애니메이션 프레임 간격 (밀리초, 기본 100ms = 10fps)
        
        # ===== 2. 프레임 관리 =====
        self.current_frame = 0  # 현재 표시 중인 프레임 인덱스
        self.total_frames = parser.get_frame_count()  # 전체 프레임 수
        
        # ===== 3. matplotlib 그래픽 요소 =====
        self.fig = None  # matplotlib Figure 객체
        self.ax1 = None  # 왼쪽 축: Bird's eye view
        self.ax2 = None  # 오른쪽 축: Camera view
        self.anim = None  # FuncAnimation 객체 (재생/일시정지 제어용)
        self.playing = False  # 재생 상태 플래그
        
        # ===== 4. 객체 추적 및 모드 설정 =====
        self.prev_objects = {}  # 이전 프레임의 객체 정보 (속도 벡터 계산용)
        self.ldm_mode = 1  # LDM 표시 모드 (1: 장애물 타입, 2: 세그멘테이션, 3: 둘 다)
        
        # ===== 5. 성능 최적화를 위한 캐시 =====
        self.frame_cache = {}  # 전처리된 프레임 데이터 캐시
        self.image_cache = {}  # 카메라 이미지 경로 캐시
        
        # ===== 6. 그래픽 요소 저장 (재사용을 위해) =====
        self.artists = {
            # 자차(Ego vehicle) 관련
            'ego_rect': None,        # 자차를 표현하는 사각형
            'ego_center': None,      # 자차 중심점
            'ego_arrow': None,       # 자차 방향 화살표
            
            # 주변 객체 관련
            'objects': [],           # 객체들 (차량, 보행자 등)
            'object_points': None,   # 객체들의 포인트 클라우드
            'object_labels': [],     # 객체 ID, 속도 등 텍스트 라벨
            
            # LDM(Local Dynamic Map) 관련
            'ldm_scatter': None,     # LDM 포인트들 (도로 표면 정보)
            
            # 경로 관련
            'trajectory_circles': [],  # GT(Ground Truth) 경로 점들
            'near_arrow': None,      # 근거리 방향 벡터
            'dest_arrow': None,      # 목적지 방향 벡터
            
            # 텍스트 정보
            'info_text': None,       # 프레임 정보 텍스트
            'ego_info_text': None,   # 자차 상태 텍스트
            
            # 카메라 이미지
            'camera_image': None     # 전방 카메라 이미지
        }
        
        # ===== 7. 전처리 실행 =====
        # 모든 프레임을 미리 처리해서 캐시에 저장 (성능 향상)
        self.preprocess_all_frames()
        
    def preprocess_all_frames(self):
        """
        모든 프레임 데이터를 사전 처리하여 캐시합니다.
        
        이 함수는 시각화 시작 전에 모든 프레임을 미리 처리하여:
        1. JSON 파싱 시간을 줄이고
        2. 좌표 변환을 미리 수행하여
        3. 실시간 재생 시 부드러운 성능을 보장합니다.
        """
        print("Preprocessing frames for better performance...")
        start_time = time.time()
        
        # 모든 프레임을 순회하며 전처리
        for i in range(self.total_frames):
            # ===== 1. 원시 JSON 데이터 가져오기 =====
            frame_data = self.parser.get_frame_data(i)
            if not frame_data:
                continue
                
            # ===== 2. 공통 파서로 데이터 추출 =====
            # 🔥 여기서 공통 data_parser.py의 메서드들을 사용!
            ego_info = self.parser.parse_ego_info(frame_data)          # 자차 정보 (위치, 속도, 방향)
            objects = self.parser.parse_objects(frame_data)             # 주변 객체들 (차량, 보행자 등)
            ldm_points = self.parser.parse_ldm_points(frame_data)       # LDM 포인트들 (도로 표면 정보)
            trajectory = self.parser.parse_trajectory(frame_data)        # GT 경로 (자차가 따라갈 경로)
            near_dir, dest_dir = self.parser.get_direction_vectors(frame_data)  # 방향 벡터들
            timestamp = self.parser.get_timestamp(frame_data)           # 타임스탬프
            
            # ===== 3. 프레임 캐시에 저장 =====
            self.frame_cache[i] = {
                # 파싱된 원본 데이터들
                'ego_info': ego_info,
                'objects': objects,
                'ldm_points': ldm_points,
                'trajectory': trajectory,
                'timestamp': timestamp,
                'near_direction': near_dir,
                'destination_direction': dest_dir,
                
                # 화면 좌표로 전처리된 데이터
                # _preprocess_frame_data()는 글로벌 좌표를 화면 좌표로 변환
                'preprocessed': self._preprocess_frame_data(
                    ego_info, objects, ldm_points, trajectory
                )
            }
            
            # ===== 4. 카메라 이미지 경로 캐싱 =====
            # 타임스탬프와 매칭되는 이미지 파일을 찾아서 저장
            if self.img_dir and timestamp:
                image_path = find_matching_image(timestamp, self.img_dir, self.db3_name)
                if image_path:
                    self.image_cache[i] = image_path
        
        elapsed = time.time() - start_time
        print(f"Preprocessing completed in {elapsed:.2f} seconds")
    
    def _preprocess_frame_data(self, ego_info: Optional[EgoInfo], 
                             objects: List[ObjectInfo],
                             ldm_points: List[LDMPoint],
                             trajectory: List[TrajectoryPoint]) -> Dict:
        """
        프레임 데이터를 화면 좌표로 변환합니다.
        
        이 함수의 역할:
        1. 글로벌 좌표계의 데이터를 자차 중심 좌표계로 변환
        2. 자차 중심 좌표를 화면 좌표로 변환 (y축이 전방)
        3. view_range를 벗어난 객체들은 필터링
        
        Args:
            ego_info: 자차 정보 (위치, 방향)
            objects: 주변 객체들 리스트
            ldm_points: LDM 포인트들 리스트
            trajectory: GT 경로 포인트들 리스트
            
        Returns:
            화면 좌표로 변환된 데이터 딕셔너리
        """
        # 결과를 저장할 캐시 딕셔너리 초기화
        cache = {
            'objects': [],          # 화면 좌표로 변환된 객체들
            'ldm_points': [],       # 화면 좌표로 변환된 LDM 포인트들
            'trajectory_points': [] # 화면 좌표로 변환된 경로 포인트들
        }
        
        # 자차 정보가 없으면 빈 캐시 반환
        if not ego_info:
            return cache
        
        # ===== 1. 객체(차량, 보행자 등) 처리 =====
        for obj in objects:
            # 🔥 공통 좌표 변환 사용! (글로벌 → 자차 중심 → 화면)
            # global_to_screen()은 내부적으로:
            # 1) 글로벌 좌표에서 자차 위치를 뺌
            # 2) 자차 방향(yaw)만큼 회전
            # 3) 화면 좌표계로 변환 (x,y 축 교체)
            screen_x, screen_y = CoordinateTransformer.global_to_screen(
                obj.global_x, obj.global_y, ego_info.x, ego_info.y, ego_info.yaw
            )
            
            # view_range 내의 객체만 처리 (화면에 보이는 것만)
            if abs(screen_x) <= self.view_range/2 and abs(screen_y) <= self.view_range/2:
                # 화면에 표시할 객체 정보 구성
                obj_data = {
                    'screen_x': screen_x,      # 화면 x 좌표
                    'screen_y': screen_y,      # 화면 y 좌표
                    'class_id': obj.class_id,  # 객체 종류 (100:보행자, 202:차량 등)
                    'id': obj.id,              # 객체 고유 ID
                    'velocity': obj.velocity,  # 속도
                    'global_yaw': obj.global_yaw,  # 글로벌 방향
                    'global_x': obj.global_x,  # 글로벌 x (속도 벡터 계산용)
                    'global_y': obj.global_y,  # 글로벌 y (속도 벡터 계산용)
                    'global_points': []        # 객체의 포인트 클라우드
                }
                
                # 객체의 포인트 클라우드도 화면 좌표로 변환
                for pt_x, pt_y in obj.global_points:
                    pt_screen_x, pt_screen_y = CoordinateTransformer.global_to_screen(
                        pt_x, pt_y, ego_info.x, ego_info.y, ego_info.yaw
                    )
                    # 화면 범위 내의 포인트만 저장
                    if abs(pt_screen_x) <= self.view_range/2 and abs(pt_screen_y) <= self.view_range/2:
                        obj_data['global_points'].append((pt_screen_x, pt_screen_y))
                
                cache['objects'].append(obj_data)
        
        # ===== 2. LDM(Local Dynamic Map) 포인트 처리 =====
        # LDM은 도로 표면 정보를 담고 있음 (장애물, 차선, 도로 유형 등)
        for ldm in ldm_points:
            screen_x, screen_y = CoordinateTransformer.global_to_screen(
                ldm.x, ldm.y, ego_info.x, ego_info.y, ego_info.yaw
            )
            
            # 화면 범위 내의 LDM 포인트만 저장
            if abs(screen_x) < self.view_range/2 and abs(screen_y) < self.view_range/2:
                cache['ldm_points'].append({
                    'x': screen_x,
                    'y': screen_y,
                    'obstacles_type': ldm.obstacles_type,  # 장애물 타입 (1:절벽, 2:정적장애물, 3:미싱포인트)
                    'class_id': ldm.class_id              # 세그멘테이션 클래스 (도로, 인도, 차선 등)
                })
        
        # ===== 3. GT(Ground Truth) 경로 처리 =====
        # 자차가 따라가야 할 정답 경로
        for traj_pt in trajectory:
            screen_x, screen_y = CoordinateTransformer.global_to_screen(
                traj_pt.x, traj_pt.y, ego_info.x, ego_info.y, ego_info.yaw
            )
            # 경로는 모든 점을 저장 (화면 밖의 점도 방향 표시를 위해)
            cache['trajectory_points'].append((screen_x, screen_y))
        
        return cache
        
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
            if self.current_frame < self.total_frames - 1:
                self.current_frame += 1
                self.update_frame_fast()
                # 일시정지 상태에서도 화면 갱신
                if not self.playing:
                    plt.draw()
        elif event.key == 'left':
            # 이전 프레임
            if self.current_frame > 0:
                self.current_frame -= 1
                self.update_frame_fast()
                # 일시정지 상태에서도 화면 갱신
                if not self.playing:
                    plt.draw()
        elif event.key == 'up':
            # 속도 증가 (interval 감소)
            self.interval = max(10, int(self.interval * 0.8))
            if self.anim:
                # 애니메이션을 재시작해야 interval이 적용됨
                was_playing = self.playing
                if was_playing:
                    self.anim.pause()
                self.anim._interval = self.interval
                self.anim.event_source.interval = self.interval
                if was_playing:
                    self.anim.resume()
            print(f"⏩ Playback speed: {1000/self.interval:.1f} FPS (interval: {self.interval}ms)")
        elif event.key == 'down':
            # 속도 감소 (interval 증가)
            self.interval = min(1000, int(self.interval * 1.25))
            if self.anim:
                # 애니메이션을 재시작해야 interval이 적용됨
                was_playing = self.playing
                if was_playing:
                    self.anim.pause()
                self.anim._interval = self.interval
                self.anim.event_source.interval = self.interval
                if was_playing:
                    self.anim.resume()
            print(f"⏪ Playback speed: {1000/self.interval:.1f} FPS (interval: {self.interval}ms)")
        elif event.key == '1':  # LDM obstacles_type 모드
            self.ldm_mode = 1
            self.update_frame_fast()
            if not self.playing:
                plt.draw()
            print("LDM Mode: Obstacles Type")
        elif event.key == '2':  # LDM class_id 모드
            self.ldm_mode = 2
            self.update_frame_fast()
            if not self.playing:
                plt.draw()
            print("LDM Mode: Class ID")
        elif event.key == '3':  # LDM both 모드
            self.ldm_mode = 3
            self.update_frame_fast()
            if not self.playing:
                plt.draw()
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
        """
        최적화된 프레임 업데이트
        
        이 함수는 매 프레임마다 호출되어 화면을 업데이트합니다.
        캐시된 데이터를 사용하여 빠르게 그래픽을 갱신합니다.
        """
        # ===== 1. 캐시된 데이터 가져오기 =====
        cache_data = self.frame_cache.get(self.current_frame, {})
        if not cache_data:
            return
            
        ego_info = cache_data.get('ego_info')           # 자차 정보
        preprocessed = cache_data.get('preprocessed', {})  # 전처리된 화면 좌표 데이터
        
        # ===== 2. LDM 포인트 업데이트 =====
        # scatter plot을 사용하여 많은 점을 효율적으로 그림
        ldm_x = []      # x 좌표 리스트
        ldm_y = []      # y 좌표 리스트
        ldm_colors = [] # 색상 리스트
        
        # LDM 모드에 따라 다른 색상 적용
        for ldm_pt in preprocessed.get('ldm_points', []):
            color = None
            
            # 모드 1: 장애물 타입만 표시
            if self.ldm_mode == 1:
                color = get_ldm_obstacles_type_color(ldm_pt['obstacles_type'])
                
            # 모드 2: 세그멘테이션 클래스만 표시
            elif self.ldm_mode == 2:
                color = get_ldm_segmentation_color(ldm_pt['class_id'])
                
            # 모드 3: 둘 다 표시 (장애물이 있으면 장애물 색상 우선)
            elif self.ldm_mode == 3:
                if ldm_pt['obstacles_type'] >= 1:
                    color = get_ldm_obstacles_type_color(ldm_pt['obstacles_type'])
                else:
                    color = get_ldm_segmentation_color(ldm_pt['class_id'])
            
            # 색상이 있고, 표시 조건을 만족하면 리스트에 추가
            if color and (self.ldm_mode != 1 or ldm_pt['obstacles_type'] >= 1):
                ldm_x.append(ldm_pt['x'])
                ldm_y.append(ldm_pt['y'])
                ldm_colors.append(color)
        
        # 기존 LDM scatter plot 제거하고 새로 그리기
        if self.artists['ldm_scatter']:
            self.artists['ldm_scatter'].remove()
        if ldm_x:
            self.artists['ldm_scatter'] = self.ax1.scatter(
                ldm_x, ldm_y, c=ldm_colors, s=4, alpha=0.8
            )
        
        # ===== 3. GT 경로 업데이트 =====
        # 기존 경로 원들을 제거
        for circle in self.artists['trajectory_circles']:
            circle.remove()
        self.artists['trajectory_circles'] = []
        
        # 경로를 따라 원을 그림 (가까운 곳은 진한 녹색, 먼 곳은 연한 녹색)
        trajectory_points = preprocessed.get('trajectory_points', [])
        for i, (x, y) in enumerate(trajectory_points):
            # 색상 그라데이션 계산 (0: 진한 녹색, 1: 연한 녹색)
            color_ratio = i / max(len(trajectory_points) - 1, 1)
            green_color = (
                0.3 * (1 - color_ratio) + 0.0 * color_ratio,  # R
                0.8 * (1 - color_ratio) + 0.5 * color_ratio,  # G
                0.3 * (1 - color_ratio) + 0.0 * color_ratio   # B
            )
            circle = plt.Circle((x, y), 0.1, color=green_color, alpha=0.8)
            self.ax1.add_patch(circle)
            self.artists['trajectory_circles'].append(circle)
        
        # ===== 4. 다른 요소들 업데이트 =====
        # 객체 업데이트 (차량, 보행자 등)
        self._update_objects(preprocessed, ego_info, cache_data.get('objects', []))
        
        # 방위벡터 업데이트 (근거리/목적지 방향)
        self._update_direction_vectors(cache_data, ego_info)
        
        # 자차 정보 텍스트 업데이트
        if ego_info:
            info_text = f"Ego Position: ({ego_info.x:.1f}, {ego_info.y:.1f}), Yaw: {np.degrees(ego_info.yaw):.1f}°"
            if ego_info.velocity > 0:
                info_text += f", Velocity: {ego_info.velocity:.1f} m/s"
            self.artists['ego_info_text'].set_text(info_text)
        
        # Camera view 업데이트 (간단한 캐싱)
        if self.current_frame in self.image_cache:
            image_path = self.image_cache[self.current_frame]
            # 이미지 캐싱
            if not hasattr(self, '_loaded_images'):
                self._loaded_images = {}
                self._empty_image = np.zeros((480, 640, 3), dtype=np.uint8)
            
            if self.current_frame not in self._loaded_images:
                img = cv2.imread(str(image_path))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self._loaded_images[self.current_frame] = img
                # 메모리 절약을 위해 최대 30개만 캐싱
                if len(self._loaded_images) > 30:
                    # 현재 프레임에서 가장 먼 프레임 제거
                    frames = list(self._loaded_images.keys())
                    frames.sort(key=lambda x: abs(x - self.current_frame))
                    del self._loaded_images[frames[-1]]
            
            if self.current_frame in self._loaded_images:
                self.artists['camera_image'].set_data(self._loaded_images[self.current_frame])
            else:
                self.artists['camera_image'].set_data(self._empty_image)
        else:
            # 빈 이미지
            if not hasattr(self, '_empty_image'):
                self._empty_image = np.zeros((480, 640, 3), dtype=np.uint8)
            self.artists['camera_image'].set_data(self._empty_image)
        
        # 제목 업데이트
        status = "[Playing]" if self.playing else "[Paused]"
        fps = 1000 / self.interval
        self.fig.suptitle(f'Frame {self.current_frame} / {self.total_frames-1} {status} [{fps:.1f} FPS]\n'
                          f'(Space: 재생/일시정지, ←/→: 프레임, ↑/↓: 속도, ESC/Q: 종료)', fontsize=14)
        
        # 정보 텍스트 업데이트
        info_text = f"Frame: {self.current_frame} / {self.total_frames-1}\nObjects: {len(preprocessed.get('objects', []))}"
        if ego_info:
            info_text += f"\nEgo: ({ego_info.x:.1f}, {ego_info.y:.1f})"
        if self.current_frame in self.image_cache:
            info_text += f"\nImage: {self.image_cache[self.current_frame].name}"
        else:
            info_text += f"\nImage: Not found"
        self.artists['info_text'].set_text(info_text)
        
        # LDM 모드에 따른 타이틀 업데이트
        ldm_mode_text = {1: "Obstacles Type", 2: "Class ID", 3: "Both (Priority: Obstacles)"}
        self.ax1.set_title(f'Bird\'s Eye View (Ego-centric, Forward: Up) - LDM Mode: {ldm_mode_text.get(self.ldm_mode, "Unknown")}')
        
    def _update_objects(self, preprocessed, ego_info, raw_objects):
        """객체 업데이트"""
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
        
        for obj in preprocessed.get('objects', []):
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
                    dx = obj['global_x'] - prev_obj.global_x
                    dy = obj['global_y'] - prev_obj.global_y
                    if abs(dx) > 0.01 or abs(dy) > 0.01:
                        actual_heading = np.arctan2(dy, dx)
                
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
                    cos_yaw = np.cos(-ego_info.yaw)
                    sin_yaw = np.sin(-ego_info.yaw)
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
        self.prev_objects = {obj.id: obj for obj in raw_objects}
        
    def _update_direction_vectors(self, cache_data, ego_info):
        """방위벡터 업데이트"""
        if self.artists['near_arrow']:
            self.artists['near_arrow'].remove()
            self.artists['near_arrow'] = None
        if self.artists['dest_arrow']:
            self.artists['dest_arrow'].remove()
            self.artists['dest_arrow'] = None
        
        near_direction = cache_data.get('near_direction')
        destination_direction = cache_data.get('destination_direction')
        
        if near_direction and destination_direction and ego_info:
            # 근접 방위벡터
            if near_direction.get('x') is not None and near_direction.get('y') is not None:
                dx_near = near_direction['x']
                dy_near = near_direction['y']
                
                cos_yaw = np.cos(-ego_info.yaw)
                sin_yaw = np.sin(-ego_info.yaw)
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
                
                cos_yaw = np.cos(-ego_info.yaw)
                sin_yaw = np.sin(-ego_info.yaw)
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
        
    def animate_update(self, _):
        """애니메이션 업데이트 함수"""
        if self.playing:
            self.current_frame = (self.current_frame + 1) % self.total_frames
            self.update_frame_fast()
        return []
    
    def save_frame(self, frame_idx: int, save_path: Path):
        """특정 프레임을 파일로 저장"""
        self.current_frame = frame_idx
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 프레임 초기화 및 렌더링
        self.init_plot()
        self.update_frame()
        
        # 파일로 저장
        plt.tight_layout()
        self.fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(self.fig)
    
    def show(self, start_frame: int = 0):
        """시각화 시작 (애니메이션 모드)"""
        # 키보드 단축키 안내
        print("\n" + "="*60)
        print("🎮 키보드 단축키:")
        print("  Space   : 재생/일시정지")
        print("  →/←     : 다음/이전 프레임")
        print("  ↑/↓     : 재생 속도 증가/감소")
        print("  1/2/3   : LDM 모드 (장애물/세그멘테이션/둘다)")
        print("  q       : 종료")
        print("="*60 + "\n")
        
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
                                frames=self.total_frames,
                                interval=self.interval, 
                                repeat=True,
                                blit=False)
        
        # 일시정지 상태로 시작
        self.anim.pause()
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='GT 데이터 시각화 - 공통 파서 사용 버전')
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
    parser.add_argument('--save', action='store_true',
                       help='이미지 파일로 저장 (GUI 대신)')
    
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
    
    # 🔥 공통 파서 사용!
    print(f"Loading data: {jsonl_path}")
    data_parser = GTDataParser(jsonl_path)
    print(f"Total {data_parser.get_frame_count()} frames loaded")
    
    # db3 이름 추출
    db3_name = None
    if 'IO_GT_matching_' in jsonl_path.name:
        db3_name = jsonl_path.name.replace('IO_GT_matching_', '').replace('.jsonl', '')
    
    # 시각화
    print("\n=== 공통 파서 사용 버전 실행 중 ===")
    print("Controls: Space(Play/Pause), ←/→(Frame), ↑/↓(Speed), 1/2/3(LDM Mode), ESC/Q(Quit)")
    
    visualizer = FastInteractiveVisualizer(
        data_parser, img_dir, args.view_range, db3_name, args.debug, args.interval
    )
    
    if args.save:
        # 파일로 저장
        if args.frame is not None:
            # 특정 프레임만 저장
            save_path = results_dir / f"visualize_frame_{args.frame}.png"
            visualizer.save_frame(args.frame, save_path)
            print(f"Saved: {save_path}")
        else:
            print("Error: --save 옵션은 --frame과 함께 사용해야 합니다.")
    else:
        # GUI 모드
        visualizer.show(start_frame=args.frame if args.frame is not None else 0)


if __name__ == '__main__':
    main()