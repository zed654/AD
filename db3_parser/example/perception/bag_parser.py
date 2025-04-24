#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import cv2
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import os
from pathlib import Path

# 커스텀 메시지 import
from autonomy_ros2_message.msg import DetectedObjectArray, DetectedObject
from sensor_msgs.msg import CompressedImage


def to_nsec(time_msg):
    """Time 메시지를 나노초 단위의 정수로 변환"""
    return time_msg.sec * 1000000000 + time_msg.nanosec


class BagParser(Node):
    def __init__(self):
        super().__init__("bag_parser")
        self.filter_ids = None
        self.filter_classes = []  # 추가
        self.bridge = CvBridge()
        self.progress_callback = None
        self.total_frames = 0
        self.output_path = None
        self.processed_frames = 0
        self._cleanup_needed = True
        self._current_processing_images = {}  # 현재 처리 중인 이미지만 저장
        import yaml

        with open(
            Path.home()
            / "ros2_ws/install/perception_2d_resource/share/perception_2d_resource/params/object_detection_class_category.yaml",
            "r",
        ) as file:
            self.object_config = yaml.safe_load(file)

        # color mapping 생성
        self.color_mapping = {}
        self.id_to_class = {}

        for category in self.object_config:
            for class_name, obj_info in self.object_config[category].items():
                self.color_mapping[obj_info["id"]] = obj_info["color"]
                self.id_to_class[obj_info["id"]] = class_name
        # Previous initializations remain the same...
        self.image_msgs = {
            "cam_f": {},
            "cam_fl": {},
            "cam_fr": {},
            "cam_bl": {},
            "cam_br": {},
        }
        self.process_image = {
            "cam_f": {},
            "cam_fl": {},
            "cam_fr": {},
            "cam_bl": {},
            "cam_br": {},
        }
        self.tracking_msgs = {}
        self.object_history = {}
        # Add variables to store min/max coordinates
        self.x_min = float("inf")
        self.x_max = float("-inf")
        self.y_min = float("inf")
        self.y_max = float("-inf")

        # Other initializations remain the same...
        self.video_writer = None
        self.frame_size = None
        self.output_path = "output.mp4"
        self.fps = 15

    def hex_to_bgr(self, hex_color):
        """HEX 색상 코드를 BGR로 변환"""
        hex_color = hex_color.lstrip("#")
        rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
        return (rgb[2], rgb[1], rgb[0])  # RGB to BGR

    def get_color_by_id(self, obj_type):
        """객체 타입에 따른 색상 반환"""
        # YAML에서 정의된 mapping을 사용하여 type에 따른 color 반환
        default_color = "#000000"  # 기본 색상 (검정)

        # YAML 구조에 따라 객체 타입별 색상 매핑
        if obj_type in self.color_mapping:
            return self.color_mapping[obj_type]
        return default_color

    def cleanup(self):
        """리소스 정리"""

        if not self._cleanup_needed:
            return

        if hasattr(self, "video_writer") and self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        # 큰 데이터 구조 정리
        self.image_msgs.clear()
        self.process_image.clear()
        self.tracking_msgs.clear()
        self.object_history.clear()

        self._cleanup_needed = False

    def set_progress_callback(self, callback):
        """진행상황을 보고할 콜백 함수 설정"""
        self.progress_callback = callback

    def update_coordinate_ranges(self, track_msg):
        """객체들의 좌표를 기반으로 x, y 범위 업데이트"""
        for obj in track_msg.objects:
            x = obj.global_center.x
            y = obj.global_center.y

            self.x_min = min(self.x_min, x)
            self.x_max = max(self.x_max, x)
            self.y_min = min(self.y_min, y)
            self.y_max = max(self.y_max, y)

    def _process_messages_batch(self, messages, type_map):
        """메시지 배치 처리"""
        for topic_name, data, t in messages:
            msg_type = get_message(type_map[topic_name])
            msg = deserialize_message(data, msg_type)

            timestamp = to_nsec(msg.header.stamp) if hasattr(msg, "header") else t

            if topic_name == "/obstacles_tracking_msg":
                self.tracking_msgs[timestamp] = msg
                self.update_coordinate_ranges(msg)
            elif "image/reduced/compressed" in topic_name:
                cam_name = topic_name.split("/")[1]  # cam_f, cam_fl 등
                if cam_name in self.image_msgs:
                    # 이미지는 필요할 때까지 압축된 상태로 저장
                    self.image_msgs[cam_name][timestamp] = msg.data

    def read_bag(self, bag_path):
        try:
            # 파일 존재 여부 확인
            if not os.path.exists(bag_path):
                raise FileNotFoundError(f"Bag file not found: {bag_path}")

            self.get_logger().info(f"Starting to read bag file: {bag_path}")

            storage_options = StorageOptions(uri=bag_path, storage_id="sqlite3")
            converter_options = ConverterOptions(
                input_serialization_format="cdr", output_serialization_format="cdr"
            )

            reader = SequentialReader()
            reader.open(storage_options, converter_options)

            try:
                topic_types = reader.get_all_topics_and_types()
                if not topic_types:
                    raise RuntimeError("No topics found in bag file")

                self.get_logger().info(f"Found topics: {[t.name for t in topic_types]}")
                type_map = {
                    topic_type.name: topic_type.type for topic_type in topic_types
                }

                # Initialize counters for logging
                msg_count = 0
                tracking_msg_count = 0
                image_msg_count = 0

                # First pass: Calculate coordinate ranges
                while reader.has_next():
                    try:
                        topic_name, data, t = reader.read_next()
                        msg_count += 1

                        msg_type = get_message(type_map[topic_name])
                        msg = deserialize_message(data, msg_type)

                        if topic_name == "/obstacles_tracking_msg":
                            self.update_coordinate_ranges(msg)
                            self.tracking_msgs[to_nsec(msg.header.stamp)] = msg
                            tracking_msg_count += 1

                        elif "image/reduced/compressed" in topic_name:
                            cam_name = topic_name.split("/")[1]  # Extract camera name
                            if cam_name in self.image_msgs:
                                self.image_msgs[cam_name][
                                    to_nsec(msg.header.stamp)
                                ] = msg
                                image_msg_count += 1

                        # Periodic logging
                        if msg_count % 1000 == 0:
                            self.get_logger().info(
                                f"Processed {msg_count} messages "
                                f"({tracking_msg_count} tracking, {image_msg_count} images)"
                            )

                    except Exception as e:
                        self.get_logger().warn(f"Error processing message: {str(e)}")
                        continue

                # Add padding to the ranges
                if self.x_max > self.x_min and self.y_max > self.y_min:
                    x_padding = (self.x_max - self.x_min) * 0.1
                    y_padding = (self.y_max - self.y_min) * 0.1

                    self.x_min -= x_padding
                    self.x_max += x_padding
                    self.y_min -= y_padding
                    self.y_max += y_padding

                    self.get_logger().info(
                        f"Finished reading bag file. "
                        f"Total messages: {msg_count} "
                        f"({tracking_msg_count} tracking, {image_msg_count} images)"
                    )
                    self.get_logger().info(
                        f"X range: [{self.x_min:.2f}, {self.x_max:.2f}]"
                    )
                    self.get_logger().info(
                        f"Y range: [{self.y_min:.2f}, {self.y_max:.2f}]"
                    )
                else:
                    raise RuntimeError("No valid coordinate ranges found in bag file")

                self.total_frames = len(self.tracking_msgs)
                if self.total_frames == 0:
                    raise RuntimeError("No tracking messages found in bag file")

                if self.progress_callback:
                    self.progress_callback(0)

            except Exception as e:
                raise RuntimeError(f"Error processing bag file content: {str(e)}")

        except Exception as e:
            self.get_logger().error(f"Error reading bag file: {str(e)}")
            # 예외를 다시 발생시켜 상위 레벨에서 처리할 수 있게 함
            raise

    def create_grid_image(self, timestamp):
        # 이미지 로딩
        images = {
            cam: self.process_image[cam].get(timestamp)
            for cam in ["cam_f", "cam_fl", "cam_fr", "cam_bl", "cam_br"]
        }

        first_valid_image = next(
            (img for img in images.values() if img is not None), None
        )
        if first_valid_image is None:
            return None

        height, width = first_valid_image.shape[:2]

        # 객체 위치 히스토리 업데이트
        track_msg = self.tracking_msgs.get(timestamp)
        if track_msg:
            for obj in track_msg.objects:
                # 필터링된 ID만 처리
                if self.filter_ids is not None and obj.id not in self.filter_ids:
                    continue

                if self.filter_classes and obj_class_name not in self.filter_classes:
                    continue

                # 객체 위치 히스토리 업데이트
                if obj.id not in self.object_history:
                    self.object_history[obj.id] = []

                # 현재 프레임의 위치 추가
                current_pos = {
                    "x": obj.global_center.x,
                    "y": obj.global_center.y,
                    "timestamp": timestamp,
                    "speed": obj.global_velocity,
                    "angle": obj.global_yaw_heading,
                }
                self.object_history[obj.id].append(current_pos)

                # 히스토리 길이 제한 (최근 100개 프레임)
                self.object_history[obj.id] = self.object_history[obj.id][-100:]

        # 객체 히스토리 기반 범위 재계산
        x_positions = []
        y_positions = []
        if self.filter_ids is not None:
            # 각 필터된 ID에 대해 반복
            for object_id in self.filter_ids:
                if object_id in self.object_history:
                    # 히스토리의 모든 위치 수집
                    x_positions.extend(
                        [pos["x"] for pos in self.object_history[object_id]]
                    )
                    y_positions.extend(
                        [pos["y"] for pos in self.object_history[object_id]]
                    )

            # 새로운 범위 계산 (히스토리 기반)
            if x_positions and y_positions:
                x_min = min(x_positions)
                x_max = max(x_positions)
                y_min = min(y_positions)
                y_max = max(y_positions)

                # 만약 x 또는 y 범위가 0이면 약간의 오프셋 추가
                if x_min == x_max:
                    x_min -= 1
                    x_max += 1
                if y_min == y_max:
                    y_min -= 1
                    y_max += 1

                # 범위에 여유 공간 추가 (10%)
                x_padding = (x_max - x_min) * 0.1
                y_padding = (y_max - y_min) * 0.1

                x_min -= x_padding
                x_max += x_padding
                y_min -= y_padding
                y_max += y_padding
            else:
                # 히스토리가 비어있다면 기존 범위 유지
                x_min, x_max = self.x_min, self.x_max
                y_min, y_max = self.y_min, self.y_max
        else:
            # 필터링된 ID가 없다면 기존 범위 유지
            x_min, x_max = self.x_min, self.x_max
            y_min, y_max = self.y_min, self.y_max
        # 안전한 나눗셈을 위한 추가 검사
        if x_max == x_min:
            x_max = x_min + 1
        if y_max == y_min:
            y_max = y_min + 1
        # 플롯 이미지 생성
        plot_image = np.ones((height, width, 3), dtype=np.uint8) * 255

        # 플롯 마진 및 축 그리기
        margin = 50
        plot_width = width - 2 * margin
        plot_height = height - 2 * margin

        # 축 그리기
        cv2.line(
            plot_image,
            (margin, height - margin),
            (width - margin, height - margin),
            (0, 0, 0),
            2,
        )  # x축
        cv2.line(
            plot_image, (margin, margin), (margin, height - margin), (0, 0, 0), 2
        )  # y축

        # ID 표시 (왼쪽 상단)
        if self.filter_ids is not None:
            x_offset = 0
            for i, filter_id in enumerate(self.filter_ids):
                cv2.putText(
                    plot_image,
                    f"Tracking ID: {filter_id}",
                    (width - margin - x_offset, margin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                )
                x_offset += 150

        if self.filter_ids is not None and self.filter_ids:
            # filter_id가 있을 때만 히스토리 plot 그리기
            for obj_id, history in self.object_history.items():
                # 필터링된 ID만 그리기
                if obj_id not in self.filter_ids:
                    continue

                for idx, pos in enumerate(history):
                    # 색상을 시간에 따라 변경 (보라에서 빨강으로)
                    color_ratio = idx / len(history)
                    color = (
                        int(255 * (1 - color_ratio)),  # Blue
                        0,  # Green
                        int(255 * color_ratio),  # Red
                    )

                    x_plot = int(
                        margin + ((pos["x"] - x_min) / (x_max - x_min)) * plot_width
                    )
                    y_plot = int(
                        height
                        - margin
                        - ((pos["y"] - y_min) / (y_max - y_min)) * plot_height
                    )

                    # 점 크기를 최근 위치일수록 크게
                    point_size = max(2, int(5 * (idx / len(history))))
                    cv2.circle(plot_image, (x_plot, y_plot), point_size, color, -1)
                    if idx == len(history) - 1:
                        cv2.putText(
                            plot_image,
                            f"ID:{obj_id}",
                            (x_plot + 10, y_plot),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),  # 빨간색 텍스트
                            1,
                        )
        else:
            # filter_id가 없을 때는 현재 프레임의 객체 위치만 표시
            if track_msg:
                for obj in track_msg.objects:
                    x_plot = int(
                        margin
                        + ((obj.global_center.x - x_min) / (x_max - x_min)) * plot_width
                    )
                    y_plot = int(
                        height
                        - margin
                        - ((obj.global_center.y - y_min) / (y_max - y_min))
                        * plot_height
                    )

                    # 화살표 그리기 (속도와 방향 표시)
                    arrow_length = min(
                        50 * (obj.global_velocity / 5.0), 100
                    )  # 속도에 따른 화살표 길이 (최대 100픽셀)
                    angle_rad = np.radians(obj.global_yaw_heading)

                    # 화살표 끝점 계산
                    arrow_end_x = int(x_plot + arrow_length * np.cos(angle_rad))
                    arrow_end_y = int(y_plot + arrow_length * np.sin(angle_rad))

                    # 화살표 시작점과 끝점을 튜플로 생성
                    start_point = (x_plot, y_plot)
                    end_point = (arrow_end_x, arrow_end_y)

                    # 화살표 그리기
                    cv2.arrowedLine(
                        plot_image,
                        start_point,
                        end_point,
                        (0, 0, 255),  # 빨간색
                        2,
                        tipLength=0.3,
                    )

                    # 모든 객체를 빨간색 점으로 표시
                    cv2.circle(plot_image, (x_plot, y_plot), 5, (0, 0, 255), -1)
                    cv2.putText(
                        plot_image,
                        f"ID:{obj.id}",
                        (x_plot + 10, y_plot),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                    )

        # 축 눈금 그리기
        for i in range(5):  # x축 눈금
            x = margin + (plot_width * i // 4)
            y = height - margin
            val = x_min + (x_max - x_min) * i // 4
            cv2.line(plot_image, (x, y + 5), (x, y - 5), (0, 0, 0), 1)
            cv2.putText(
                plot_image,
                f"{val:.1f}",
                (x - 20, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

        for i in range(5):  # y축 눈금
            x = margin
            y = margin + (plot_height * i // 4)
            val = y_max - (y_max - y_min) * i // 4
            cv2.line(plot_image, (x - 5, y), (x + 5, y), (0, 0, 0), 1)
            cv2.putText(
                plot_image,
                f"{val:.1f}",
                (x - 35, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )
        current_frame = (
            f"Img Time: {track_msg.header.stamp.sec}.{track_msg.header.stamp.nanosec}"
        )
        cv2.putText(
            plot_image,
            f"Frame: {current_frame}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        # 그리드 레이아웃
        grid_layout = [["cam_fl", "cam_f", "cam_fr"], ["cam_bl", "plot", "cam_br"]]
        grid_height = height * 2
        grid_width = width * 3
        grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

        for row_idx, row in enumerate(grid_layout):
            for col_idx, cam in enumerate(row):
                y_start = row_idx * height
                x_start = col_idx * width

                if cam == "plot":
                    image_to_place = plot_image
                else:
                    image_to_place = images.get(
                        cam, np.zeros((height, width, 3), dtype=np.uint8)
                    )

                if image_to_place is not None:
                    grid_image[
                        y_start : y_start + height, x_start : x_start + width
                    ] = image_to_place

        return grid_image

    def calculate_camera_direction(self, obj, cam_name):
        """
        객체의 global velocity를 카메라 좌표계로 변환
        Returns: (dx, dy, dz) - 카메라 좌표계에서의 이동 방향
        """

        # global velocity를 x,y 성분으로 분해
        speed = obj.global_velocity
        angle_rad = np.radians(obj.global_yaw_heading)
        dx = speed * np.cos(angle_rad)  # global x 방향 속도
        dy = speed * np.sin(angle_rad)  # global y 방향 속도

        if cam_name == "cam_f":
            # 전방 카메라 (차량 기준 0도)
            camera_dx = dy  # global y -> camera x (좌우)
            camera_dy = 0  # 수직 운동은 무시
            camera_dz = -dx  # global x -> camera z (전후), 부호 반대로 변경

        elif cam_name == "cam_fl":
            # 전방 좌측 카메라 (차량 기준 72도)
            angle_rad = np.radians(72)
            camera_dx = dx * np.sin(angle_rad) + dy * np.cos(angle_rad)
            camera_dy = 0
            camera_dz = dx * np.cos(angle_rad) - dy * np.sin(
                angle_rad
            )  # 부호 반대로 변경

        elif cam_name == "cam_fr":
            # 전방 우측 카메라 (차량 기준 -72도)
            angle_rad = np.radians(-72)
            camera_dx = dx * np.sin(angle_rad) + dy * np.cos(angle_rad)
            camera_dy = 0
            camera_dz = dx * np.cos(angle_rad) - dy * np.sin(
                angle_rad
            )  # 부호 반대로 변경

        elif cam_name == "cam_bl":
            # 후방 좌측 카메라 (차량 기준 144도)
            angle_rad = np.radians(144)
            camera_dx = dx * np.sin(angle_rad) + dy * np.cos(angle_rad)
            camera_dy = 0
            camera_dz = dx * np.cos(angle_rad) - dy * np.sin(
                angle_rad
            )  # 부호 반대로 변경

        elif cam_name == "cam_br":
            # 후방 우측 카메라 (차량 기준 -144도)
            angle_rad = np.radians(-144)
            camera_dx = dx * np.sin(angle_rad) + dy * np.cos(angle_rad)
            camera_dy = 0
            camera_dz = dx * np.cos(angle_rad) - dy * np.sin(
                angle_rad
            )  # 부호 반대로 변경

        else:
            camera_dx = 0
            camera_dy = 0
            camera_dz = 0
            self.get_logger().warn(f"Unknown camera: {cam_name}")

        return camera_dx, camera_dy, camera_dz

    def draw_bbox(self, img_msg, track_msg, cam_name, time_stamp, min_time_diffs):
        if isinstance(img_msg, bytes):
            np_arr = np.frombuffer(img_msg, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            cv_image = img_msg

        if cam_name not in self.process_image:
            self.process_image[cam_name] = {}
        """이미지에 bbox를 그리고 저장"""
        try:
            np_arr = np.frombuffer(img_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if cv_image is None:
                self.get_logger().error("Failed to decode image")
                return

            processed_objects = 0

            for obj in track_msg.objects:

                obj_class_name = self.id_to_class.get(obj.class_id, "unknown")

                if self.filter_classes:
                    if obj_class_name not in self.filter_classes:
                        continue

                cam_mapping = {
                    1: "cam_f",
                    5: "cam_fr",
                    2: "cam_fl",
                    4: "cam_br",
                    3: "cam_bl",
                }
                cam_name_from_obj = cam_mapping.get(obj.source_cam, None)

                if cam_name_from_obj == cam_name:
                    processed_objects += 1

                    # bbox 좌표 계산
                    center_x = obj.roi_detection.x_offset * 0.49382716049
                    center_y = obj.roi_detection.y_offset * 0.49382716049
                    width = obj.roi_detection.width * 0.49382716049960
                    height = obj.roi_detection.height * 0.49382716049

                    x1 = int(center_x - width / 2)
                    y1 = int(center_y - height / 2)
                    x2 = int(center_x + width / 2)
                    y2 = int(center_y + height / 2)

                    # object type에 따른 색상 가져오기
                    color = self.get_color_by_id(obj.class_id)
                    # HEX to BGR 변환
                    color_bgr = self.hex_to_bgr(color)
                    # bbox 그리기
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), color_bgr, 2)

                    cam_angle_offset = {
                        "cam_f": 0,  # 정면 카메라
                        "cam_fl": 72,  # 전방 좌측
                        "cam_fr": -72,  # 전방 우측
                        "cam_bl": 144,  # 후방 좌측
                        "cam_br": -144,  # 후방 우측
                    }

                    # draw_bbox 메서드 내에서
                    if cam_name_from_obj == cam_name:
                        # bbox 중심점 계산은 그대로 유지
                        center_point = (int(center_x), int(center_y))

                        # 카메라 좌표계에서의 이동 방향 계산
                        dx, dy, dz = self.calculate_camera_direction(obj, cam_name)

                        # 화살표 길이 계산 (박스 크기에 비례)
                        arrow_length = min(width, height) * 0.8
                        speed_scale = min(obj.global_velocity / 5.0, 2.0)
                        arrow_length = arrow_length * speed_scale

                        # z방향 속도에 따라 화살표 각도 조절
                        # dz가 음수면 다가오는 것(아래 방향), 양수면 멀어지는 것(위 방향)
                        if abs(dz) > abs(dx):
                            # 주로 전후 운동
                            if dz < 0:
                                angle_deg = 90  # 아래 방향
                            else:
                                angle_deg = -90  # 위 방향
                            # dx의 영향을 추가 (측면 이동 표현)
                            angle_deg += np.degrees(np.arctan2(dx, abs(dz))) * 0.5
                        else:
                            # 주로 측면 운동
                            angle_deg = np.degrees(np.arctan2(dx, abs(dz)))

                        # 화살표 끝점 계산
                        angle_rad = np.radians(angle_deg)
                        arrow_end_x = int(center_x + arrow_length * np.sin(angle_rad))
                        arrow_end_y = int(center_y + arrow_length * np.cos(angle_rad))

                        # 화살표 그리기
                        cv2.arrowedLine(
                            cv_image,
                            center_point,
                            (arrow_end_x, arrow_end_y),
                            color_bgr,
                            2,
                            tipLength=0.3,
                        )

                    # ID 텍스트 표시
                    id_text = f"ID: {obj.id}"
                    font_scale = 0.7
                    thickness = 2
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    # 텍스트 크기 계산
                    (text_width, text_height), baseline = cv2.getTextSize(
                        id_text, font, font_scale, thickness
                    )

                    # ID를 bbox 위에 표시
                    text_x = x1
                    text_y = y1 - 10  # bbox보다 약간 위에 표시

                    # 텍스트 배경 그리기 (가독성을 위해)
                    cv2.rectangle(
                        cv_image,
                        (text_x, text_y - text_height - 5),
                        (text_x + text_width + 5, text_y + 5),
                        (255, 255, 255),
                        -1,
                    )  # 흰색 배경

                    # 텍스트 테두리 그리기
                    cv2.rectangle(
                        cv_image,
                        (text_x, text_y - text_height - 5),
                        (text_x + text_width + 5, text_y + 5),
                        color_bgr,
                        1,
                    )  # 초록색 테두리

                    # 텍스트 그리기
                    cv2.putText(
                        cv_image,
                        id_text,
                        (text_x, text_y),
                        font,
                        font_scale,
                        color_bgr,
                        thickness,
                    )
                    # 진행 상황 로깅
                    if processed_objects % 10 == 0:
                        self.get_logger().info(
                            f"Camera: {cam_name}, Time: {time_stamp}, "
                            f"Processed objects: {processed_objects}, "
                            f"Bbox coordinates: ({x1}, {y1}), ({x2}, {y2})"
                        )
            current_frame = f"Img Time: {img_msg.header.stamp.sec}.{img_msg.header.stamp.nanosec}, {min_time_diffs}"
            cv2.putText(
                cv_image,
                f"Frame: {current_frame}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            self.process_image[cam_name][time_stamp] = cv_image

            self.get_logger().info(
                f"Completed frame processing - Camera: {cam_name}, "
                f"Time: {time_stamp}, "
                f"Total objects processed: {processed_objects}"
            )

        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")
            raise e

    def process_synchronized_data(self, time_threshold=0.05):
        """이미지와 bbox 데이터를 시간 동기화하여 처리하고 비디오 생성"""
        try:
            threshold_nsec = int(time_threshold * 1e9)
            processed_count = 0

            # 메모리 효율을 위해 시간 순서대로 처리
            sorted_times = sorted(self.tracking_msgs.keys())
            batch_size = min(100, len(sorted_times))

            for i in range(0, len(sorted_times), batch_size):
                batch_times = sorted_times[i : i + batch_size]

                # 배치 단위로 프레임 처리
                for track_time_nsec in batch_times:
                    closest_img_times = {cam: None for cam in self.image_msgs.keys()}
                    min_time_diffs = {
                        cam: float("inf") for cam in self.image_msgs.keys()
                    }

                    # 각 카메라별로 가장 가까운 시간의 이미지 찾기
                    for cam in self.image_msgs.keys():
                        for img_time_nsec in self.image_msgs[cam].keys():
                            time_diff = abs(track_time_nsec - img_time_nsec)
                            if time_diff < min_time_diffs[cam]:
                                min_time_diffs[cam] = time_diff
                                closest_img_times[cam] = img_time_nsec

                    # 모든 카메라가 시간 임계값 내에 있는 경우에만 처리
                    if all(
                        time_diff <= threshold_nsec
                        for time_diff in min_time_diffs.values()
                    ):
                        # bbox 그리기 처리
                        for cam in self.image_msgs.keys():
                            if closest_img_times[cam] is not None:
                                self.draw_bbox(
                                    self.image_msgs[cam][closest_img_times[cam]],
                                    self.tracking_msgs[track_time_nsec],
                                    cam,
                                    track_time_nsec,
                                    min_time_diffs[cam],
                                )

                        # 그리드 이미지 생성
                        grid_image = self.create_grid_image(track_time_nsec)
                        if grid_image is not None:
                            if self.video_writer is None:
                                height, width = grid_image.shape[:2]
                                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                                self.video_writer = cv2.VideoWriter(
                                    self.output_path, fourcc, self.fps, (width, height)
                                )
                                self.get_logger().info(
                                    f"Initialized video writer with frame size: {(width, height)}"
                                )

                            self.video_writer.write(grid_image)
                            processed_count += 1

                            if processed_count % 100 == 0:
                                self.get_logger().info(
                                    f"Processed {processed_count} frames"
                                )

                    self.processed_frames += 1
                    if self.progress_callback:
                        progress = int(
                            (self.processed_frames / self.total_frames) * 100
                        )
                        self.progress_callback(progress)

                # 처리된 데이터 정리
                for t in batch_times:
                    self.tracking_msgs.pop(t, None)
                    for cam in self.image_msgs:
                        self.image_msgs[cam].pop(t, None)
                        self.process_image[cam].pop(t, None)

            if self.video_writer is not None:
                self.video_writer.release()
                self.get_logger().info(
                    f"Finished processing {processed_count} frames. Video saved to {self.output_path}"
                )

        finally:
            self.cleanup()


def main():
    rclpy.init()
    parser = BagParser()

    bag_path = "/home/neubility/bag/2222/sensor_data_111041_13.db3"

    try:
        parser.read_bag(bag_path)
        parser.process_synchronized_data()
    except Exception as e:
        parser.get_logger().error(f"Error processing bag file: {str(e)}")
        raise e
    finally:
        parser.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
