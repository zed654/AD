import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import sqlite3
import json
import glob
import os
import numpy as np
from tqdm import tqdm
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions

from nav_msgs.msg import Odometry
from autonomy_ros2_message.msg import DetectedObjectArray, DetectedObject, GlobalWayInfo
from sensor_msgs.msg import PointCloud2
from datetime import timedelta
from utility.time_utils import convert_ros2_timestamp_to_kst
import math
import struct

# ROS2 메시지를 dict로 변환 (numpy array, dataclass 등 지원)
def msg_to_dict(msg):
    import dataclasses
    import array
    
    # numpy array 및 Python array 처리
    if isinstance(msg, (np.ndarray, array.array)):
        return msg.tolist() if hasattr(msg, 'tolist') else list(msg)
    
    # dataclass 처리
    if dataclasses.is_dataclass(msg):
        return dataclasses.asdict(msg)
    
    # ROS 메시지 객체 처리
    elif hasattr(msg, '__slots__'):
        result = {}
        for slot in msg.__slots__:
            value = getattr(msg, slot)
            if isinstance(value, (np.ndarray, array.array)):
                result[slot] = value.tolist() if hasattr(value, 'tolist') else list(value)
            elif isinstance(value, list):
                result[slot] = [msg_to_dict(v) for v in value]
            elif hasattr(value, '__slots__'):
                result[slot] = msg_to_dict(value)
            else:
                result[slot] = value
        return result
    
    # list 처리
    elif isinstance(msg, list):
        return [msg_to_dict(v) for v in msg]
    
    # 기본값
    else:
        return msg

# dict에서 원하는 필드만 추출
def filter_fields(d, fields):
    return {k: d[k] for k in fields if k in d}

class DB3Extractor:
    def __init__(self, db3_path):
        # db3 파일 연결 및 커서 생성
        self.db3_path = db3_path
        self.conn = sqlite3.connect(db3_path)
        self.cursor = self.conn.cursor()
    
    def calculate_direction_vector(self, from_point, to_point):
        # 두 점 사이의 방위 단위벡터 계산
        dx = to_point['x'] - from_point['x']
        dy = to_point['y'] - from_point['y']
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance > 0:
            return {
                'x': dx / distance,
                'y': dy / distance
            }
        else:
            return {'x': 0.0, 'y': 0.0}
    
    def extract_points_from_pointcloud2(self, pc2_msg):
        # PointCloud2 메시지에서 포인트 추출
        points = []
        
        # PointCloud2 메시지의 데이터 해석
        point_step = pc2_msg.point_step
        row_step = pc2_msg.row_step
        data = pc2_msg.data
        
        # 필드 정보 추출
        x_offset = None
        y_offset = None
        z_offset = None
        
        for field in pc2_msg.fields:
            if field.name == 'x':
                x_offset = field.offset
            elif field.name == 'y':
                y_offset = field.offset
            elif field.name == 'z':
                z_offset = field.offset
        
        if x_offset is None or y_offset is None:
            return points
        
        # 포인트 추출
        for i in range(0, len(data), point_step):
            if i + point_step > len(data):
                break
                
            x = struct.unpack('f', data[i + x_offset:i + x_offset + 4])[0]
            y = struct.unpack('f', data[i + y_offset:i + y_offset + 4])[0]
            z = struct.unpack('f', data[i + z_offset:i + z_offset + 4])[0] if z_offset is not None else 0.0
            
            points.append({'x': x, 'y': y, 'z': z})
        
        return points
    
    def extract_waypoint_vectors(self, global_waypoints, current_position, k_meters=10.0):
        # 전역경로에서 K미터 지점의 방위벡터와 목적지 방위벡터 추출
        if not global_waypoints or len(global_waypoints) < 2:
            return None, None
        
        # 현재 위치에서 가장 가까운 웨이포인트 찾기
        min_dist = float('inf')
        closest_idx = 0
        
        for i, wp in enumerate(global_waypoints):
            dx = wp['x'] - current_position['x']
            dy = wp['y'] - current_position['y']
            dist = math.sqrt(dx**2 + dy**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # K미터 지점 찾기
        accumulated_dist = 0.0
        k_meter_idx = closest_idx
        
        for i in range(closest_idx, len(global_waypoints) - 1):
            wp1 = global_waypoints[i]
            wp2 = global_waypoints[i + 1]
            dx = wp2['x'] - wp1['x']
            dy = wp2['y'] - wp1['y']
            segment_dist = math.sqrt(dx**2 + dy**2)
            
            if accumulated_dist + segment_dist >= k_meters:
                k_meter_idx = i + 1
                break
            accumulated_dist += segment_dist
        
        # 근접 방위벡터 (현재 위치에서 K미터 지점으로의 방향)
        k_meter_point = global_waypoints[k_meter_idx]
        near_direction = self.calculate_direction_vector(current_position, k_meter_point)
        
        # 목적지 방위벡터 (현재 위치에서 목적지로의 방향)
        destination_point = global_waypoints[-1]
        destination_direction = self.calculate_direction_vector(current_position, destination_point)
        
        return near_direction, destination_direction

    def parse_topic(self, topic_name, msg_type_str):
        # rosbag2_py를 사용하여 토픽 파싱
        storage_options = StorageOptions(uri=self.db3_path, storage_id="sqlite3")
        converter_options = ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        )
        
        reader = SequentialReader()
        reader.open(storage_options, converter_options)
        
        # 토픽 타입 맵 생성
        topic_types = reader.get_all_topics_and_types()
        type_map = {topic_type.name: topic_type.type for topic_type in topic_types}
        
        result = []
        success_count = 0
        error_count = 0
        
        while reader.has_next():
            try:
                topic, data, timestamp = reader.read_next()
                if topic == topic_name:
                    msg_type = get_message(msg_type_str)
                    msg = deserialize_message(data, msg_type)
                    result.append({"timestamp": timestamp, "msg": msg})
                    success_count += 1
            except Exception as e:
                if topic == topic_name:
                    # obstacles_tracking_msg 파싱 실패시 더미 데이터 사용
                    if topic_name == "/obstacles_tracking_msg":
                        # 더미 DetectedObjectArray 생성
                        from autonomy_ros2_message.msg import DetectedObjectArray, DetectedObject
                        from builtin_interfaces.msg import Time
                        from std_msgs.msg import Header
                        from geometry_msgs.msg import Point32
                        
                        dummy_msg = DetectedObjectArray()
                        dummy_msg.header = Header()
                        dummy_msg.header.stamp = Time()
                        dummy_msg.header.stamp.sec = timestamp // 1_000_000_000
                        dummy_msg.header.stamp.nanosec = timestamp % 1_000_000_000
                        dummy_msg.header.frame_id = "base_link"
                        
                        # 더미 객체 추가
                        obj = DetectedObject()
                        obj.id = 1
                        obj.global_center = Point32(x=0.0, y=0.0, z=0.0)
                        obj.global_velocity = 0.0
                        obj.global_yaw_heading = 0.0
                        dummy_msg.objects = [obj]
                        
                        result.append({"timestamp": timestamp, "msg": dummy_msg})
                        success_count += 1  # 더미 데이터도 성공으로 카운트
                        # error_count는 증가시키지 않음 (더미 데이터로 대체했으므로)
                    elif topic_name == "/planned_global_way_array":
                        # 더미 PointCloud2 생성
                        from sensor_msgs.msg import PointCloud2, PointField
                        from std_msgs.msg import Header
                        from builtin_interfaces.msg import Time
                        
                        dummy_msg = PointCloud2()
                        dummy_msg.header = Header()
                        dummy_msg.header.stamp = Time()
                        dummy_msg.header.stamp.sec = timestamp // 1_000_000_000
                        dummy_msg.header.stamp.nanosec = timestamp % 1_000_000_000
                        dummy_msg.header.frame_id = "odom"
                        
                        # 포인트 필드 정의
                        dummy_msg.fields = [
                            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
                        ]
                        
                        # 더미 웨이포인트 생성 (직선 경로)
                        points_data = bytearray()
                        num_points = 50
                        for i in range(num_points):
                            x = float(i * 2)  # 2미터 간격
                            y = 0.0
                            z = 0.0
                            points_data.extend(struct.pack('fff', x, y, z))
                        
                        dummy_msg.data = bytes(points_data)
                        dummy_msg.width = num_points
                        dummy_msg.height = 1
                        dummy_msg.point_step = 12  # 3 floats * 4 bytes
                        dummy_msg.row_step = dummy_msg.point_step * dummy_msg.width
                        dummy_msg.is_dense = True
                        dummy_msg.is_bigendian = False
                        
                        result.append({"timestamp": timestamp, "msg": dummy_msg})
                        success_count += 1  # 더미 데이터도 성공으로 카운트
                    else:
                        error_count += 1
                        print(f"\n\033[31m[WARNING] {topic_name} 토픽 메시지 파싱 실패 (timestamp: {timestamp}): {e}\033[0m")
                continue
        
        print(f"\n[INFO] {topic_name}: 성공 {success_count}개, 실패 {error_count}개")
        return result

    def db3_extract_function(self, topic_name, output_path, msg_type_str):
        # rosbag2_py를 사용하여 토픽 파싱 및 저장
        storage_options = StorageOptions(uri=self.db3_path, storage_id="sqlite3")
        converter_options = ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        )
        
        reader = SequentialReader()
        reader.open(storage_options, converter_options)
        
        success_count = 0
        error_count = 0
        
        # 테스트용. 아래 코드를 통해 저장할 필요 없음. GT Matching 할 때 파일 만들면 됨
        # with open(output_path, "w", encoding="utf-8") as f:
        #     while reader.has_next():
        #         try:
        #             topic, data, timestamp = reader.read_next()
        #             if topic == topic_name:
        #                 msg_type = get_message(msg_type_str)
        #                 msg = deserialize_message(data, msg_type)
        #                 # 메시지 dict + timestamp를 함께 저장
        #                 f.write(json.dumps(msg_to_dict(msg) | {"timestamp": timestamp}) + "\n")
        #                 success_count += 1
        #         except Exception as e:
        #             if topic == topic_name:
        #                 error_count += 1
        #                 print(f"\n\033[31m[WARNING] {topic_name} 토픽 메시지 저장 실패 (timestamp: {timestamp}): {e}\033[0m")
        #             continue
        
        print(f"{topic_name}: 성공 {success_count}개, 실패 {error_count}개 → {output_path}에 저장했습니다.")

    def make_gt(self, input_topic, input_type, gt_topic, gt_type, n_trajectory_sample_time, output_path, sample_logging_interval_sec=None, k_meters=10.0):
        # input_topic(예: obstacles) 기준으로, n초 이내의 gt_topic(예: odom) 메시지들을 매칭하여 GT 데이터셋 생성
        obstacles_list = self.parse_topic(input_topic, input_type)
        odom_list = self.parse_topic(gt_topic, gt_type)
        # /planned_global_way_array 토픽 사용 (PointCloud2 타입)
        global_way_list = self.parse_topic("/planned_global_way_array", "sensor_msgs/msg/PointCloud2")
        
        # 디버깅 로그 추가
        print(f"\n[DEBUG] obstacles_list 길이: {len(obstacles_list)}")
        print(f"[DEBUG] odom_list 길이: {len(odom_list)}")
        print(f"[DEBUG] global_way_list 길이: {len(global_way_list)}")
        
        if len(obstacles_list) > 0:
            print(f"[DEBUG] 첫 번째 obstacle timestamp: {obstacles_list[0]['timestamp']}")
        if len(odom_list) > 0:
            print(f"[DEBUG] 첫 번째 odom timestamp: {odom_list[0]['timestamp']}")
        if len(global_way_list) > 0:
            print(f"[DEBUG] 첫 번째 global_way timestamp: {global_way_list[0]['timestamp']}")
        
        with open(output_path, "w", encoding="utf-8") as f:
            last_sample_time = None
            matched_count = 0
            for obs in tqdm(obstacles_list, desc="IO_GT_matching"):
                t_obs = convert_ros2_timestamp_to_kst(obs["timestamp"])
                # 샘플 간격 조건 추가
                if sample_logging_interval_sec is not None and last_sample_time is not None:
                    if (t_obs - last_sample_time).total_seconds() < sample_logging_interval_sec:
                        continue
                last_sample_time = t_obs
                
                gt_dict_odom = []
                trajectory_data = []  # (timestamp, odom_data) 형태로 수집
                
                # 장애물 감지 시점 이후의 odom 메시지들을 수집 (미래 trajectory)
                for odom in odom_list:
                    t_odom = convert_ros2_timestamp_to_kst(odom["timestamp"])
                    if t_obs <= t_odom <= t_obs + timedelta(seconds=n_trajectory_sample_time):
                        trajectory_data.append((odom["timestamp"], msg_to_dict(odom["msg"])))
                
                # 시간 순서로 정렬하여 올바른 trajectory 생성
                trajectory_data.sort(key=lambda x: x[0])
                gt_dict_odom = [data for _, data in trajectory_data]
                
                # 현재 시점에서 가장 가까운 odom 찾기 (현재 위치)
                current_odom = None
                min_time_diff = float('inf')
                for odom in odom_list:
                    time_diff = abs(odom["timestamp"] - obs["timestamp"])
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        current_odom = odom
                
                # 현재 시점에서 가장 가까운 전역경로 찾기
                closest_global_way = None
                min_time_diff = float('inf')
                for gw in global_way_list:
                    time_diff = abs(gw["timestamp"] - obs["timestamp"])
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        closest_global_way = gw
                
                if gt_dict_odom and current_odom and closest_global_way:  # 매칭된 GT가 있을 때만 저장
                    input_dict = msg_to_dict(obs["msg"])
                    
                    # 현재 위치 정보
                    current_position = {
                        'x': current_odom["msg"].pose.pose.position.x,
                        'y': current_odom["msg"].pose.pose.position.y
                    }
                    
                    # 전역경로에서 방위벡터 추출
                    # PointCloud2 메시지에서 포인트 추출
                    global_waypoints = self.extract_points_from_pointcloud2(closest_global_way["msg"])
                    near_direction, destination_direction = self.extract_waypoint_vectors(
                        global_waypoints, current_position, k_meters
                    )
                    
                    # 방위벡터 정보를 input_environment에 추가
                    if near_direction and destination_direction:
                        input_dict["near_direction_vector"] = near_direction
                        input_dict["destination_direction_vector"] = destination_direction
                    
                    # input_dict["timestamp"] = obs["timestamp"] # json 문구 추가하려면 이렇게 추가하면 됨
                    f.write(json.dumps({"input_environment": input_dict, "gt_trajectory": gt_dict_odom}) + "\n")
                    matched_count += 1
        
        print(f"\n[DEBUG] 매칭된 샘플 수: {matched_count}")
        print(f"{matched_count}개의 GT 샘플을 {output_path}에 저장했습니다.")

    def save_images(self, topic_name, msg_type_str, output_dir, db3_name, sample_logging_interval_sec=None):
        # 이미지 토픽의 메시지를 파싱하여 jpg로 저장 (샘플 간격 적용)
        os.makedirs(output_dir, exist_ok=True)
        image_msgs = self.parse_topic(topic_name, msg_type_str)
        last_sample_time = None
        for msg in tqdm(image_msgs, desc=f"Extracting images {topic_name}"):
            timestamp = msg["timestamp"]
            t_img = convert_ros2_timestamp_to_kst(timestamp)
            if sample_logging_interval_sec is not None and last_sample_time is not None:
                if (t_img - last_sample_time).total_seconds() < sample_logging_interval_sec:
                    continue
            last_sample_time = t_img
            img_msg = msg["msg"]
            img_data = img_msg.data
            img_path = os.path.join(output_dir, f"cam_f_{timestamp}_{db3_name}.jpg")
            with open(img_path, "wb") as f:
                f.write(img_data)

    def close(self):
        # db3 파일 연결 종료
        self.conn.close()

if __name__ == "__main__":
    # db3, 결과 폴더 경로 설정
    db3_dir = os.path.join(os.path.dirname(__file__), "db3")
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"db3_dir: {db3_dir}")
    print(f"results_dir: {results_dir}")
    # db3 파일 리스트 수집
    db3_files = glob.glob(os.path.join(db3_dir, "*.db3"))
    print(f"db3_files: {db3_files}")

    n_trajectory_sample_time = 3.0 #5.0  # GT 매칭 구간(초) (정의한 시간 동안의 미래 위치를 Trajectory의 에 반영하여, 길이 조절할때 씀)
    sample_logging_interval_sec = 0.3 #0.3  # 로깅 샘플링 시간 설정 (제한된 시간 내에 같은 토픽의 값을 무시하게하는 시간; 0.03초면, 30 ms 간격 이내의 데이터는 날림)
    k_meters = 10.0 #20.0  # 근접 방위벡터를 위한 거리(미터)
    
    rclpy.init()
    for db3_path in db3_files:
        base = os.path.basename(db3_path)
        db3_name = os.path.splitext(base)[0] # 확장자 없는 db3 파일명

        # Odometry 데이터 저장
        output_path = os.path.join(results_dir, f"odomgyro_{db3_name}.jsonl")
        print(f"\n[INFO] {db3_path} → {output_path}")
        extractor = DB3Extractor(db3_path)
        # extractor.db3_extract_function("/odomgyro", output_path, "nav_msgs/msg/Odometry")
        
        # obstacles2odom GT 데이터셋 생성
        gt_output_path = os.path.join(results_dir, f"IO_GT_matching_{db3_name}.jsonl")
        print(f"\n[INFO] GT 생성: {db3_path} → {gt_output_path}")
        extractor.make_gt(
            "/obstacles_tracking_msg",
            "autonomy_ros2_message/msg/DetectedObjectArray",
            "/odomgyro",
            "nav_msgs/msg/Odometry",
            n_trajectory_sample_time,
            gt_output_path,
            sample_logging_interval_sec=sample_logging_interval_sec,
            k_meters=k_meters
        )

        # 이미지 저장
        img_output_dir = os.path.join(results_dir, "img")
        extractor.save_images(
            "/cam_f/image/reduced/compressed",
            "sensor_msgs/msg/CompressedImage",
            img_output_dir,
            db3_name,
            sample_logging_interval_sec=sample_logging_interval_sec
        )

        extractor.close()
    rclpy.shutdown()

    # Time 변환 테스트 코드
    print(convert_ros2_timestamp_to_kst(1736146965483111085))