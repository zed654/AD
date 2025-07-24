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
    
    def extract_points_from_pointcloud2(self, pc2_msg, max_points=10):
        # PointCloud2 메시지에서 포인트 추출 (최대 10개만)
        points = []
        
        # PointCloud2 메시지의 데이터 해석
        point_step = pc2_msg.point_step
        row_step = pc2_msg.row_step
        data = pc2_msg.data
        
        # 모든 필드 정보 수집
        field_info = {}
        for field in pc2_msg.fields:
            field_info[field.name] = {
                'offset': field.offset,
                'datatype': field.datatype,
                'count': field.count
            }
        
        # 필수 필드 확인
        if 'x' not in field_info or 'y' not in field_info:
            return points
        
        # 데이터 타입별 언팩 포맷
        datatype_formats = {
            1: 'b',  # INT8
            2: 'B',  # UINT8
            3: 'h',  # INT16
            4: 'H',  # UINT16
            5: 'i',  # INT32
            6: 'I',  # UINT32
            7: 'f',  # FLOAT32
            8: 'd'   # FLOAT64
        }
        
        # 포인트 추출 (최대 10개)
        # max_points = 10
        point_count = 0
        
        for i in range(0, len(data), point_step):
            if i + point_step > len(data) or point_count >= max_points:
                break
            
            point = {}
            
            # 모든 필드 추출
            for field_name, info in field_info.items():
                offset = info['offset']
                datatype = info['datatype']
                
                if datatype in datatype_formats:
                    fmt = datatype_formats[datatype]
                    size = struct.calcsize(fmt)
                    
                    try:
                        value = struct.unpack(fmt, data[i + offset:i + offset + size])[0]
                        point[field_name] = value
                    except:
                        point[field_name] = None
                else:
                    point[field_name] = None
            
            points.append(point)
            point_count += 1
        
        return points
    
    def interpolate_velocity(self, wheelodom_list, target_timestamp):
        # 타겟 타임스탬프에서의 속도를 인터폴레이션
        if not wheelodom_list:
            return 0.0
        
        # 타겟 시간 전후의 wheelodom 찾기
        before_odom = None
        after_odom = None
        
        for i, odom in enumerate(wheelodom_list):
            if odom["timestamp"] <= target_timestamp:
                before_odom = odom
            else:
                after_odom = odom
                break
        
        # 인터폴레이션
        if before_odom and after_odom:
            # 선형 인터폴레이션
            t1 = before_odom["timestamp"]
            t2 = after_odom["timestamp"]
            v1 = before_odom["msg"].linear_vel
            v2 = after_odom["msg"].linear_vel
            
            # 가중치 계산
            weight = (target_timestamp - t1) / (t2 - t1)
            return v1 * (1 - weight) + v2 * weight
        elif before_odom:
            # 이전 값만 있으면 그 값 사용
            return before_odom["msg"].linear_vel
        elif after_odom:
            # 이후 값만 있으면 그 값 사용
            return after_odom["msg"].linear_vel
        else:
            return 0.0
    
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
        near_destination_direction = self.calculate_direction_vector(current_position, k_meter_point)
        
        # 목적지 방위벡터 (현재 위치에서 목적지로의 방향)
        destination_point = global_waypoints[-1]
        final_destination_direction = self.calculate_direction_vector(current_position, destination_point)
        
        return near_destination_direction, final_destination_direction

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
                        error_count += 1  # 에러 카운트도 증가시켜서 추적
                        print(f"\n\033[33m[WARNING] obstacles_tracking_msg 파싱 실패, 더미 데이터 사용 (timestamp: {timestamp}): {e}\033[0m")
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

    def make_gt(self, topics_config, n_trajectory_sample_time, output_path, k_meters=10.0, trajectory_interval_sec=0.1):
        # topics_config 딕셔너리에서 토픽 정보 추출
        # 각 토픽의 threshold와 required 설정은 topics_config 내부에 정의됨
        
        # 각 토픽 파싱
        obstacles_list = self.parse_topic(topics_config['obstacles']['topic'], topics_config['obstacles']['type'])
        odom_list = self.parse_topic(topics_config['ego_pos']['topic'], topics_config['ego_pos']['type'])
        global_way_list = self.parse_topic(topics_config['global_way']['topic'], topics_config['global_way']['type'])
        wheelodom_list = self.parse_topic(topics_config['ego_velocity']['topic'], topics_config['ego_velocity']['type'])
        ldm_msg_list = self.parse_topic(topics_config['environment_info']['topic'], topics_config['environment_info']['type'])
        
        # 디버깅 로그 추가
        print(f"\n[DEBUG] obstacles_list 길이: {len(obstacles_list)}")
        print(f"[DEBUG] odom_list 길이: {len(odom_list)}")
        print(f"[DEBUG] global_way_list 길이: {len(global_way_list)}")
        print(f"[DEBUG] wheelodom_list 길이: {len(wheelodom_list)}")
        print(f"[DEBUG] ldm_msg_list 길이: {len(ldm_msg_list)}")
        
        if len(obstacles_list) > 0:
            print(f"[DEBUG] 첫 번째 obstacle timestamp: {obstacles_list[0]['timestamp']}")
        if len(odom_list) > 0:
            print(f"[DEBUG] 첫 번째 odom timestamp: {odom_list[0]['timestamp']}")
        if len(global_way_list) > 0:
            print(f"[DEBUG] 첫 번째 global_way timestamp: {global_way_list[0]['timestamp']}")
        if len(ldm_msg_list) > 0:
            print(f"[DEBUG] 첫 번째 ldm_msg timestamp: {ldm_msg_list[0]['timestamp']}")
        
        with open(output_path, "w", encoding="utf-8") as f:
            last_sample_time = None
            matched_count = 0
            skipped_count = 0
            timeout_count = 0
            
            # Ego(odom) 기준으로 순회
            for i, odom in enumerate(tqdm(odom_list, desc="Ego-based GT matching")):
                t_ego = convert_ros2_timestamp_to_kst(odom["timestamp"])
                
                # trajectory_interval_sec 간격으로 샘플링
                if last_sample_time is not None:
                    if (t_ego - last_sample_time).total_seconds() < trajectory_interval_sec:
                        continue
                last_sample_time = t_ego
                
                # 현재 Ego 위치
                current_odom = odom
                current_position = {
                    'x': current_odom["msg"].pose.pose.position.x,
                    'y': current_odom["msg"].pose.pose.position.y
                }
                
                # 미래 trajectory 수집 (현재 시점부터 n_trajectory_sample_time 동안)
                trajectory_data = []
                last_sampled_time = None
                
                for future_odom in odom_list[i:]:  # 현재 인덱스부터 시작
                    t_future = convert_ros2_timestamp_to_kst(future_odom["timestamp"])
                    if t_ego <= t_future <= t_ego + timedelta(seconds=n_trajectory_sample_time):
                        # Trajectory 샘플링 적용
                        if last_sampled_time is None or (t_future - last_sampled_time).total_seconds() >= trajectory_interval_sec:
                            odom_dict = msg_to_dict(future_odom["msg"])
                            # 해당 시점의 속도를 인터폴레이션하여 추가
                            linear_vel = self.interpolate_velocity(wheelodom_list, future_odom["timestamp"])
                            odom_dict["_linear_vel"] = linear_vel
                            trajectory_data.append((future_odom["timestamp"], odom_dict))
                            last_sampled_time = t_future
                
                # 시간 순서로 정렬하여 올바른 trajectory 생성
                trajectory_data.sort(key=lambda x: x[0])
                gt_dict_odom = [data for _, data in trajectory_data]
                
                # 현재 시점에서 시간 범위 내의 obstacles 찾기
                obstacles_threshold = topics_config.get('obstacles', {}).get('threshold', 0.3)  # 기본값 0.3초
                obstacles_in_range = []
                for obs in obstacles_list:
                    time_diff = abs(obs["timestamp"] - odom["timestamp"]) / 1_000_000_000.0  # 나노초를 초로 변환
                    if time_diff <= obstacles_threshold:
                        obstacles_in_range.append(obs)
                
                # 가장 가까운 것 선택 (범위 내에 여러 개 있을 경우)
                closest_obstacles = None
                if obstacles_in_range:
                    closest_obstacles = min(obstacles_in_range, key=lambda x: abs(x["timestamp"] - odom["timestamp"]))
                
                # 현재 시점에서 시간 범위 내의 전역경로 찾기
                global_way_threshold = topics_config.get('global_way', {}).get('threshold', 0.3)  # 기본값 0.3초
                closest_global_way = None
                for gw in global_way_list:
                    time_diff = abs(gw["timestamp"] - odom["timestamp"]) / 1_000_000_000.0
                    if time_diff <= global_way_threshold:
                        if closest_global_way is None or abs(gw["timestamp"] - odom["timestamp"]) < abs(closest_global_way["timestamp"] - odom["timestamp"]):
                            closest_global_way = gw
                
                # 현재 시점에서 시간 범위 내의 ldm_msg 찾기
                ldm_threshold = topics_config.get('environment_info', {}).get('threshold', 0.3)  # 기본값 0.3초
                closest_ldm_msg = None
                for ldm in ldm_msg_list:
                    time_diff = abs(ldm["timestamp"] - odom["timestamp"]) / 1_000_000_000.0
                    if time_diff <= ldm_threshold:
                        if closest_ldm_msg is None or abs(ldm["timestamp"] - odom["timestamp"]) < abs(closest_ldm_msg["timestamp"] - odom["timestamp"]):
                            closest_ldm_msg = ldm
                
                # 필수 토픽 체크 (각 토픽별 설정 사용)
                skip_due_to_timeout = False
                missing_topics = []
                
                # obstacles 체크
                if topics_config.get('obstacles', {}).get('required', False):
                    if len(obstacles_in_range) == 0:
                        missing_topics.append(f"obstacles (threshold: {obstacles_threshold}s)")
                
                # ldm_msg 체크
                if topics_config.get('environment_info', {}).get('required', False):
                    if closest_ldm_msg is None:
                        missing_topics.append(f"ldm_msg (threshold: {ldm_threshold}s)")
                
                # global_way 체크
                if topics_config.get('global_way', {}).get('required', False):
                    if closest_global_way is None:
                        missing_topics.append(f"global_way (threshold: {global_way_threshold}s)")
                
                # 필수 토픽이 누락되면 skip
                if missing_topics:
                    skip_due_to_timeout = True
                    timeout_count += 1
                    if timeout_count <= 5:  # 처음 5개만 상세 로그 출력
                        print(f"\n[WARNING] Ego timestamp {odom['timestamp']} (t={t_ego}) - 필수 토픽 누락:")
                        for topic in missing_topics:
                            print(f"  - {topic}")
                
                # 필수 데이터 체크 및 저장
                if gt_dict_odom and not skip_due_to_timeout:  # trajectory가 있고 timeout이 아닐 때만 저장
                    # input_environment 구성
                    input_dict = {}
                    
                    # obstacles 정보 추가
                    if closest_obstacles:
                        input_dict.update(msg_to_dict(closest_obstacles["msg"]))
                    else:
                        # obstacles가 없으면 빈 리스트로 설정
                        input_dict["_objects"] = []
                        skipped_count += 1
                    
                    # obstacles_tracking_msg 필터링 - 필요한 필드만 하드코딩으로 추가
                    if "_objects" in input_dict:
                        filtered_objects = []
                        for obj in input_dict["_objects"]:
                            filtered_obj = {}
                            # 필요한 필드만 하드코딩으로 추가
                            if "_id" in obj:
                                filtered_obj["_id"] = obj["_id"]
                            if "_status" in obj:
                                filtered_obj["_status"] = obj["_status"]
                            if "_class_category" in obj:
                                filtered_obj["_class_category"] = obj["_class_category"]
                            if "_source_cam" in obj:
                                filtered_obj["_source_cam"] = obj["_source_cam"]
                            if "_tracking_age" in obj:
                                filtered_obj["_tracking_age"] = obj["_tracking_age"]
                            if "_class_id" in obj:
                                filtered_obj["_class_id"] = obj["_class_id"]
                            if "_model_image_detection_identificator" in obj:
                                filtered_obj["_model_image_detection_identificator"] = obj["_model_image_detection_identificator"]
                            if "_cluster_id" in obj:
                                filtered_obj["_cluster_id"] = obj["_cluster_id"]
                            if "_max_z" in obj:
                                filtered_obj["_max_z"] = obj["_max_z"]
                            if "_min_z" in obj:
                                filtered_obj["_min_z"] = obj["_min_z"]
                            if "_global_points" in obj:
                                filtered_obj["_global_points"] = obj["_global_points"]
                            if "_global_center" in obj:
                                filtered_obj["_global_center"] = obj["_global_center"]
                            if "_local_points" in obj:
                                filtered_obj["_local_points"] = obj["_local_points"]
                            if "_local_center" in obj:
                                filtered_obj["_local_center"] = obj["_local_center"]
                            if "_global_velocity" in obj:
                                filtered_obj["_global_velocity"] = obj["_global_velocity"]
                            if "_global_yaw_heading" in obj:
                                filtered_obj["_global_yaw_heading"] = obj["_global_yaw_heading"]
                            if "_local_velocity" in obj:
                                filtered_obj["_local_velocity"] = obj["_local_velocity"]
                            if "_local_yaw_heading" in obj:
                                filtered_obj["_local_yaw_heading"] = obj["_local_yaw_heading"]
                            if "_covariance" in obj:
                                filtered_obj["_covariance"] = obj["_covariance"]
                            if "_track_confidence" in obj:
                                filtered_obj["_track_confidence"] = obj["_track_confidence"]
                            if "_roi_detections" in obj:
                                filtered_obj["_roi_detections"] = obj["_roi_detections"]
                            if "_measurement_ids" in obj:
                                filtered_obj["_measurement_ids"] = obj["_measurement_ids"]
                            if "_cam_sources" in obj:
                                filtered_obj["_cam_sources"] = obj["_cam_sources"]
                            if "_gating_cost" in obj:
                                filtered_obj["_gating_cost"] = obj["_gating_cost"]
                            if "_class_covariance" in obj:
                                filtered_obj["_class_covariance"] = obj["_class_covariance"]
                            filtered_objects.append(filtered_obj)
                        input_dict["_objects"] = filtered_objects
                    
                    # 전역경로에서 방위벡터 추출
                    if closest_global_way:
                        # PointCloud2 메시지에서 포인트 추출
                        global_waypoints = self.extract_points_from_pointcloud2(closest_global_way["msg"], max_points=999999)
                        near_destination_direction, final_destination_direction = self.extract_waypoint_vectors(
                            global_waypoints, current_position, k_meters
                        )
                        
                        # 방위벡터 정보를 input_environment에 추가
                        if near_destination_direction and final_destination_direction:
                            input_dict["near_destination_direction_vector"] = near_destination_direction
                            input_dict["final_destination_direction_vector"] = final_destination_direction
                    else:
                        # global_way가 없으면 기본값 설정
                        input_dict["near_destination_direction_vector"] = {'x': 0.0, 'y': 0.0}
                        input_dict["final_destination_direction_vector"] = {'x': 0.0, 'y': 0.0}
                    
                    # ldm_msg 정보를 input_environment에 추가
                    if closest_ldm_msg:
                        # PointCloud2 메시지에서 포인트 추출
                        ldm_points = self.extract_points_from_pointcloud2(closest_ldm_msg["msg"], max_points=40000)
                        # 필요한 필드만 필터링 (class_id, obstacles_type, x, y, z)
                        filtered_ldm_points = []
                        for point in ldm_points:
                            filtered_point = {}
                            if 'class_id' in point:
                                filtered_point['class_id'] = point['class_id']
                            if 'obstacles_type' in point:
                                filtered_point['obstacles_type'] = point['obstacles_type']
                            if 'x' in point:
                                filtered_point['x'] = point['x']
                            if 'y' in point:
                                filtered_point['y'] = point['y']
                            if 'z' in point:
                                filtered_point['z'] = point['z']
                            filtered_ldm_points.append(filtered_point)
                        input_dict["_ldm_points"] = filtered_ldm_points
                    else:
                        # ldm_msg가 없으면 빈 리스트로 설정
                        input_dict["_ldm_points"] = []
                    
                    # 타임스탬프 정보 추가 (디버깅용)
                    input_dict["_ego_timestamp"] = odom["timestamp"]
                    if closest_obstacles:
                        input_dict["_obstacles_timestamp"] = closest_obstacles["timestamp"]
                    
                    f.write(json.dumps({"input_environment": input_dict, "gt_trajectory": gt_dict_odom}) + "\n")
                    matched_count += 1
        
        print(f"\n[DEBUG] 매칭된 샘플 수: {matched_count}")
        print(f"[DEBUG] 시간 범위 밖으로 스킵된 obstacles 수: {skipped_count}")
        print(f"[DEBUG] 필수 토픽 누락으로 스킵된 샘플 수: {timeout_count}")
        print(f"{matched_count}개의 GT 샘플을 {output_path}에 저장했습니다.")
        
        # 통계 반환
        return {
            'matched_count': matched_count,
            'skipped_count': skipped_count,
            'timeout_count': timeout_count,
            'success': matched_count > 0
        }

    def save_images(self, topic_name, msg_type_str, output_dir, db3_name, image_interval_sec=0.1):
        # 이미지 토픽의 메시지를 파싱하여 jpg로 저장 (Ego 시간 기준 샘플링)
        os.makedirs(output_dir, exist_ok=True)
        image_msgs = self.parse_topic(topic_name, msg_type_str)
        last_sample_time = None
        for msg in tqdm(image_msgs, desc=f"Extracting images {topic_name}"):
            timestamp = msg["timestamp"]
            t_img = convert_ros2_timestamp_to_kst(timestamp)
            if last_sample_time is not None:
                if (t_img - last_sample_time).total_seconds() < image_interval_sec:
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

    n_trajectory_sample_time = 5.0 #0.05 #5.0  # GT 매칭 구간(초) (정의한 시간 동안의 미래 위치를 Trajectory의 에 반영하여, 길이 조절할때 씀)
    trajectory_interval_sec = 0.1  # Trajectory 샘플링 간격 (초) - 0.1이면 10Hz로 샘플링 ... sample_time 이 3.0 sec 이고 interval_sec 가 0.3 이면, 10개 뽑힘.
    k_meters = 20.0 #20.0  # 근접 방위벡터를 위한 거리(미터)
    
    # 전체 통계 초기화
    total_stats = {
        'total_files': len(db3_files),
        'success_files': [],
        'failed_files': [],
        'total_matched_samples': 0,
        'total_skipped_samples': 0,
        'total_timeout_samples': 0,
        'file_details': []
    }
    
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
        
        # 토픽 설정 딕셔너리 (threshold: 해당 토픽의 시간 매칭 임계값, required: 필수 여부)
        gt_topics_config = {
            'obstacles': {
                'topic': '/obstacles_tracking_msg', 
                'type': 'autonomy_ros2_message/msg/DetectedObjectArray',
                'threshold': 0.3,  # 300ms
                'required': True # 선택사항 (인터폴레이션 가능)
            },
            'ego_pos': {
                'topic': '/odomgyro', 
                'type': 'nav_msgs/msg/Odometry'
            },
            'global_way': {
                'topic': '/planned_global_way_array', 
                'type': 'sensor_msgs/msg/PointCloud2',
                'threshold': 3.0,  # 3초 (발행 주기가 5초라서)
                'required': True # 선택사항 (인터폴레이션 가능)
            },
            'ego_velocity': {
                'topic': '/wheelodom', 
                'type': 'autonomy_ros2_message/msg/MotorDriver',
                'threshold': 0.1,  # 100ms (20Hz 발행 주기)
                'required': False  # 선택사항 (인터폴레이션 가능)
            },
            'environment_info': {
                'topic': '/ldm_msg', 
                'type': 'sensor_msgs/msg/PointCloud2',
                'threshold': 0.3,  # 300ms
                'required': True # 선택사항 (인터폴레이션 가능)
            }
        }
        
        try:
            result = extractor.make_gt(
                gt_topics_config,
                n_trajectory_sample_time,
                gt_output_path,
                k_meters=k_meters,
                trajectory_interval_sec=trajectory_interval_sec
            )
            
            # 파일별 통계 저장
            file_stat = {
                'file': db3_name,
                'matched': result['matched_count'],
                'skipped': result['skipped_count'],
                'timeout': result['timeout_count'],
                'success': result['success']
            }
            total_stats['file_details'].append(file_stat)
            
            if result['success']:
                total_stats['success_files'].append(db3_name)
                total_stats['total_matched_samples'] += result['matched_count']
                total_stats['total_skipped_samples'] += result['skipped_count']
                total_stats['total_timeout_samples'] += result['timeout_count']
            else:
                total_stats['failed_files'].append(db3_name)
                
        except Exception as e:
            print(f"\n[ERROR] {db3_name} 처리 중 오류 발생: {str(e)}")
            total_stats['failed_files'].append(db3_name)
            total_stats['file_details'].append({
                'file': db3_name,
                'error': str(e),
                'success': False
            })

        # 이미지 저장
        img_output_dir = os.path.join(results_dir, "img")
        extractor.save_images(
            "/cam_f/image/reduced/compressed",
            "sensor_msgs/msg/CompressedImage",
            img_output_dir,
            db3_name,
            image_interval_sec=trajectory_interval_sec  # Ego 시간 기준과 동일한 간격 사용
        )

        extractor.close()
    
    # 최종 통계 출력
    print("\n" + "="*80)
    print("                          최종 처리 결과 요약")
    print("="*80)
    print(f"전체 DB3 파일 수: {total_stats['total_files']}")
    print(f"성공한 파일 수: {len(total_stats['success_files'])}")
    print(f"실패한 파일 수: {len(total_stats['failed_files'])}")
    print(f"\n전체 통계:")
    print(f"  - 총 매칭된 샘플: {total_stats['total_matched_samples']}")
    print(f"  - 총 스킵된 샘플 (obstacles 범위 밖): {total_stats['total_skipped_samples']}")
    print(f"  - 총 timeout 샘플 (필수 토픽 누락): {total_stats['total_timeout_samples']}")
    
    if total_stats['success_files']:
        print(f"\n✅ 성공한 파일들:")
        for file in total_stats['success_files']:
            detail = next(d for d in total_stats['file_details'] if d['file'] == file and d.get('success', False))
            print(f"  - {file}: {detail['matched']}개 매칭")
    
    if total_stats['failed_files']:
        print(f"\n❌ 실패한 파일들:")
        for file in total_stats['failed_files']:
            detail = next(d for d in total_stats['file_details'] if d['file'] == file and not d.get('success', True))
            error_msg = detail.get('error', '알 수 없는 오류')
            print(f"  - {file}: {error_msg}")
    
    print("="*80)
    print(f"처리 완료: 성공률 {len(total_stats['success_files'])/total_stats['total_files']*100:.1f}%")
    print("="*80)
    
    rclpy.shutdown()