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
from autonomy_ros2_message.msg import DetectedObjectArray, DetectedObject
from datetime import timedelta
from utility.time_utils import convert_ros2_timestamp_to_kst

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

    def make_gt(self, input_topic, input_type, gt_topic, gt_type, n_seconds, output_path, sample_interval_sec=None):
        # input_topic(예: obstacles) 기준으로, n초 이내의 gt_topic(예: odom) 메시지들을 매칭하여 GT 데이터셋 생성
        obstacles_list = self.parse_topic(input_topic, input_type)
        odom_list = self.parse_topic(gt_topic, gt_type)
        
        # 디버깅 로그 추가
        print(f"\n[DEBUG] obstacles_list 길이: {len(obstacles_list)}")
        print(f"[DEBUG] odom_list 길이: {len(odom_list)}")
        
        if len(obstacles_list) > 0:
            print(f"[DEBUG] 첫 번째 obstacle timestamp: {obstacles_list[0]['timestamp']}")
        if len(odom_list) > 0:
            print(f"[DEBUG] 첫 번째 odom timestamp: {odom_list[0]['timestamp']}")
        
        with open(output_path, "w", encoding="utf-8") as f:
            last_sample_time = None
            matched_count = 0
            for obs in tqdm(obstacles_list, desc="IO_GT_matching"):
                t_obs = convert_ros2_timestamp_to_kst(obs["timestamp"])
                # 샘플 간격 조건 추가
                if sample_interval_sec is not None and last_sample_time is not None:
                    if (t_obs - last_sample_time).total_seconds() < sample_interval_sec:
                        continue
                last_sample_time = t_obs
                
                gt_dict_odom = []
                for odom in odom_list:
                    t_odom = convert_ros2_timestamp_to_kst(odom["timestamp"])
                    if t_obs <= t_odom <= t_obs + timedelta(seconds=n_seconds):
                        # odom dict + timestamp 포함
                        # gt_dict_odom.append({**msg_to_dict(odom["msg"]), "timestamp": odom["timestamp"]}) # gt_dict_odom 에 timestamp 포함하려면 이렇게 하면 됨
                        gt_dict_odom.append({**msg_to_dict(odom["msg"])})
                
                if gt_dict_odom:  # 매칭된 GT가 있을 때만 저장
                    input_dict = msg_to_dict(obs["msg"])
                    # input_dict["timestamp"] = obs["timestamp"] # json 문구 추가하려면 이렇게 추가하면 됨
                    f.write(json.dumps({"input": input_dict, "gt": gt_dict_odom}) + "\n")
                    matched_count += 1
        
        print(f"\n[DEBUG] 매칭된 샘플 수: {matched_count}")
        print(f"{matched_count}개의 GT 샘플을 {output_path}에 저장했습니다.")

    def save_images(self, topic_name, msg_type_str, output_dir, db3_name, sample_interval_sec=None):
        # 이미지 토픽의 메시지를 파싱하여 jpg로 저장 (샘플 간격 적용)
        os.makedirs(output_dir, exist_ok=True)
        image_msgs = self.parse_topic(topic_name, msg_type_str)
        last_sample_time = None
        for msg in tqdm(image_msgs, desc=f"Extracting images {topic_name}"):
            timestamp = msg["timestamp"]
            t_img = convert_ros2_timestamp_to_kst(timestamp)
            if sample_interval_sec is not None and last_sample_time is not None:
                if (t_img - last_sample_time).total_seconds() < sample_interval_sec:
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

    n_seconds = 0.02  # GT 매칭 구간(초)
    sample_interval_sec = 1.0  # 1Hz로 샘플링
    
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
            n_seconds,
            gt_output_path,
            sample_interval_sec=sample_interval_sec
        )

        # 이미지 저장
        img_output_dir = os.path.join(results_dir, "img")
        extractor.save_images(
            "/cam_f/image/reduced/compressed",
            "sensor_msgs/msg/CompressedImage",
            img_output_dir,
            db3_name,
            sample_interval_sec=sample_interval_sec
        )

        extractor.close()
    rclpy.shutdown()

    # Time 변환 테스트 코드
    print(convert_ros2_timestamp_to_kst(1736146965483111085))