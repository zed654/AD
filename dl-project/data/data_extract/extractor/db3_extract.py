#!/usr/bin/env python3
import rclpy
# import typer
from rclpy.node import Node
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import os

# app = typer.Typer()

def to_nsec(time_msg):
    """Time 메시지를 나노초 단위의 정수로 변환"""
    return time_msg.sec * 1000000000 + time_msg.nanosec

class BagParser(Node):
    def __init__(self):
        super().__init__("bag_parser")
        self.tracking_msgs = {}
        self.x_min = float("inf")
        self.x_max = float("-inf")
        self.y_min = float("inf")
        self.y_max = float("-inf")

    def update_coordinate_ranges(self, track_msg):
        """객체들의 좌표를 기반으로 x, y 범위 업데이트"""
        for obj in track_msg.objects:
            x = obj.global_center.x
            y = obj.global_center.y

            self.x_min = min(self.x_min, x)
            self.x_max = max(self.x_max, x)
            self.y_min = min(self.y_min, y)
            self.y_max = max(self.y_max, y)

    # @app.command()
    def read_bag(self, bag_path):
        try:
            # 파일 존재 여부 확인
            if not os.path.exists(bag_path):
                raise FileNotFoundError(f"Bag file not found: {bag_path}")

            # 로그를 통해 bag 파일 읽기 시작을 알림
            self.get_logger().info(f"Starting to read bag file: {bag_path}")

            # StorageOptions 객체 생성
            # - uri: 읽을 bag 파일의 경로
            # - storage_id: 저장소 타입 (sqlite3 형식의 데이터베이스)
            storage_options = StorageOptions(uri=bag_path, storage_id="sqlite3")

            # ConverterOptions 객체 생성
            # - input_serialization_format: 입력 데이터 직렬화 형식 (CDR: Common Data Representation)
            # - output_serialization_format: 출력 데이터 직렬화 형식
            # cdr 외에 json, xml 형식도 지원
            converter_options = ConverterOptions(
                input_serialization_format="cdr", 
                output_serialization_format="cdr"
            )

            # SequentialReader 인스턴스 생성
            # - rosbag 파일을 순차적으로 읽기 위한 리더 객체
            reader = SequentialReader()

            # 생성된 reader로 bag 파일 열기
            # - storage_options: 저장소 설정
            # - converter_options: 변환 설정
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

                        # Periodic logging
                        if msg_count % 1000 == 0:
                            self.get_logger().info(
                                f"Processed {msg_count} messages "
                                f"({tracking_msg_count} tracking)"
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
                        f"({tracking_msg_count} tracking)"
                    )
                    self.get_logger().info(
                        f"X range: [{self.x_min:.2f}, {self.x_max:.2f}]"
                    )
                    self.get_logger().info(
                        f"Y range: [{self.y_min:.2f}, {self.y_max:.2f}]"
                    )
                else:
                    raise RuntimeError("No valid coordinate ranges found in bag file")

                if len(self.tracking_msgs) == 0:
                    raise RuntimeError("No tracking messages found in bag file")

            except Exception as e:
                raise RuntimeError(f"Error processing bag file content: {str(e)}")

        except Exception as e:
            self.get_logger().error(f"Error reading bag file: {str(e)}")
            # 예외를 다시 발생시켜 상위 레벨에서 처리할 수 있게 함
            raise


def main():
    rclpy.init()
    parser = BagParser()

    bag_path = "/home/chp/Documents/docker_shared/rosbag/sensor_data_153645_73.db3"

    try:
        parser.read_bag(bag_path)
    except Exception as e:
        parser.get_logger().error(f"Error processing bag file: {str(e)}")
        raise e
    finally:
        rclpy.shutdown()