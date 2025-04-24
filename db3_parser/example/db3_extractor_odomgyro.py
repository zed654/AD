import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import sqlite3
import json

def extract_odom_from_db3(db3_path, topic_name, output_path):
    # db3 파일 연결
    conn = sqlite3.connect(db3_path)
    cursor = conn.cursor()

    # topic_id 찾기
    cursor.execute("SELECT id, type FROM topics WHERE name=?", (topic_name,))
    topic_row = cursor.fetchone()
    if not topic_row:
        print(f"토픽 {topic_name}을(를) 찾을 수 없습니다.")
        return
    topic_id, msg_type = topic_row

    # 메시지 추출
    cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id=?", (topic_id,))
    messages = cursor.fetchall()

    # 메시지 타입 동적 import
    msg_module = get_message(msg_type)

    # 결과 저장 (예: jsonl)
    with open(output_path, "w", encoding="utf-8") as f:
        for timestamp, data in messages:
            msg = deserialize_message(data, msg_module)
            # Odometry 메시지에서 필요한 정보만 추출 (예: position, orientation)
            odom_dict = {
                "timestamp": timestamp,
                "position": {
                    "x": msg.pose.pose.position.x,
                    "y": msg.pose.pose.position.y,
                    "z": msg.pose.pose.position.z,
                },
                "orientation": {
                    "x": msg.pose.pose.orientation.x,
                    "y": msg.pose.pose.orientation.y,
                    "z": msg.pose.pose.orientation.z,
                    "w": msg.pose.pose.orientation.w,
                },
                "linear_velocity": {
                    "x": msg.twist.twist.linear.x,
                    "y": msg.twist.twist.linear.y,
                    "z": msg.twist.twist.linear.z,
                },
                "angular_velocity": {
                    "x": msg.twist.twist.angular.x,
                    "y": msg.twist.twist.angular.y,
                    "z": msg.twist.twist.angular.z,
                }
            }
            f.write(json.dumps(odom_dict) + "\n")

    print(f"{len(messages)}개의 Odometry 메시지를 {output_path}에 저장했습니다.")
    conn.close()

if __name__ == "__main__":
    db3_path = "/home/chp/ai_ws/db3_parser/db3/sensor_data_160245_99.db3"
    topic_name = "/odomgyro"
    output_path = "results/odomgyro.jsonl"

    rclpy.init()
    extract_odom_from_db3(db3_path, topic_name, output_path)
    rclpy.shutdown()

