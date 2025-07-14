# db3_parser/utility/time_utils.py

from datetime import datetime, timezone, timedelta

def convert_ros2_timestamp_to_kst(timestamp):
    # ROS2 timestamp는 나노초 단위이므로 1e9로 나누어 초 단위로 변환
    total_sec = timestamp / 1e9

    dt_utc = datetime.fromtimestamp(total_sec, tz=timezone.utc)
    dt_kst = dt_utc.astimezone(timezone(timedelta(hours=9)))
    return dt_kst