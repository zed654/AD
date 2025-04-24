# db3_parser/utility/time_utils.py

from datetime import datetime, timezone, timedelta

def convert_ros2_timestamp_to_kst(timestamp):
    ros2_sec = timestamp / 1e9
    ros2_nsec = timestamp % 1e9
    total_sec = ros2_sec + ros2_nsec / 1e9

    dt_utc = datetime.fromtimestamp(total_sec, tz=timezone.utc)
    dt_kst = dt_utc.astimezone(timezone(timedelta(hours=9)))
    return dt_kst