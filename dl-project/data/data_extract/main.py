"""
ROS2 bag 파일(.db3)에서 데이터를 추출하고 처리하는 메인 모듈입니다.

이 모듈은 사용자가 GUI를 통해 db3 파일을 선택하고, 
선택된 파일에서 데이터를 추출하여 처리하는 기능을 제공합니다.
"""

# import typer
import rclpy
try:
    from .extractor import db3_extract
except:
    from extractor import db3_extract

# app = typer.Typer()
# app.add_typer(db3_extract.app, name="extract")

import os
def directory_selection_gui():
    """
    GUI 기반의 파일 선택 다이얼로그를 표시합니다.

    Returns:
        str: 선택된 .db3 파일의 경로
            사용자가 파일을 선택하지 않은 경우 빈 문자열 반환
    """
    import tkinter as tk
    from tkinter import filedialog
    import json

    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # 설정 파일 경로
    config_file = os.path.join(os.path.expanduser("~"), ".bag_selector_config.json")
    
    # 이전 경로 읽기
    try:
        with open(config_file, "r") as f:
            config = json.load(f)
            last_path = config.get("last_path", os.path.expanduser("~"))
    except (FileNotFoundError, json.JSONDecodeError):
        last_path = os.path.expanduser("~")

    bag_path = filedialog.askopenfilename(
        initialdir=last_path,
        title="Select bag file",
        filetypes=(("DB3 files", "*.db3"), ("All files", "*.*")),
    )

    if bag_path:
        # 선택된 경로를 파일에 저장
        config = {"last_path": os.path.dirname(bag_path)}
        try:
            with open(config_file, "w") as f:
                json.dump(config, f)
        except Exception as e:
            print(f"설정 저장 중 오류 발생: {e}")

    return bag_path


def main():
    """
    메인 실행 함수입니다.
    
    ROS2 bag 파일을 파싱하고 동기화된 데이터를 처리합니다.
    
    프로세스:
        1. ROS2 초기화
        2. BagParser 인스턴스 생성
        3. 사용자로부터 bag 파일 선택
        4. 데이터 처리 및 동기화
        
    Raises:
        Exception: bag 파일 처리 중 발생하는 모든 예외
    """
    rclpy.init()
    parser = db3_extract.BagParser()
    # bag_path = "/home/chp/Documents/docker_shared/rosbag/sensor_data_153645_73.db3"

    bag_path = directory_selection_gui()

    try:
        parser.read_bag(bag_path)
    except Exception as e:
        parser.get_logger().error(f"Error processing bag file: {str(e)}")
        raise e
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()