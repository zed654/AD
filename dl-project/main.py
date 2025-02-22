
import rclpy
# from data.data_extract.main import directory_selection_gui
from data.data_extract import main as data_extract_main
from data.data_extract.extractor import db3_extract


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

    # bag_path = directory_selection_gui()
    bag_path = data_extract_main.directory_selection_gui()
    
    try:
        parser.read_bag(bag_path)
    except Exception as e:
        parser.get_logger().error(f"Error processing bag file: {str(e)}")
        raise e
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()