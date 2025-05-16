의존성 파일
        pip install pandas
        sudo apt install ros-humble-rosbag2-python
        pip install tqdm


db3 파싱 python 코드 작성

    Sample
        Input
            장애물 데이터 (/obstacles_tracking_msg)
            이미지 데이터 (/cam_f)
        GT
            Trajectory (측위데이터 [/odomgyro] 미래위치 이용하여 구성)


db3 를 NDJSON 형태로 파싱

        results/IO_GT_matching_{db3 이름}.jsonl
            ({"input":......} {"input":....} 형식임)
        results/img/{cam_pos}_{timestamp}_{db3 이름}.jpg