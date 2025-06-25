의존성 파일
        pip install pandas
        sudo apt install ros-humble-rosbag2-python
        pip install tqdm

설명

    db3 에서 학습을 위한 입력데이터와 GT 를 매칭하여 추출하는 프로그램
    
    입력데이트<->GT 매칭 방법
         Input 과 GT 는 Timestamp 로 매칭하면 됨
            - 이미지: 이미지 파일 이름에 timestamp 가 있음
            - GT: jsonl 의 각 데이터에 timestamp 가 있음
    
기타

    파싱된 ndjson 은 https://jsonviewer.stack.hu/ 에 복붙하여 형태 확인 가능


데이터 구조 형태 샘플
    
        Input
            장애물 데이터 (/obstacles_tracking_msg)
            이미지 데이터 (/cam_f)
        GT
            Trajectory (측위데이터 [/odomgyro] 미래위치 이용하여 구성)


데이터 저장 형식

        results/IO_GT_matching_{db3 이름}.jsonl
            ({"input":......} {"input":....} 형식임)
        results/img/{cam_pos}_{timestamp}_{db3 이름}.jpg