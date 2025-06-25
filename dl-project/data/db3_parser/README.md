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
    

사용법

    1. 추출할 db3 입력
        data/db3_parser/db3 에 추출할 db3 파일 넣기
            - sensor_data_110416_63.db3
            - sensor_data_110516_64.db3
            - ...

    2. 추출 프로그램 실행
        python3 data/db3_parser/db3/db3_extractor.py 실행

    3. 추출 결과 확인
        data/db3_parser/db3/result 에서 결과 확인
            jsonl(ndjson) 의 경우, https://jsonviewer.stack.hu/ 에 복붙하여 형태 확인 가능

기타
    db3 파싱 샘플 코드는 example/db3_extractor_odomgyro.py 참고


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