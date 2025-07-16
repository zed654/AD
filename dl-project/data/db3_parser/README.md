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


## Rosetta Stone (rosetta.json)

`rosetta.json` 파일은 추출된 데이터의 구조와 의미를 해석하기 위한 참조 문서입니다.

### 주요 내용

1. **토픽 정보 (topics)**
   - 각 ROS2 토픽의 메시지 타입과 필드 설명
   - 필드값의 의미와 단위
   - 예: obstacles_tracking_msg, ldm_msg, odomgyro 등

2. **객체 클래스 ID 매핑**
   - Object Detection 클래스 ID (98-512)
   - 각 ID별 객체 타입 (Person, Car, Bus, Bicycle 등)
   - 카테고리 분류 (person, vehicle, static_obstacles 등)

3. **LDM(Local Dynamic Map) 정보**
   - obstacles_type: 1=절벽, 2=정적장애물, 3=누락포인트
   - class_id (Segmentation): 0-25까지의 지면/도로 타입
     - 0: Background, 2: Colored Road, 4: Crosswalk
     - 8: Grass, 21-24: 차선 타입 등

4. **좌표계 설명 (coordinate_systems)**
   - Global: 전역 좌표계 (World/Map frame)
   - Local: 자차 중심 좌표계 (Ego/Base_link frame)

5. **샘플링 파라미터 (sampling_parameters)**
   - n_trajectory_sample_time: GT trajectory 수집 시간
   - trajectory_interval_sec: Trajectory 샘플링 간격
   - sample_logging_interval_sec: 장애물 데이터 샘플링 간격

6. **시각화 색상 (visualization_colors)**
   - 각 객체/지면 타입별 HEX 색상 코드
   - visualize_gt.py에서 사용하는 색상 정보

### 사용 방법

```python
import json

# rosetta.json 로드
with open('rosetta.json', 'r') as f:
    rosetta = json.load(f)

# 객체 클래스 ID 확인
class_id = 202
object_type = rosetta['topics']['obstacles_tracking_msg']['fields']['_objects']['sub_fields']['_class_id']['values'][str(class_id)]
print(f"Class ID {class_id}: {object_type}")  # "Car (자동차)"

# LDM segmentation 타입 확인
seg_id = 4
seg_type = rosetta['topics']['ldm_msg']['fields']['class_id']['values'][str(seg_id)]
print(f"Segmentation ID {seg_id}: {seg_type}")  # "Crosswalk (횡단보도)"
```

### 파일 위치
- `data/db3_parser/rosetta.json`