# ROS2 Bag Video Converter and Player

이 애플리케이션은 ROS2 bag 파일(.db3)을 비디오 파일(MP4)로 변환하고 재생할 수 있는 GUI 도구입니다.

## 주요 기능

- ROS2 bag 파일을 MP4 비디오로 변환
- 단일 파일 또는 폴더 단위 일괄 처리 지원
- 실시간 변환 진행률 표시
- 통합 비디오 플레이어 제공
- 객체 클래스 기반 필터링
- 객체 ID 기반 필터링

## 필수 요구사항

- Python 3.x
- PyQt5
- OpenCV (cv2)
- ROS2
- rclpy
- NumPy

## 설치 방법

1. 필요한 패키지 설치:
```bash
pip install PyQt5 opencv-python numpy
```

2. ROS2 설치 (아직 설치하지 않은 경우):
- [ROS2 설치 가이드](https://docs.ros.org/en/humble/Installation.html) 참조

## 사용 방법

1. 프로그램 실행:
```bash
python3 bag_video_converter.py
```

2. 파일 선택:
   - "Select Bag File": 단일 또는 다수의 .db3 파일 선택
   - "Select Folder": .db3 파일이 있는 폴더 선택

3. 필터링 옵션 설정:
   - 왼쪽 패널의 체크박스를 통해 특정 객체 클래스 선택
   - 변환 시작 시 객체 ID 입력 가능

4. 비디오 변환:
   - "Convert Next" 버튼을 클릭하여 변환 시작
   - 진행률 바에서 변환 진행 상황 확인

5. 비디오 재생:
   - 변환된 MP4 파일은 자동으로 재생됨
   - 재생 컨트롤:
     - Space: 재생/일시정지
     - 좌우 화살표: 이전/다음 프레임
     - 재생 바를 통한 구간 이동

## 주요 단축키

- `Space`: 재생/일시정지
- `←`: 이전 프레임
- `→`: 다음 프레임

## 필터링 기능

### 클래스 필터링
다음 카테고리별 객체 클래스 필터링 지원:
- Person: person
- Vehicle: bicycle, bus, car, motorcycle, scooter, truck, cart
- Static Obstacles: bollard
- Dynamic Obstacles: neubie
- Unclassified: animal, wheelchair, stroller

### ID 필터링
- 쉼표(,)로 구분된 객체 ID 입력
- -1 입력 시 모든 객체 포함

## 주의사항

1. 변환 중인 파일은 취소할 수 없으므로 신중하게 선택
2. 대용량 bag 파일의 경우 변환에 시간이 소요될 수 있음
3. 변환된 MP4 파일은 원본 bag 파일과 동일한 디렉토리에 저장

## 추가 정보

- 변환된 MP4 파일명: `[원본파일명]_frame_reader.mp4`
- 이미 변환된 파일은 MP4 파일 목록에 표시
- 변환 중 프로그램을 종료하면 현재 진행 중인 변환이 중단됨
