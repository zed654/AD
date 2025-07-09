1. Bird's Eye View 시각화
    - 자차(노란색 사각형)와 주변 객체들 표시
    - 객체 타입별 색상 구분 (차량:빨강, 보행자:파랑 등)
    - 객체 ID와 속도 벡터 표시
  2. 카메라 이미지 동시 표시
    - 매칭된 카메라 이미지를 함께 표시
  3. 사용 방법
  # 애니메이션으로 전체 재생
  python data/db3_parser/visualize_gt.py IO_GT_matching_sensor_data_110416_63.jsonl

  # 특정 프레임만 보기
  python data/db3_parser/visualize_gt.py IO_GT_matching_sensor_data_110416_63.jsonl --frame 10

  # 뷰 범위 조정 (기본 100m)
  python data/db3_parser/visualize_gt.py IO_GT_matching_sensor_data_110416_63.jsonl --view-range 50