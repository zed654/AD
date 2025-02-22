# Deep Learning Project for Autonomous Driving

자율주행을 위한 딥러닝 프로젝트입니다. ROS2 bag 파일에서 데이터를 추출하고, 모델을 학습하고 추론하는 기능을 제공합니다.

## 프로젝트 구조

```
dl-project/
├── config/ # 설정 파일
├── data/ # 데이터 처리 관련 모듈
│ └── data_extract/ # ROS2 bag 파일 데이터 추출
├── models/ # 딥러닝 모델 정의
├── training/ # 모델 학습 관련 모듈
├── inference/ # 모델 추론 관련 모듈
└── main.py # 메인 실행 파일
```

## 설치 방법

1. 프로젝트 클론

bash:ai_ws/AD/dl-project/README.md
git clone https://github.com/yourusername/dl-project.git
cd dl-project

2. 패키지 설치

```bash
pip install -e .
```

## 사용 방법

### 데이터 추출

ROS2 bag 파일(.db3)에서 데이터를 추출합니다.

```bash
dl-extract
```

### 모델 학습

추출된 데이터를 사용하여 모델을 학습합니다.

```bash
dl-train
```

### 모델 추론

학습된 모델을 사용하여 추론을 수행합니다.

```bash
dl-infer
```

## 주요 기능

- ROS2 bag 파일(.db3) 데이터 추출 및 처리
- 딥러닝 모델 학습
- 실시간 추론

## 의존성

- Python 3.8+
- ROS2
- typer
- minio
- loguru

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

## 기여 방법

1. 이 저장소를 포크합니다.
2. 새로운 브랜치를 생성합니다.
3. 변경사항을 커밋합니다.
4. 브랜치에 푸시합니다.
5. Pull Request를 생성합니다.
