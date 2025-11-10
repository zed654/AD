# 자율주행을 위한 딥러닝 프로젝트 (Deep Learning Project for Autonomous Driving)

이 프로젝트는 자율주행을 위한 딥러닝 프로젝트입니다. ROS2 bag 파일에서 데이터를 추출하고, MuZero 모델을 학습, 평가, 추론하는 기능을 제공합니다.

## 주요 기능

-   **명확한 실행 모드:** 학습(Train), 추론(Inference), 평가(Evaluate) 모드를 명확히 분리하여 사용성을 개선했습니다.
-   **유연한 설정:** JSON 설정 파일 또는 CLI 옵션을 통해 학습 및 실행 환경을 유연하게 변경할 수 있습니다.
-   **상세한 모니터링:** 학습 및 추론 과정의 상세한 로그와 결과 출력을 제공합니다.
-   **데이터 처리:** ROS2 bag 파일(.db3)에서 데이터를 추출하고 처리하는 기능을 포함합니다.

## 프로젝트 구조

```
dl-project/
├── data/             # 데이터 처리 관련 모듈
│   └── db3_parser/   # ROS2 bag 파일 데이터 추출
├── models/           # 딥러닝 모델 정의 (학습/추론 모듈)
│   └── MuZero/
├── main_muzero.py    # 메인 실행 파일
└── requirements.txt
```

## 설치 및 환경 설정

#### 1. 프로젝트 클론

```bash
git clone https://github.com/yourusername/dl-project.git
cd dl-project
```

#### 2. 가상환경 설정 및 패키지 설치

```bash
# ROS2
conda install -c conda-forge mamba
mamba create -n chp_env python=3.10
mamba activate chp_env
mamba install -c conda-forge colcon-common-extensions

conda config --env --add channels conda-forge
conda config --env --add channels robostack-staging
conda config --env --remove channels defaults
mamba install ros-humble-desktop

# ETC
mamba install -c conda-forge tqdm

# Pytorch
# conda create -n chp_env python=3.10 -y
mamba activate chp_env
mamba install pytorch=2.1.0 torchvision=0.16.0 torchaudio=2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y

pip install matplotlib pillow scipy tensorboard
pip install "numpy<2.0"

Test
 - GPU 사용 가능 여부 확인
      python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

Issue
- Intel VTune 애러 발생 시
    conda install intel::intel-vtune-profiler -y
    export PYTORCH_JIT=0
    export INTEL_JIT_PROFILING=0
- 선택사항
    export CUDA_VISIBLE_DEVICES=1
```

# tmux 설치
# 목적: 터미널 분리 (SSH 접속 후, 연결 끊겨도 살아있음)
```bash
  ㅁ tmux 사용법
   - 새로운 새션 열기: tmux new -s train
   - 새션 목록 확인: tmux ls
   - 새선 재접속: tmux attach -t train
   - 새션 분리법: tmux detach
   - 접속 여부 확인: echo $TMUX
   - 세션 종료: tmux kill-session -t train 혹은 exit

mamba activate chp_env
conda install -c conda-forge tmux

# python -m venv venv
# source venv/bin/activate
# pip install -r requirements.txt
```

# 진행 바 설치 (rich)
pip install rich

## 사용 방법

`main_muzero.py` (또는 `muzero_improved.py`)를 통해 프로젝트의 주요 기능을 실행할 수 있습니다.

---

### 0. 시각화 디버깅
# GT(jsonl) -> Python Episode 정상 파싱 여부 시각화 방법 (python3 visualize_muzero_episode.py 실행됨)
```bash
./run_viz.sh
```

### 1. 학습 모드 (Train)

#### 기본 학습
# 학습 중 시각화 결과가 models/MuZero/results 에 png 형태로 저장되고, png 샘플은 training_result_sample.png 와 같음
```bash
python3 main_muzero.py train
```

#### Ground Truth (GT) 데이터로 학습
```bash
python3 main_muzero.py train --provider gt --gt-data data/db3_parser/results/matched_data.jsonl
```

#### 이전 학습 재개
```bash
python3 main_muzero.py train --resume --checkpoint checkpoints/muzero_epoch_50.pth
```

#### 커스텀 설정을 이용한 학습
```bash
python3 main_muzero.py train --epochs 200 --config custom_config.json
```

---

### 2. 추론 모드 (Inference)

#### 기본 추론 (1 에피소드)
```bash
python main_muzero.py inference --checkpoint checkpoints/muzero_final.pth
```

#### ROS2 실시간 데이터를 이용한 추론
```bash
python main_muzero.py inference --provider ros2 --episodes 10
```

#### 렌더링 없이 빠른 추론 실행
```bash
python main_muzero.py inference --episodes 100 --no-render
```

---

### 3. 평가 모드 (Evaluate)

#### 저장된 모델 성능 평가
```bash
# 다른 GT 데이터 사용
python3 main_muzero.py evaluate --gt-data path/to/other.jsonl

# 특정 체크포인트 사용  
python3 main_muzero.py evaluate --checkpoint path/to/specific.pth
```

---

### 4. 학습 과정 모니터링

TensorBoard를 사용하여 모델의 학습 과정을 시각적으로 분석할 수 있습니다.

```bash
tensorboard --logdir models/MuZero/results
```

## 의존성

-   Python 3.8+
-   ROS2
-   PyTorch
-   Typer
-   Loguru

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

## 기여 방법

1.  이 저장소를 포크(Fork)합니다.
2.  새로운 브랜치(Branch)를 생성합니다. (`git checkout -b feature/AmazingFeature`)
3.  변경사항을 커밋(Commit)합니다. (`git commit -m 'Add some AmazingFeature'`)
4.  브랜치에 푸시(Push)합니다. (`git push origin feature/AmazingFeature`)
5.  Pull Request를 생성합니다.
