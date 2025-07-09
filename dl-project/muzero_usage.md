# MuZero 개선된 인터페이스 사용법

## 🎯 개선 사항
- **Train과 Inference 명확히 분리**: 각 모드별 전용 함수
- **CLI 인터페이스 개선**: 직관적인 명령어 옵션
- **Evaluate 모드 추가**: 모델 성능 평가 전용 모드
- **유연한 설정 관리**: JSON 파일 또는 CLI 옵션으로 설정 가능

## 📋 사용 예시

### 1. 학습 모드 (Train)

#### 기본 학습
```bash
python muzero_improved.py train
```

#### GT 데이터로 학습
```bash
python muzero_improved.py train --provider gt --gt-data data/db3_parser/results/matched_data.jsonl
```

#### 학습 재개
```bash
python muzero_improved.py train --resume --checkpoint checkpoints/muzero_epoch_50.pth
```

#### 커스텀 설정으로 학습
```bash
python muzero_improved.py train --epochs 200 --config custom_config.json
```

### 2. 추론 모드 (Inference)

#### 기본 추론 (1 에피소드)
```bash
python muzero_improved.py inference --checkpoint checkpoints/muzero_final.pth
```

#### ROS2 실시간 데이터로 추론
```bash
python muzero_improved.py inference --provider ros2 --episodes 10
```

#### 렌더링 없이 빠른 추론
```bash
python muzero_improved.py inference --episodes 100 --no-render
```

### 3. 평가 모드 (Evaluate)

#### 모델 성능 평가
```bash
python muzero_improved.py evaluate --checkpoint checkpoints/muzero_final.pth --episodes 100
```

## 🛠️ 설정 파일 예시

`custom_config.json`:
```json
{
    "observation_shape": [10, 12],
    "env_max_steps": 1000,
    "num_selfplay_games": 10,
    "batch_size": 256,
    "lr_init": 0.01,
    "num_training_epochs": 200,
    "observation_provider_type": "gt",
    "gt_data_path": "data/db3_parser/results/custom_data.jsonl"
}
```

## 📊 출력 예시

### Train 모드
```
============================================================
🎓 MuZero 학습 모드
============================================================
📱 Device: cuda
🎯 GT 데이터 사용: data/db3_parser/results/matched_data.jsonl

🚀 학습 시작 (총 100회)

📍 Epoch 1/100
  1️⃣ Self-play 데이터 수집...
     - 평균 길이: 45.3
     - 평균 보상: 12.5
  2️⃣ 네트워크 학습...
  💾 체크포인트 저장: checkpoints/muzero_epoch_1.pth
```

### Inference 모드
```
============================================================
🚗 MuZero 추론 모드
============================================================
📱 Device: cuda
📂 체크포인트 로드: checkpoints/muzero_final.pth
🤖 ROS2 실시간 데이터 사용

🎮 1개 에피소드 실행

📍 Episode 1/1
  Step 1: Action=Position.POS2, StopAndGo.GO, Reward=0.50
  Step 2: Action=Position.POS2, StopAndGo.GO, Reward=0.45
  ...

  📊 Episode 1 결과:
     - 총 스텝: 87
     - 총 보상: 42.35
     - 종료 이유: goal_reached
     - 액션 분포: {
        "Position.POS2, StopAndGo.GO": 65,
        "Position.POS1, StopAndGo.GO": 22
     }
```

### Evaluate 모드
```
============================================================
📊 MuZero 평가 모드
============================================================

🎮 100개 에피소드로 평가 시작...
  진행: 10/100
  진행: 20/100
  ...

📈 평가 결과:
   - 평균 보상: 35.7 (±12.3)
   - 평균 에피소드 길이: 92.4
   - 성공률: 78.0%
   - 최고 보상: 98.5
   - 최저 보상: -23.1
```

## 🔧 주요 차이점 (기존 vs 개선)

### 기존 (`muzero.py`)
- 모든 기능이 하나의 main() 함수에 혼재
- Train/Inference 구분 없음
- 하드코딩된 설정값
- 제한적인 CLI 옵션

### 개선 (`muzero_improved.py`)
- 모드별 전용 함수 (train_mode, inference_mode, evaluate_mode)
- 명확한 용도 구분
- 유연한 설정 관리 (JSON 파일, CLI 옵션)
- 풍부한 CLI 인터페이스
- 체크포인트 저장/로드 기능
- 상세한 진행 상황 표시