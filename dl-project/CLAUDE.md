# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language Preference
모든 응답은 한국어로 작성해주세요. (Please respond in Korean for all interactions.)

## Project Overview

This is a deep learning project for autonomous driving that processes ROS2 bag files (.db3) and implements reinforcement learning models, specifically MuZero for decision-making.
The Ego Agent is Mobile Robot (Simply, think of it like Humanoid Robots)

## Common Commands
프로그램을 실행하기 전에 Conda 환경을 사용해야해. 환경 활성화 방법은 커맨드라인에 conda_init 입력(alias 등록되어있어)하여 conda 환경 불러온 후, conda activate pytorch_2_2 입력하여 설정 가능해

작업 이후에는 항상 검증을해서 정상적인 동작이 가능한 상태인지 직접 변경된 코드를 실행시켜서 확인해야해
동작 가능 상태태이고 아키텍처가 변경된 경우라면, CLAUDE.md 에 변경된 아키텍처들을 업데이트해줘. 업데이트 할 때 오늘의 날짜(yyyy-mm-dd)를 서버시간을 기준으로 기입해줘야해.

visualize_muzero_episode.py 의 경우, Episode 의 정보를 시각화하는 프로그램으로, 시각화적인 표현 외에에 다른 정보들은 절대 임의로 가공하면 안돼.

<!-- ### Environment Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
``` -->

### Data Extraction from ROS2 Bags
```bash
# ROS2 환경에서 실행 필요 (conda 환경 비활성화)
conda deactivate

# Extract data from .db3 files in data/db3_parser/db3/
python3 data/db3_parser/db3_extractor.py
```

ROS2 Bags 인 db3 는 raw 파일이고, 이를 학습하기위해 가공한 파일은 json 파일이야. json 파일은 data/db3_parser/results 폴더 안에 있어.

### Data Visualization
```bash
# GT 데이터 시각화 (ROS2 환경 불필요)
cd data/db3_parser
python3 visualize_gt_fast.py  # 모든 JSONL 파일 애니메이션
python3 visualize_gt_fast.py --frame 0  # 특정 프레임만 표시

# MuZero Episode 시각화
python visualize_muzero_episode.py --save  # 파일로 저장
python visualize_muzero_episode.py  # 인터랙티브 모드 (GUI 환경 필요)
```

### Model Training
README.md 파일 내 기록
<!-- 
```bash
# Run MuZero training
python main_muzero.py

# Run with debug mode (includes visualization)
python main_muzero.py --debug
``` -->

### Monitoring Training
README.md 파일 내 기록

```bash
# View training progress in TensorBoard
tensorboard --logdir models/MuZero/results
```

## Architecture Overview

### Training/Evaluation Data Parsing
/home/chp/ai_ws/AD/dl-project/models/MuZero/data_parser.py 를 통해 학습할 데이터들을 GT 인 Json 으로부터 가져온다.
JSON GT 는 /home/chp/ai_ws/AD/dl-project/data/db3_parser/db3 의 db3 로부터 파싱되며, /home/chp/ai_ws/AD/dl-project/data/db3_parser/db3_extractor.py 를 통해 파싱된다.

### Data Processing Pipeline
The `data/db3_parser/` module extracts data from ROS2 bag files:
- **Input**: ROS2 .db3 files containing obstacle tracking, camera images, and odometry data
- **Output**: JSONL files with matched input-output pairs and extracted images
- **Key Topics**: `/obstacles_tracking_msg` (obstacles), `/ldm_msg` (environment; terrain info such as semantic class, cliff), `/odomgyro` (ego position), `wheelodom` (ego heading)

### MuZero Reinforcement Learning
The `models/MuZero/` implements the MuZero algorithm for autonomous driving:
- **Action Space**: 16 actions (8 positions × 2 stop/go states)
- **Components**:
  - `environment.py`: Driving simulation environment with collision detection (60cm threshold)
  - `mcts.py`: Monte Carlo Tree Search for action selection in latent space
  - `network.py`: Neural network with representation, dynamics, and prediction functions
  - `self_play.py`: Generates training data through self-play with collision value adjustment (-1 for collision)
  - `replay_buffer.py`: Stores and samples game experiences in Episode format
  - `train.py`: Training loop implementation with target value calculation
  - `expert_data_loader.py`: Converts GT JSON data to training episodes (used when observation_provider_type="gt")
  - `observation_provider.py`: Interface for different data sources (GT/Mock/ROS2) - only provides observations
  - `data_parser.py`: Common JSON parsing utilities for GT data + shared color/grid mapping functions
- **Results**: Saved in `models/MuZero/results/` with timestamps

### SSR Models (Templates)
The `models/SSR/` contains template implementations for Structured State Representation models that need to be completed.

## Key Implementation Details

1. **MuZero Network**: Uses fully connected layers by default, configured in `models/MuZero/config.py`
2. **Training Data**: 
   - Self-play: Generated through MCTS simulations
   - Expert data: Loaded from GT JSON when using observation_provider_type="gt"
3. **Observation Shape**: Configured as (1, 99, 99) representing the driving environment state (channel, height, width)
   - Grid size: 99x99 cells (19.8m x 19.8m area with 0.2m per cell)
   - Ego is always at grid center (49, 49) in the first step of each episode
4. **Time Matching**: 
   - Data extraction is now based on Ego (odomgyro) timestamps
   - Uses `time_matching_threshold_sec` (50ms) for closest match within range
   - Uses `required_topics_timeout_sec` (300ms) to ensure all required topics exist
5. **Collision Handling**: 
   - Environment detects collisions when objects are within 60cm of ego
   - Collision episodes get value=-1 for the final state
   - MCTS operates in latent space and cannot detect real-world collisions
6. **Value vs Reward**: 
   - Reward: Immediate reward for single step
   - Value: Expected sum of future rewards (used for training targets)

## Recent Updates (2025-07-23)

### 주요 성과
- GT 데이터 생성을 Ego 기준으로 전면 개편
- 토픽별 개별 시간 매칭으로 데이터 품질 대폭 향상 (99% 성공률)
- 프레임 점프 문제 해결 (global_way 5초 주기 문제 해결)
- 상세한 처리 결과 통계 제공

### 상세 변경사항

1. **GT 데이터 생성 방식 변경**:
   - Obstacles 기준에서 **Ego(odomgyro) 기준**으로 변경
   - `trajectory_interval_sec` (0.1초)로 Ego 시간축 샘플링 통일
   - `sample_logging_interval_sec` 파라미터 완전 제거

2. **토픽별 개별 시간 매칭 설정**:
   - `time_matching_threshold_sec` 파라미터 제거
   - 각 토픽별로 개별 `threshold`와 `required` 설정 가능
   - 예: global_way는 3초, obstacles는 0.3초 등 토픽별 최적화
   - gt_topics_config 내부에 통합 관리

3. **db3_extractor.py 리팩토링**:
   - `make_gt()`: Ego 기준 순회로 변경, 통계 정보 반환
   - `save_images()`: `image_interval_sec`로 파라미터 통일
   - 시간 범위 밖 데이터는 빈 리스트로 처리 (에러 대신)
   - 예외 처리 강화 및 파일별 처리 결과 추적

4. **최종 통계 보고 기능**:
   - 전체 처리 완료 후 상세 통계 출력
   - 성공/실패 파일 목록 및 원인 표시
   - 파일별 매칭된 샘플 수 표시
   - 전체 성공률 계산 및 출력

5. **토픽 발행 주기**:
   - odomgyro: 100Hz
   - obstacles_tracking_msg: 10Hz
   - ldm_msg: 15Hz
   - wheelodom: 20Hz
   - planned_global_way_array: 0.2Hz (5초 주기)

6. **visualize_gt_fast.py 개선**:
   - 재생 속도 조절 기능 수정 (↑/↓ 키)
   - FuncAnimation interval 실시간 변경 가능
   - 키보드 단축키 안내 메시지 추가
   - 속도 변경 시 현재 FPS 표시

## Previous Updates (2025-07-22)

1. **Episode 구조 개선**:
   - `expert_actions`: GT에서 추출한 전역좌표 (x, y) 튜플로 저장
   - `actions`: Self-play action (초기값 0)
   - `policy`: GT 데이터에서는 None으로 초기화 (MCTS에서 계산)
   - `reward`: GT 데이터에서는 0.0으로 초기화
   
2. **Expert Data Loader 리팩토링**:
   - 불필요한 reward 계산 제거
   - expert_actions을 전역좌표로 저장하여 유연성 향상
   - policy는 None으로 초기화 (self-play에서 계산)

3. **시각화 개선**:
   - `visualize_muzero_episode.py`: expert_actions 전역좌표 표시 추가
   - Policy가 None일 때 처리 추가 (GT 데이터 구분)
   - Episode 데이터를 가공하지 않고 있는 그대로 시각화

4. **Grid 크기 변경**:
   - 100x100에서 99x99로 변경하여 정확한 중심점 (49, 49) 확보
   - `config.py`: observation_shape을 (1, 99, 99)로 수정
   - `data_parser.py`: 주석 업데이트

## Previous Updates (2025-07-21)

1. **Grid Value Mapping 통합**:
   - `data_parser.py`에 색상 및 그리드 값 매핑 함수 통합
   - 상수 정의: `GRID_VALUE_EMPTY`, `GRID_VALUE_PEDESTRIAN`, `GRID_VALUE_VEHICLE`, `GRID_VALUE_EGO` 등
   - 공용 함수: `get_object_color()`, `get_grid_value_from_class_id()`, `get_grid_value_info()`

2. **Observation 개선**:
   - 객체의 중심점뿐만 아니라 모든 꼭짓점(points)도 그리드에 표시
   - Episode 첫 Step에서 Ego가 항상 그리드 정중앙(50, 50)에 위치하도록 수정

3. **시각화 개선**:
   - `visualize_muzero_episode.py`: Episode 데이터를 있는 그대로 표현, 색상 범례 추가
   - `visualize_gt_fast.py`: 공용 색상 함수 사용

## Development Notes

- The project uses Korean documentation extensively - refer to README files for additional context
- Entry points defined in setup.py are not fully implemented (dl-extract, dl-train, dl-infer)
- SSR models are skeleton implementations requiring completion
- No formal testing framework or linting configuration exists yet
- Python 버전 주의: ROS2 관련 작업은 시스템 Python 3.10 사용, 학습은 conda 환경의 Python 3.8 사용