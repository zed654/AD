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
동작 가능 상태태이고 아키텍처가 변경된 경우라면, CLAUDE.md 에 변경된 아키텍처들을 업데이트해줘. 업데이트 할 때 오늘의 날짜(yyyy-mm-dd)를 서버시간을 기준으로 기입해줘야해. 서버시간으로 묶어 기록 이후에, 전체 내용을 요약하는 설명 부분에도 변경점을 잘 정리해서 반영해야해.

visualize_muzero_episode.py 의 경우, Episode 의 정보를 시각화하는 프로그램으로, 시각화적인 표현 외에에 다른 정보들은 절대 임의로 가공하면 안돼.

데이터를 파싱하는 인덱싱 정보들은 rosseta.py 에서 가져와야해. 없다면 추가 방안을 모색해서 추가 여부를 사용자에게 물어봐.

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

## Architecture Overview (Updated 2025-07-25)

### Training/Evaluation Data Parsing
/home/chp/ai_ws/AD/dl-project/models/MuZero/data_parser.py 를 통해 학습할 데이터들을 GT 인 Json 으로부터 가져온다.
JSON GT 는 /home/chp/ai_ws/AD/dl-project/data/db3_parser/db3 의 db3 로부터 파싱되며, /home/chp/ai_ws/AD/dl-project/data/db3_parser/db3_extractor.py 를 통해 파싱된다.
모든 데이터 정의(class_id, 색상, grid values)는 /home/chp/ai_ws/AD/dl-project/data/db3_parser/rosetta.py 에 중앙 집중식으로 관리된다.

### Training Data Flow (GT + MCTS)
1. **GT Episode 생성**: `ExpertDataLoader`가 JSON에서 observation과 expert_ego_pos 추출
2. **MCTS Enrichment**: `SelfPlayManager.enrich_episodes_with_mcts()`로 각 observation에 대해 MCTS 실행
   - policy: MCTS가 계산한 action 확률 분포
   - value: MCTS가 예측한 상태 가치
   - reward: expert 이동 거리 기반 계산
3. **통합 Episode**: GT 정보(observation, expert_ego_pos) + MCTS 결과(policy, value, reward) 모두 포함
4. **학습**: Replay buffer에서 샘플링하여 네트워크 학습

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
  - `rosetta.py`: Centralized data definitions (class IDs, colors, grid values) - located in `data/db3_parser/`
- **Results**: Saved in `models/MuZero/results/` with timestamps

### SSR Models (Templates)
The `models/SSR/` contains template implementations for Structured State Representation models that need to be completed.

## Key Implementation Details

1. **MuZero Network**: Uses fully connected layers by default, configured in `models/MuZero/config.py`
2. **Training Data**: 
   - Self-play: Generated through MCTS simulations
   - Expert data: Loaded from GT JSON when using observation_provider_type="gt"
3. **Observation Shape**: Configured as (3, 99, 99) representing the driving environment state (channel, height, width)
   - Grid size: 99x99 cells (19.8m x 19.8m area with 0.2m per cell)
   - Ego position managed separately via state_dict['ego_pos'], not displayed in grid
   - **Coordinate system**: 180-degree rotated view (forward=down, right=left) for vehicle perspective
   - **Direction vectors**: Stored in vehicle coordinate system (converted from global in db3_extractor.py)
   - Channel 0: Dynamic objects from /obstacles_tracking_msg (pedestrians, vehicles)
   - Channel 1: Static environment from /ldm_msg (cliff, static obstacles, missing points)
   - Channel 2: Ground semantic class_id from /ldm_msg (segmentation information)
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

## Recent Updates (2025-07-25)

### 좌표계 180도 회전 및 방향 벡터 처리 개선
1. **좌표 변환 방식 변경**:
   - 기존: 전방이 위쪽(row 감소), 오른쪽이 오른쪽(col 증가)
   - 변경: 전방이 아래쪽(row 증가), 오른쪽이 왼쪽(col 감소)
   - 차량 뒤에서 바라보는 관점으로 180도 회전

2. **방향 벡터 처리 통합**:
   - `db3_extractor.py`: 전역 좌표계 방향 벡터를 차량 좌표계로 변환하여 JSON 저장
   - `extract_waypoint_vectors()` 함수에 ego yaw 파라미터 추가 및 변환 로직 구현
   - `data_parser.py`, `visualize_muzero_episode.py`: 추가 변환 없이 원본 데이터 사용
   - `expert_data_loader.py`: 방향 벡터를 소수점 둘째 자리까지 반올림하여 저장

3. **수정된 파일**:
   - `data_parser.py`: grid_row = center + rel_x, grid_col = center - rel_y
   - `expert_data_loader.py`: 동일한 좌표 변환 적용, 방향 벡터 round(value, 2)
   - `visualize_muzero_episode.py`: 방향 벡터 표시 수정 (dx = -y*10, dy = x*10)
   - `db3_extractor.py`: extract_waypoint_vectors에 차량 좌표계 변환 추가

4. **UI/UX 개선**:
   - Ego 토글(4번)과 방향 벡터 토글(5번) 독립적으로 작동
   - Step Information 위치 상향 조정 (0.5 → 0.7)
   - 마우스 좌표 표시 수정 (x=row, y=col)
   - 채널별 범례 동적 업데이트

## Recent Updates (2025-07-24)

### Ego 위치 관리 개선 및 시각화 UI 향상
1. **Ego 위치를 Grid에서 분리**:
   - Ego는 더 이상 Grid에 표시되지 않음 (GRID_VALUE_EGO 제거)
   - state_dict의 'ego_pos' 필드로 독립적으로 관리
   - Channel 2에 Ego class_id = -1 할당 제거

2. **시각화 UI 개선**:
   - CheckButtons 크기 축소 (좌측 0.15 → 0.10)
   - X 마커 크기 감소 (markersize 8 → 4)
   - 별표(expert position) 크기 감소 (markersize 10 → 5)
   - Ego 위치 토글 추가 (키보드 '4', 버튼 클릭)
   - Ego 위치 최우선 렌더링으로 항상 가시성 보장

3. **Channel별 독립 시각화**:
   - Grid 0, 1, 2 채널 독립적으로 토글 가능
   - 우선순위 기반 렌더링으로 색상 혼재 방지
   - get_grid_color() 함수에 channel 파라미터 추가

### 메모리 효율성 및 코드 정리
1. **중복 데이터 제거**:
   - state_dict에서 불필요한 'objects' 필드 제거
   - build_observation() 반환값 직접 활용
   - 'class_id' → 'ground_class_id'로 명칭 변경

2. **LDM 데이터 처리 개선**:
   - obstacles_type과 무관하게 모든 LDM 포인트의 class_id 저장
   - Channel 2 legend 추가 (상위 5개 클래스 표시)

### Channel 2 처리 개선 - Object class_id 제거
1. **Channel 2는 LDM semantic 정보만 저장**:
   - Channel 0: 동적 객체의 grid value (0.3=Pedestrian, 0.6=Vehicle 등)
   - Channel 1: 정적 환경의 grid value (0.3=Cliff, 0.6=Static Obstacle 등)
   - Channel 2: LDM semantic class_id만 저장 (0-25의 segmentation 정보)
   - Object의 class_id(100=Person, 202=Car 등)는 grid value로 변환되어 Channel 0에만 표시

2. **코드 수정 사항**:
   - `data_parser.py`: objects 처리 시 Channel 2에 class_id 저장하지 않음
   - `expert_data_loader.py`: 불필요한 class_id tensor 변환 코드 제거

### 3채널 Observation 구조 및 rosetta.py 도입
1. **3채널 Observation 구조로 확장**:
   - observation_shape을 (2, 99, 99)에서 (3, 99, 99)로 확장
   - Channel 0: 동적 객체 (EMPTY(0.0), PEDESTRIAN(0.3), OTHER_OBJECT(0.4), VEHICLE(0.6))
   - Channel 1: 정적 환경 (EMPTY(0.0), CLIFF(0.3), STATIC_OBSTACLE(0.6), MISSING_POINTS(0.9))
   - Channel 2: LDM semantic class_id 정보

2. **rosetta.py 도입 - 데이터 정의 통합 관리**:
   - rosetta.json을 Python 모듈로 변환하여 `/data/db3_parser/rosetta.py`에 위치
   - 모든 class_id, 색상, 매핑 정보를 중앙 집중식으로 관리
   - data_parser.py와 visualize_muzero_episode.py에서 하드코딩 완전 제거
   - 주요 내용:
     - 객체 class_id 정의 (100: Person, 202: Car, 205: Truck 등)
     - LDM segmentation 정의 (0-25: Background, Blocks, Crosswalk, Grass 등)
     - Grid values 상수 (GRID_VALUE_EMPTY, GRID_VALUE_PEDESTRIAN 등)
     - 유틸리티 함수 제공 (get_object_name, get_grid_value_info 등)

3. **코드 개선 사항**:
   - `data_parser.py`: rosetta.py에서 상수 import하여 사용
   - `visualize_muzero_episode.py`: rosetta.py import 추가, 하드코딩 제거
   - `expert_data_loader.py`: Channel 2 (class_id) 추출 로직 추가

### ObservationProvider 아키텍처 개선
1. **책임 분리**:
   - `ObservationProvider`: 데이터 제공 인터페이스 (get_frame_count, get_frame_data, reset, get_observation)
   - `ExpertDataLoader`: Episode 생성 및 관리 (시작/종료, 길이 등)
   - Episode 관리 로직이 ObservationProvider에서 ExpertDataLoader로 이동

2. **GT 학습 플로우 개선**:
   - GT Episode 생성 → MCTS enrichment → Replay buffer 저장
   - `SelfPlayManager.enrich_episodes_with_mcts()` 메서드 추가
   - GT observation에 대해 MCTS를 실행하여 policy와 value 계산

3. **Episode 구조**:
   - states: observation dictionary (obstacles, objects, ldm 등)
   - expert_ego_pos: GT에서 추출한 전문가 위치 (그리드 좌표)
   - actions: self-play action (초기값 0)
   - policies: MCTS가 계산한 action 확률 분포
   - values: MCTS가 예측한 상태 가치
   - rewards: 이동 거리 기반 계산

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

4. **시각화 개선**:
   - `visualize_muzero_episode.py`: obstacles와 static_environment 두 레이어를 하나의 99x99 그리드로 병합하여 표시
   - 동적 장애물이 있는 위치는 동적 장애물 표시, 없는 곳에만 정적 환경 표시
   - static_environment가 없거나 dict 형태인 경우 obstacles만 사용
   - Channel 2 (class_id)는 시각화에 사용하지 않음 (의미론적 정보이므로)

## Development Notes

- The project uses Korean documentation extensively - refer to README files for additional context
- Entry points defined in setup.py are not fully implemented (dl-extract, dl-train, dl-infer)
- SSR models are skeleton implementations requiring completion
- No formal testing framework or linting configuration exists yet
- Python 버전 주의: ROS2 관련 작업은 시스템 Python 3.10 사용, 학습은 conda 환경의 Python 3.8 사용