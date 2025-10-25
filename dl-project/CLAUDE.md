# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language Preference
모든 응답은 한국어로 작성해주세요. (Please respond in Korean for all interactions.)

## Project Overview

This is a deep learning project for autonomous driving that processes ROS2 bag files (.db3) and implements reinforcement learning models, specifically MuZero for decision-making.
The Ego Agent is Mobile Robot (Simply, think of it like Humanoid Robots)

## Common Commands
프로그램을 실행하기 전에 Conda 환경을 사용해야해. 환경 활성화 방법은 커맨드라인에 conda_init 입력(alias 등록되어있어)하여 conda 환경 불러온 후, conda activate chp_env 입력하여 설정 가능해

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

## Architecture Overview (Updated 2025-07-30)

### Training/Evaluation Data Parsing
/home/chp/ai_ws/AD/dl-project/models/MuZero/data_parser.py 를 통해 학습할 데이터들을 GT 인 Json 으로부터 가져온다.
JSON GT 는 /home/chp/ai_ws/AD/dl-project/data/db3_parser/db3 의 db3 로부터 파싱되며, /home/chp/ai_ws/AD/dl-project/data/db3_parser/db3_extractor.py 를 통해 파싱된다.
모든 데이터 정의(class_id, 색상, grid values)는 /home/chp/ai_ws/AD/dl-project/data/db3_parser/rosetta.py 에 중앙 집중식으로 관리된다.

### Training Data Flow (GT + MCTS) - Updated 2025-09-10
1. **GT Episode 생성**: `ExpertDataLoader`가 JSON에서 observation과 expert_ego_pos 추출 (330개 Episode)
2. **MCTS Enrichment**: `SelfPlayManager.enrich_episodes_with_mcts()`로 각 observation에 대해 MCTS 실행
   - **policy**: MCTS가 계산한 action 확률 분포 (30회 시뮬레이션 기반)
   - **value**: MCTS가 예측한 상태 가치 (root node value)
   - **action**: 결정적 선택 (가장 많이 방문한 action, 확률적 선택 사용 안함)
   - **reward**: 이론적으로 정확한 potential-based shaping + EMA smoothness + collision penalty
3. **Target Value 계산**: n-step bootstrap으로 학습용 target_values 별도 계산
4. **통합 Episode**: GT + MCTS + Target Values 모두 포함된 완전한 학습 데이터
5. **학습**: Replay buffer에서 배치 샘플링하여 네트워크 학습 (1000 steps/epoch)

### Data Processing Pipeline
The `data/db3_parser/` module extracts data from ROS2 bag files:
- **Input**: ROS2 .db3 files containing obstacle tracking, camera images, and odometry data
- **Output**: JSONL files with matched input-output pairs and extracted images
- **Key Topics**: `/obstacles_tracking_msg` (obstacles), `/ldm_msg` (environment; terrain info such as semantic class, cliff), `/odomgyro` (ego position), `wheelodom` (ego heading)

### MuZero Reinforcement Learning
The `models/MuZero/` implements the MuZero algorithm for autonomous driving:
- **Action Space**: 9 actions (9 positions: current, 8 directions)
- **Components**:
  - `environment.py`: Driving simulation environment with collision detection (60cm threshold)
  - `mcts.py`: Monte Carlo Tree Search for action selection in latent space
  - `models.py`: **Advanced neural network with FiLM+ConvGRU dynamics and DirectionalGating representation**
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
   - **Primary**: GT Episode + MCTS Enrichment (99.9% of training data)
     - GT observations from JSON files
     - MCTS-computed policies and values for each observation
     - Potential-based reward shaping for goal following
     - EMA-based smoothness rewards for trajectory consistency
   - **Secondary**: Pure Self-play (minimal usage, num_selfplay_games=5)
   - **Action Selection**: All deterministic (MCTS best action), no stochastic selection used
3. **Observation Shape**: Configured as (6, 99, 99) representing the driving environment state (channel, height, width)
   - Grid size: 99x99 cells (19.8m x 19.8m area with 0.2m per cell)
   - Ego position managed separately via Episode's expert_ego_pos field (grid coordinates), not displayed in grid
   - **Coordinate system**: 180-degree rotated view (forward=down, right=left) for vehicle perspective
   - **Direction vectors**: Fixed from first step of Episode (vehicle coordinate system at Episode start)
   - Channel 0: Dynamic objects from /obstacles_tracking_msg (pedestrians, vehicles)
   - Channel 1: Static environment from /ldm_msg (cliff, static obstacles, missing points)
   - Channel 2: Ground semantic class_id from /ldm_msg (segmentation information)
   - Channel 3: Near destination direction vector (x, y)
   - Channel 4: Final destination direction vector (x, y)
   - Channel 5: Simulated ego position (row, col in grid coordinates)
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

## Recent Updates (2025-09-19)

### MuZero 학습 구조 개선 및 최적화 (2025-09-19)

#### 1. **GT 데이터 폴더 전체 활용**
모든 JSONL 파일을 학습에 활용하도록 설정 변경:
- **폴더 경로 사용**: `gt_data_path = Path("data/db3_parser/results/")`
- **모든 데이터 활용**: `gt_num_episodes = 999999` (안전장치로 가능한 만큼만 생성)
- **다중 파일 지원**: ExpertDataLoader가 폴더 내 모든 JSONL 파일 자동 로드
- **데이터 품질 향상**: 단일 파일 대신 전체 데이터셋으로 더 robust한 학습

#### 2. **학습 아키텍처 단순화**
불필요한 중첩 루프 제거 및 책임 분리:
- **Epoch 개념 정리**: MuZero에서 training_steps가 실질적인 epoch 역할
- **구조 단순화**: `iteration → training_steps` (기존 `iteration → epoch → training_steps` 제거)
- **책임 분리**:
  - `main_muzero.py`: 전체 워크플로우 관리
  - `train.py`: 순수 학습 로직 및 시각화
- **Deep Copy 적용**: Episode 객체 수정 방지로 replay buffer 안정성 확보

#### 3. **Replay Buffer 최적화**
버퍼 관리 및 데이터 품질 개선:
- **FIFO 방식**: 오래된 Episode 자동 제거로 메모리 효율성
- **용량 증가**: `replay_buffer_size = 330000` (더 많은 데이터 보관)
- **Episode 추적**: 출처 정보 (source_file, start_frame, end_frame) 관리
- **깊은 복사**: 원본 expert_episodes 보호로 데이터 무결성 보장

#### 4. **시각화 및 모니터링 강화**
학습 진행 상황 실시간 확인:
- **학습 시각화**: `enable_training_visualization = True`
- **저장 간격**: `visualization_interval_iteration = 10`
- **체크포인트**: `pth_checkpoint_interval_iteration = 10`
- **손실 추적**: value_loss, policy_loss, reward_loss 개별 모니터링

#### 5. **성능 설정 최적화**
현재 학습 파라미터:
```python
# 핵심 학습 설정
self.num_iterations = 10          # Iteration 수
self.training_steps = 10          # Step당 학습 (일시적 축소, 원래 1000)
self.num_simulations = 50         # MCTS 시뮬레이션 횟수
self.batch_size = 128             # 배치 크기
self.unroll_steps = 5             # K-step unrolling

# Episode 생성 설정
self.gt_num_episodes = 999999     # 모든 데이터 활용
self.max_episode_step_length = 25 # Episode 길이
self.gt_episode_overlap = 10      # 중첩 스텝
```

#### 6. **학습 데이터 플로우**
1. **GT 데이터 로드**: 폴더 내 모든 JSONL 파일에서 Episode 생성
2. **MCTS Enrichment**: `enrich_episodes_with_mcts()`로 정책/가치 계산
3. **Replay Buffer**: 풍부해진 Episode들을 FIFO 방식으로 저장
4. **배치 학습**: K-step unrolling으로 네트워크 업데이트
5. **체크포인트**: 주기적으로 모델 상태 저장

#### 7. **예상 학습 효과**
- **데이터 다양성**: 전체 데이터셋 활용으로 일반화 성능 향상
- **학습 안정성**: Deep copy와 버퍼 관리로 데이터 무결성 보장
- **모니터링**: 실시간 시각화로 학습 진행 상황 명확히 파악
- **확장성**: 새로운 JSONL 파일 추가 시 자동으로 학습 데이터에 포함

## Previous Updates (2025-09-12)

### Predicted Path 우측상단 수렴 문제 분석 (2025-09-12)

#### 1. **문제 현상**
학습 중 시각화 결과에서 predicted path가 지속적으로 우측상단으로 수렴하는 현상 발생:
- **Top Action**: pos2 (상단) 66.67% 선택
- **Action Averages**: pos3 (우측상단) -0.020으로 가장 높은 reward
- **좌표계**: Action 정의와 시각화 간 일치성 확인됨

#### 2. **원인 분석**
**주요 원인**: 이전 편향된 체크포인트에서 학습 재개
```python
# config.py:106
self.resume_checkpoint_path = "/home/chp/.../2025-09-10--20-17-02/checkpoints/muzero_epoch_25.pth"
```
- 2025-09-10 학습에서 이미 우측상단 편향 발생
- 해당 체크포인트로부터 추가 학습하여 편향성 강화
- pos3(우측상단) action이 높은 reward를 받도록 네트워크 가중치 편향

#### 3. **좌표계 검증**
- **Action 정의**: pos2="상단" = (dx=+1, dy=0), pos3="우측상단" = (dx=+1, dy=+1)
- **좌표 변환**: `simulated_ego_row += dx`, `simulated_ego_col += dy`
- **시각화**: matplotlib 수학 좌표계 (y값 증가 = 위쪽)
- **결론**: 좌표계는 정상적으로 일치함

#### 4. **해결 방안**
1. **새 학습 시작**: `resume_checkpoint_path = None` 설정
2. **편향 이전 체크포인트 사용**: 더 이른 epoch의 체크포인트 활용
3. **Reward 함수 검증**: pos3 방향 편향 요소 확인
4. **학습 모니터링**: Action 분포의 균등성 지속 관찰

#### 5. **교훈**
- 체크포인트 재개 시 이전 편향성이 누적될 수 있음
- Action Average는 각 action의 평균 reward를 의미
- 시각화에서 정책 편향성 조기 감지 중요

## Previous Updates (2025-09-10)

### Reward 함수 이론적 개선 및 최적화 (2025-09-10)

#### 1. **Potential-based Reward Shaping 구현**
기존의 단순한 거리 기반 reward를 이론적으로 정확한 **potential-based reward shaping**으로 완전 재구성:

**핵심 개선사항:**
- **이론적 정확성**: γ * Φ(s') - Φ(s) 공식으로 optimal policy 보존
- **미터 단위 변환**: `meters_per_cell = 0.2`로 물리적 의미 명확화
- **데드존 구현**: 목표 근처 0.3m 내에서 진동/파밍 방지
- **터미널 상태 처리**: Episode 종료 시 shaping 중단으로 MDP 준수
- **gamma_shaping = 1.0**: 할인으로 인한 바닥값 문제 해결 (탁월한 아이디어!)

**새로운 Goal Following Reward:**
```python
def _calculate_goal_following_reward(self, episode: Episode, step_idx: int) -> float:
    # 미터 단위 변환
    goal = to_meters(episode.expert_ego_pos[-1])
    p_t = to_meters(st[step_idx].get('simulated_ego_pos'))
    
    # 데드존 처리 (진동 방지)
    if np.linalg.norm(p_t - goal) <= r_goal:
        return 0.0
    
    # Potential-based shaping (gamma=1.0으로 바닥값 방지)
    gamma_shaping = 1.0
    delta_phi = gamma_shaping * Phi(p_tp1) - Phi(p_t)
    
    # 이중 클리핑으로 세밀한 제어
    r = alpha_p * np.clip(shaped, -c_inner, c_inner)
    return float(np.clip(r, -0.1, 0.1))
```

#### 2. **Smoothness Reward 정밀화**
직진성 강화를 위한 **EMA 기반 smoothness reward** 구현:
- **방향 일관성**: 코사인 유사도로 연속된 이동 벡터 간 일관성 측정
- **EMA 평활화**: β=0.7로 변동성 완화 및 노이즈 제거
- **방향 반전 페널티**: 급격한 방향 전환 시 추가 페널티
- **반환 범위**: [-0.03, 0.03]로 적절한 강도 유지

#### 3. **충돌 감지 시스템 강화**
3x3 그리드 영역에서 정밀한 충돌 검사:
- **Dynamic objects** (Channel 0): 차량, 보행자 등
- **Static environment** (Channel 1): 절벽, 정적 장애물 등
- **즉시 종료**: 충돌 발생 시 Episode 즉시 terminate (-1.0 페널티)
- **Tensor 호환성**: PyTorch Tensor와 NumPy Array 모두 처리

#### 4. **학습 데이터 품질 향상**
- **Target Goal 시각화**: 학습 진행 상황 PNG에 보라색 별표로 목표점 표시
- **Action 다양성 확보**: MCTS 시뮬레이션 30회로 증가하여 탐색 강화
- **Replay Buffer 안정성**: 더 이상 빈 buffer 문제 없음
- **Loss 감소 추세**: 1.77 → 0.31로 안정적인 학습 진행

#### 5. **파라미터 설정 최적화**
현재 학습에서 **Self-play를 거의 사용하지 않음**이 확인됨:
- **주요 학습 방식**: `enrich_episodes_with_mcts()` (GT + MCTS)
- **Self-play 비중**: `num_selfplay_games = 5` (매우 적음)
- **Action 선택**: 모두 결정적 선택 (MCTS 최선 action)
- **`num_stochastic_steps`**: 현재 사용하지 않음 (주석으로 표시)

**설정 정리:**
```python
# config.py 주요 설정
self.gt_num_episodes = 330           # GT Episode 수 (폴더 전체 활용)
self.training_steps = 1000           # Epoch당 학습 스텝 증가
self.num_simulations = 30            # MCTS 시뮬레이션 횟수 증가
self.num_stochastic_steps = 30       # [지금 얘 안씀] 확률적 선택
```

## Recent Updates (2025-09-02)

### MuZero 네트워크 아키텍처 대폭 개선 (2025-09-02)

#### 1. **Advanced Dynamics 함수 재설계**
기존의 단순한 CNN 기반 dynamics를 **FiLM + ConvGRU** 아키텍처로 완전 재구성:

**기존 문제점:**
- Action이 전역적으로만 작용 (모든 위치에 동일한 action 정보 복제)
- 시간적 일관성 부족으로 MCTS rollout에서 불안정
- Reward 예측이 next_latent만 활용하여 action의 즉각 효과 반영 부족

**새로운 아키텍처:**
```python
# 1. FiLM (Feature-wise Linear Modulation): Action Conditioning
action_modulated = self.action_film(latent, action_one_hot)  # 채널별 modulation

# 2. ConvGRU: Temporal Consistency  
next_latent = self.dynamics_gru(action_modulated, latent)    # 이전 상태를 hidden으로 활용

# 3. Enhanced Reward Prediction
reward_input = cat([prev_features, next_features, action_one_hot])  # 3가지 정보 결합
```

**핵심 개선사항:**
- **FiLM**: Action이 latent feature를 채널별로 조절하여 지역적 효과 표현
- **ConvGRU**: 시간적 일관성 확보로 MCTS rollout 안정성 크게 향상
- **Enhanced Reward**: 현재/다음 상태 + action 정보를 모두 활용하여 정확한 reward 예측
- **Clean Architecture**: `_normalize_action_input()` 헬퍼 함수로 코드 가독성 향상

#### 2. **Representation 함수 최적화**
**Goal Vector 차원 축소:**
- `goal_vec` 차원을 5D → 4D로 축소 (불필요한 거리 정보 제거)
- 방향 벡터만 활용: [cosθ_near, sinθ_near, cosθ_final, sinθ_final]
- `DirectionalGating`도 `gdim=4`로 조정하여 효율성 향상

**데이터 처리 개선:**
- EDT(Euclidean Distance Transform) 계산을 위해 binary mask 사용
- Repulsion 분기에서 EDT만 활용 (occupancy_grid 제거)
- 불필요한 정규화 단계 제거로 정보 손실 최소화

#### 3. **Action Space 및 MCTS 개선**
**MCTS 시뮬레이션 증가:**
- `num_simulations = 5 → 30`으로 증가하여 더 다양한 action 탐색
- 이전에 모든 스텝에서 Action 0 (제자리)만 선택하던 문제 해결

**예상 효과:**
- 시각화에서 **Predicted Path (빨간색 경로)**가 제대로 나타날 것
- MCTS가 의미 있는 다양한 action을 선택하게 됨

#### 4. **네트워크 구조 상세**
**새로 추가된 모듈:**
```python
# FiLM for Action Conditioning
self.action_film = FiLM(action_dim=9, feature_dim=64, hidden_dim=64)

# ConvGRU for Temporal Consistency  
self.dynamics_gru = ConvGRU(input_dim=64, hidden_dim=64, kernel_size=3)

# Enhanced Reward Network
reward_input_size = 64 + 64 + 9  # prev + next + action
self.dynamics_reward_network = mlp(reward_input_size, fc_reward_layers, support_size)
```

**데이터 플로우:**
- Input: (B, 6, 99, 99) observation
- Representation: (B, 6, 99, 99) → (B, 64, 25, 25) latent
- Dynamics: (B, 64, 25, 25) + action → FiLM → ConvGRU → (B, 64, 25, 25)
- Prediction: (B, 64, 25, 25) → policy/value

## Previous Updates (2025-08-13)

### GPU 성능 최적화 및 학습 가속화 (2025-08-13)
1. **텐서 메모리 풀링 시스템 구현**:
   - `utils.py`에 글로벌 텐서 풀 추가로 동적 메모리 할당 오버헤드 제거
   - `get_pooled_tensor()`: 재사용 가능한 텐서로 GPU 메모리 할당 비용 최소화
   - Pin Memory 활용으로 CPU-GPU DMA 전송 최적화
   - 효과: 58,806개 요소 텐서 재할당 비용 완전 제거

2. **Mixed Precision Training (AMP) 적용**:
   - `torch.cuda.amp.autocast` + `GradScaler` 도입
   - FP16 연산으로 GPU Tensor Cores 활용하여 2-4배 속도 향상
   - GPU 메모리 사용량 약 50% 절약 (11GB 중 1GB → 5-8GB 활용 가능)
   - Gradient Underflow 방지를 위한 Loss Scaling 자동화

3. **배치 처리 최적화**:
   - 개별 episode 순차 처리 → 배치 단위 병렬 처리로 변경
   - `build_observation_batch()`: 여러 state_dict를 배치로 한번에 처리
   - 네트워크 inference 배치화로 GPU 활용도 극대화
   - 배치 크기 32 → 128로 증가 (4배 처리량 향상)

4. **Gradient Accumulation 구현**:
   - 4-step accumulation으로 효과적 배치 크기 512 시뮬레이션
   - 메모리 제약 하에서 대용량 배치 학습 효과 달성
   - 안정적인 gradient로 학습 수렴성 개선

5. **비동기 데이터 전송 최적화**:
   - `non_blocking=True`로 CPU-GPU 전송과 연산 병렬화
   - Zero-Copy 메모리 복사로 데이터 이동 비용 최소화
   - `copy_()` 메서드로 in-place 연산 활용

6. **성능 개선 결과**:
   - GPU 활용률: 20-50% → 70-90% (목표)
   - GPU 메모리 활용도: 1GB/11GB → 6-8GB/11GB (목표)  
   - 학습 처리량: 기존 대비 3-5배 향상 (예상)
   - 에폭당 학습 시간 단축으로 전체 학습 효율성 대폭 개선

### Target Values 분리 및 학습 개선
1. **Target Values와 MCTS Values 분리**:
   - `episode.values`: MCTS root value (네트워크 예측값)
   - `episode.target_values`: n-step bootstrap 학습 타깃
   - train.py에서 target_values 우선 사용하도록 수정
   - 예측값과 타깃값 명확히 분리하여 학습 안정성 향상

2. **n-step Bootstrap 정석 구현**:
   - `_calculate_target_value()` 함수 완전 재작성
   - 터미널 상태 처리 개선 (dones 필드 활용)
   - 충돌 시 Episode 즉시 종료 및 부트스트랩 중단
   - Monte Carlo return과 TD learning 적절히 조합

3. **충돌 감지 및 Episode 종료**:
   - 충돌 발생 시 reward = -1.0 부여
   - Episode를 충돌 지점에서 즉시 잘라냄
   - dones 필드로 터미널 상태 명시적 관리
   - 불필요한 계산 방지 및 메모리 효율성 개선

### 시각화 개선
1. **학습 진행 상황 시각화 강화** (visualizations/*.png):
   - Target Rewards 표시: 총합, 평균, non-zero 개수
   - Target Values 표시: 평균, 최소/최대값
   - Target Policy 표시: 가장 높은 확률의 action
   - MCTS Values와 비교하여 학습 품질 평가 가능

2. **Episode 시각화 개선**:
   - target_values 우선 표시 (없으면 values)
   - 충돌 정보 및 경로 추종 정확도 표시

### 좌표계 일관성 확보
1. **좌표계 통일**:
   - 전방(상단) → row 증가 (아래 방향)
   - 우측 → col 증가
   - environment.py, data_parser.py, self_play.py 모두 일치

2. **Action 좌표 변환**:
   - `simulated_ego_row += dx` (전방/후방) ... 상방이 +
   - `simulated_ego_col += dy` (좌/우) ... 우측이 +
   - 모든 모듈에서 동일한 변환 규칙 적용

## Recent Updates (2025-08-12)

### Episode 평가 함수 구현 및 개선
1. **충돌 감지 함수 구현** (`_calculate_collision_reward`):
   - simulated_ego_pos 주변 3x3 그리드 영역에서 장애물 체크
   - Channel 0 (Dynamic objects)와 Channel 1 (Static environment) 모두 검사
   - 충돌 감지 시 즉시 -1.0 반환하여 최대 페널티 부여
   - 로봇 크기(60cm)를 고려한 3x3 그리드(60cm x 60cm) 충돌 판정

2. **경로 추종 평가 통합**:
   - `_calculate_trajectory_following_reward`만 사용하도록 단순화
   - Missing Rate 함수 제거 (중복 기능)
   - 평균 거리 오차를 연속적인 value로 변환 (0 거리→1.0, 10 거리→0.0, 20 거리→-1.0)

3. **최종 value 계산 로직**:
   - 충돌 발생 시 무조건 -1.0 반환
   - 충돌이 없으면 경로 추종 정확도를 최종 value로 사용
   - 명확한 로그 출력으로 각 평가 요소 확인 가능

4. **데이터 파싱 개선**:
   - `rosetta.py`에 `get_grid_value_from_ldm_obstacles_type()` 함수 추가
   - LDM obstacles_type을 그리드 값으로 변환하는 함수 통합
   - Objects와 LDM 데이터 모두 rosetta.py 함수로 일관되게 처리

### MCTS 및 네트워크 안정성 개선
1. **MCTS 버그 수정**:
   - `_recurrent_inference`에서 `hidden_state`를 올바른 device로 이동
   - 루트 노드 `visit_count=0`일 때 PUCT 탐색항이 0이 되는 문제 수정 (`max(parent.visit_count, 1)` 사용)
   - `_get_pi_targets`에서 분모가 0인 경우 처리 (균등 분포 반환)
   - softmax 계산 시 `dim=-1` 사용으로 일관성 개선
   - config의 `pb_c_init`, `pb_c_base` 값을 MCTS에 전달하도록 수정

2. **초기 reward 계산 개선**:
   - `models.py`의 `initial_inference`에서 `torch.log(one_hot)` 사용으로 인한 `-inf` 문제 해결
   - logits를 직접 생성: 모든 값을 -1e8로 초기화하고 중앙 인덱스(reward=0)만 0으로 설정
   - softmax 적용 시 안전하게 reward=0을 표현

3. **코드 수정 상세**:
   - `mcts.py`: hidden_state device 설정, PUCT 최소값 보장, policy 분모 0 방지
   - `models.py`: reward logits 생성 방식 개선으로 수치 안정성 확보
   - `self_play.py`: 충돌 감지 및 경로 추종 평가 함수 구현
   - `rosetta.py`: LDM 데이터 변환 함수 추가
   - `data_parser.py`: rosetta.py 함수 사용으로 코드 일관성 개선

## Recent Updates (2025-07-30)

### MuZero 네트워크 구조 개선
1. **Dynamics 함수 개선**:
   - 불필요한 min-max 정규화 제거 (latent space 일관성 문제 해결)
   - 배치별 정규화가 같은 상태를 다르게 표현하는 문제 수정
   - 명확한 docstring 추가: g(s_t, a_t) -> (r_t, s_{t+1})

2. **LayerNorm 전면 적용**:
   - `mlp()` 함수에 `use_layer_norm` 파라미터 추가
   - 모든 네트워크에 LayerNorm 적용:
     - representation_network
     - dynamics_latent_network
     - dynamics_reward_network
     - prediction_policy_network
     - prediction_value_network
   - 안정적인 gradient flow와 학습 개선

3. **Representation 함수 개선**:
   - latent state 정규화 제거
   - LayerNorm이 네트워크 내부에서 정규화 담당
   - 일관된 latent representation 보장

### 6채널 Observation 구조 및 Simulated Ego Position 추가
1. **Observation 구조 확장**:
   - observation_shape을 (5, 99, 99)에서 (6, 99, 99)로 확장
   - Channel 5: Simulated ego position 추가
   - Channel 5의 [5, 0, 0]에 row 좌표, [5, 0, 1]에 col 좌표 저장

2. **Simulated Ego Position 관리**:
   - GT 데이터: 항상 그리드 중앙 (49, 49)에서 시작
   - Self-play: MCTS가 선택한 action에 따라 누적 이동
   - 그리드 좌표 (row, col) 형태로 관리
   - environment.py와 self_play.py에서 일관되게 처리

3. **코드 수정 사항**:
   - `config.py`: observation_shape = (6, 99, 99)
   - `data_parser.py`: build_observation_tensor()에 Channel 5 처리 추가
   - `models.py`: representation network 입력 차원 58,810으로 수정
   - `expert_data_loader.py`: simulated_ego_pos 초기값 (49, 49) 설정
   - `self_play.py`: enrich_episodes_with_mcts()에서 action에 따른 위치 업데이트

### MCTS 코드 개선 및 게임 오버 Value 계산 구조화
1. **MCTS 버그 수정**:
   - 불필요한 action 재계산 코드 제거
   - parent-child 관계 재탐색 로직 삭제
   - 엣지 케이스 처리 개선

2. **게임 오버 Value 계산 함수화**:
   - `_calculate_trajectory_following_reward()`: 경로 추종 정확도 평가
   - `_calculate_collision_reward()`: 충돌 기반 평가 (TODO)
   - 향후 확장 가능한 구조로 설계

3. **Value Backward Propagation**:
   - Episode 마지막 스텝의 value로부터 역방향 전파
   - V[t] = r[t] + γ * V[t+1] 공식 사용
   - discount factor로 미래 가치 할인

## Recent Updates (2025-07-25)

### Direction Vector 고정 방식 적용
1. **방향 벡터 일관성 개선**:
   - Episode 첫 스텝의 direction vector를 전체 Episode에서 고정 사용
   - Ego가 회전해도 방향 정보의 일관성 유지
   - "미래의 큰 방향성"이라는 목적에 부합

2. **구현 상세**:
   - `expert_data_loader.py`: 첫 스텝에서 episode_near_direction, episode_final_direction 저장
   - 이후 스텝에서는 저장된 값 재사용
   - Episode 좌표계와 direction 좌표계의 일치성 확보

### state_dict에서 ego_pos 제거 및 구조 개선
1. **중복 데이터 제거**:
   - state_dict에서 'ego_pos' 필드 제거 (전역 좌표로 저장되어 좌표계 불일치)
   - GT ego 위치는 Episode의 'expert_ego_pos'로만 관리 (그리드 좌표)
   - state_dict는 순수하게 observation 데이터만 포함

2. **수정된 state_dict 구조**:
   - 'obstacles': Channel 0 동적 장애물 (그리드 좌표)
   - 'static_environment': Channel 1 정적 환경 (그리드 좌표)
   - 'ground_class_id': Channel 2 지면 semantic class ID (그리드 좌표)
   - 'near_destination_direction': 근거리 방향 벡터 (Episode 첫 스텝에서 고정)
   - 'final_destination_direction': 목적지 방향 벡터 (Episode 첫 스텝에서 고정)

3. **시각화 개선**:
   - visualize_muzero_episode.py에서 ego 위치는 expert_ego_pos 사용
   - Step Information에 그리드 좌표로 표시

### 방향 벡터 처리 개선
1. **방향 벡터 처리 통합**:
   - `db3_extractor.py`: 전역 좌표계 방향 벡터를 차량 좌표계로 변환하여 JSON 저장
   - `extract_waypoint_vectors()` 함수에 ego yaw 파라미터 추가 및 변환 로직 구현
   - `data_parser.py`, `visualize_muzero_episode.py`: 추가 변환 없이 원본 데이터 사용
   - `expert_data_loader.py`: 방향 벡터를 소수점 둘째 자리까지 반올림하여 저장

2. **수정된 파일**:
   - `data_parser.py`: grid_row = center + rel_x, grid_col = center - rel_y
   - `expert_data_loader.py`: 동일한 좌표 변환 적용, 방향 벡터 round(value, 2)
   - `visualize_muzero_episode.py`: 방향 벡터 표시 수정 (dx = -y*10, dy = x*10)
   - `db3_extractor.py`: extract_waypoint_vectors에 차량 좌표계 변환 추가

3. **UI/UX 개선**:
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
   - pi: MCTS가 계산한 방문횟수분포
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
   - 공용 함수: `get_object_color()`, `get_grid_value_from_object_class_id()`, `get_grid_value_info()`

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
