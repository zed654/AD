# MuZero Observation Provider 사용법

## 개요
MuZero의 Observation을 다양한 소스에서 가져올 수 있도록 ObservationProvider 인터페이스를 설계했습니다.

## 아키텍처

### 1. ObservationProvider 인터페이스
```python
class ObservationProvider(ABC):
    @abstractmethod
    def reset(self) -> np.ndarray
    @abstractmethod
    def get_observation(self, action) -> Tuple[np.ndarray, float, bool, Dict]
    @abstractmethod
    def close(self)
```

### 2. 구현체들

#### MockObservationProvider (기본값)
- 테스트용 임시 랜덤 데이터 제공
- 현재 기본으로 사용중

#### GTObservationProvider (학습용)
- `data/db3_parser/results/matched_data.jsonl`에서 GT 데이터 로드
- 실제 주행 데이터와 미래 경로(GT) 정보 포함
- **TODO**: 실제 관측값 추출 로직 구현 필요

#### ROS2ObservationProvider (실행용)
- ROS2 Topic에서 실시간 데이터 수신
- Topics:
  - `/apollo/agent_0/detected_object_array` (장애물)
  - `/carla/ego_vehicle/odometry` (자차 상태)
  - `/carla/ego_vehicle/rgb_front/image` (카메라)
- **TODO**: ROS2 초기화 및 콜백 구현 필요

## 사용 방법

### 1. 설정 파일에서 Provider 선택
```python
# models/MuZero/config.py
self.observation_provider_type = "mock"  # "mock", "gt", "ros2" 중 선택
self.gt_data_path = Path("data/db3_parser/results/matched_data.jsonl")
```

### 2. 학습 모드 (GT 데이터 사용)
```bash
# config.py에서 observation_provider_type을 "gt"로 변경 후
python muzero.py
```

### 3. 실행 모드 (ROS2 사용)
```bash
# config.py에서 observation_provider_type을 "ros2"로 변경 후
python muzero.py
```

## 구현 필요 사항

### GTObservationProvider
1. `_extract_observation()`: obstacle tracking 데이터를 observation_shape로 변환
2. `_calculate_reward()`: 실제 보상 함수 구현 (충돌 회피, 목표 도달 등)
3. GT trajectory 매칭 로직 개선

### ROS2ObservationProvider  
1. ROS2 노드 초기화
2. Topic 구독자 생성 및 콜백 구현
3. `_create_observation_from_ros2()`: ROS2 메시지를 observation으로 변환

## 현재 상태
- 아키텍처 설계 완료 ✓
- 인터페이스 및 기본 구조 구현 ✓
- Mock provider로 기존 기능 유지 ✓
- GT/ROS2 provider는 TODO로 표시된 부분 구현 필요