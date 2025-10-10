"""
MuZero 설정 파일
하이퍼파라미터와 학습 설정 관리
"""
from pathlib import Path
import datetime
import torch
from models.MuZero.action import Action

# MuZero 설정 클래스
class MuZeroConfig:
    def __init__(self):
        self.test_mode = 0  # 0: 학습모드, 1: 빠른테스트모드(빠른학습), 2: 수렴테스트모드(Episode 1개 수렴여부 확인)

        # 네트워크 설정
        self.seed = 0
        self.max_num_gpus = None
        
        # 새로운 Representation 구조 설정
        self.use_coordconv = False  # CoordConv 사용 여부 (기본 OFF)
        self.use_goal_map = False   # Goal을 2D 가우시안 히트맵으로 투영 여부 (기본 OFF) 
        self.use_film = False       # FiLM 채널 단위 주입 실험용 (기본 OFF)
        self.alpha_init = 0.5       # DirectionalGating alpha 초기값 (0.3~0.7 권장)

        # 게임 관련 설정
        self.action_space = list(range(Action.get_action_space_size()))  # Position 9개 액션 (POS0~POS8)
        self.players = list(range(1))  # 플레이어 목록
        self.stacked_observations = 0  # 현재 관찰에 추가할 이전 관찰 및 이전 액션 수

        # 환경 설정 (DrivingEnvironment에서 사용)
        # 원본: 200x200 (10cm 셀) -> 99x99 (20cm 셀)로 다운샘플링
        self.observation_shape = (6, 99, 99)  # 게임 관찰의 차원 (channel, height, width) - 19.8m x 19.8m 영역 (20cm 셀)
        # Channel 0: Dynamic objects (vehicles, pedestrians)
        # Channel 1: Static info (Cliff, Static obstacles, Missing points - LDM obstacles_type=1,2,3)
        # Channel 2: Class ID info (raw class_id values for semantic information)
        # Channel 3: Near destination direction (x, y in cells [0,0] and [0,1])
        # Channel 4: Final destination direction (x, y in cells [0,0] and [0,1])
        # Channel 5: Simulated ego position (1.0 at ego location)
        self.original_resolution = 0.1  # 원본 데이터 해상도 (10cm)
        self.downsample_factor = 2  # 다운샘플링 비율 (2x2 셀을 1개로)
        self.collision_threshold = 0.5  # 충돌 판정 거리
        self.collision_penalty = -2.0  # 충돌 시 패널티 (reward 값)
        self.goal_threshold = 1.0  # 목표 도달 판정 거리
        self.cell_size = 0.2  # 각 그리드 셀의 크기 (m)
        
        # PUCB 하이퍼 파라미터 설정
        self.pb_c_base = 19652
        self.pb_c_init = 2.5 # 높을수록 새로운 가지 탐색이 늘어남. Deafult: 1.25

        # 네트워크 설정
        self.network = "fullyconnected"  # "resnet" 또는 "fullyconnected" 또는 "fullyconnected_test"

        # MuZero Categorical Distribution Value/Reward 설정
        # 논문: 범위와 해상도를 독립적으로 설정
        # Reward 와 Value 모두 적용됨
        # Reward 실제 범위: -2.0 ~ +0.1 → h_transform 후 -0.73 ~ +0.05
        # Value는 누적되므로 더 넓은 범위 필요
        self.categorical_distribution_min = -2.0    # 최소 값
        self.categorical_distribution_max = 2.0     # 최대 값
        self.num_categorical_distribution_atoms = 21      # 카테고리 개수 (홀수 권장; 해상도임)
        # -2.0~2.0 에 atoms 21 이면
        #     [2.0, -1.8, -1.6, -1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]


        # Fully Connected Network 설정
        self.encoding_size = 64  # 8 -> 64로 증가
        self.fc_representation_layers = [128, 64]  # 히든 레이어 추가
        self.fc_dynamics_layers = [64, 32]  # 16 -> 더 큰 네트워크
        self.fc_reward_layers = [32, 16]
        self.fc_value_layers = [32, 16]
        self.fc_policy_layers = [32, 16]

        # Observation Provider 설정
        self.observation_provider_type = "gt"  # "mock", "gt", "ros2" 중 선택
        # GT 데이터 경로: 파일(.jsonl) 또는 폴더 경로 모두 지원
        if self.test_mode == 1 or self.test_mode == 2:
            self.gt_data_path = Path("data/db3_parser/results/IO_GT_matching_sensor_data_171626_95.jsonl")  # 단일 파일 ... 정상 주행 데이터 (코너 상황 대표 샘플)
        elif self.test_mode == 0:
            self.gt_data_path = Path("data/db3_parser/results/")  # 폴더 경로로 변경하면 모든 JSONL 파일 로드
        # self.gt_data_path = Path("data/db3_parser/results/IO_GT_matching_sensor_data_171626_95.jsonl")  # 단일 파일 ... 정상 주행 데이터 (코너 상황 대표 샘플)
        # self.gt_data_path = Path("data/db3_parser/results/IO_GT_matching_sensor_data_171226_91.jsonl")  # 단일 파일 ... Episode 5 에서 Near Destination 조금 이상함
        # self.gt_data_path = Path("data/db3_parser/results/IO_GT_matching_sensor_data_170026_79.jsonl")  # 단일 파일 ... 주행 전 대기상태 데이터임
        # self.gt_data_path = Path("data/db3_parser/results/IO_GT_matching_sensor_data_170226_81.jsonl")  # 단일 파일 ... 주행 전 대기상태 데이터임
        # self.gt_data_path = Path("data/db3_parser/results/IO_GT_matching_sensor_data_170326_82.jsonl")  # 단일 파일 ... 정상 주행 데이터. 보행자에 가려 멈추는 씬 많음
        # self.gt_data_path = Path("data/db3_parser/results/IO_GT_matching_sensor_data_170426_83.jsonl")  # 단일 파일 ... 정상 주행 데이터
        # self.gt_data_path = Path("data/db3_parser/results/IO_GT_matching_sensor_data_170626_85.jsonl")  # 단일 파일 ... 정상 주행 데이터
        self.use_obstacle_channels = True  # False: 방향 벡터만 사용, True: 장애물 채널도 사용

        # GT Episode 생성 설정
        # josnl 의 줄 개수 < num_episodes * (max_episode_step_length - gt_episode_overlap) 이어야 함
        if self.test_mode == 1:
            self.gt_num_episodes = 3
        elif self.test_mode == 2:
            self.gt_num_episodes = 1  # Episode 1개로 수렴여부 확인용
        elif self.test_mode == 0:
            self.gt_num_episodes = 999999 # 모든데이터 사용하려면 999999 로 설정  # GT 데이터에서 생성할 에피소드 수 (jsonl 파일 여러개면 더 많아도 됨. jsonl 이 현재 7700개 정도의 Steps 임. Episode * (Steps - Overlap Steps) 개수만큼 커버 가능)
        self.max_episode_step_length = 25  # 에피소드 별 최대 스텝 수 (약 10초, 5Hz 기준). 너무 클 필요 없음. 1 스텝 최대이동이 0.2m 임. 50스텝=10m 이동
        self.gt_episode_overlap = 10 # 10 . 25  # 에피소드 간 겹침 스텝 수
        self.replay_buffer_size = 3300 # 3300  # Replay Buffer 최대 사이즈. 너무 크게하면 과거 데이터의 업데이트가 늦어질 수 있음.
        
        # Self-Play 설정
        # TODO: 실제 Inference 할 때, MCTS Sim 이후 Step 선택하는 Action 부에 Temperature 를 0으로 줘야함
        self.num_workers = 1
        self.selfplay_on_gpu = False
        self.num_simulations = 50  # MCTS 시뮬레이션 횟수 (더 다양한 action 탐색)
        self.value_bootstrap_discount = 0.97
        self.mcts_value_backup_discount = 0.99
        self.puct_discount = 0.997
        self.temperature_threshold = None
        self.mcts_dirichlet_alpha = 1.2  # dirichlet noise - 탐험도를 높히기 위해 값을 키워야함 (a = 1 이면 정규분포임. 현 코드에서 얘를 기반으로 랜덤확률로 씀. 즉, 탐험항 높히려면 가치 낮은애가 커질 수 있어야함. 그렇기 때문에 alpha 값은 작을수록 탐험도가 강해짐). alpha 가 작으면 값이 큰 인덱스(prior 높은애) 의 변화량이 더 커짐
        self.mcts_exploration_fraction = 0.25  # dirichlet noise (클 수록 탐험도 커짐) -> 기존 prior 를 얼마나 깎고, noise 로 대체할건지 결정하는 항임. 값이 크면 prior 덜 반영됨 (0~1 사이). 1이면, prior 결과 안쓰고 noise 만 쓴다는거임
        self.num_stochastic_steps = 30  # [지금 얘 안씀] 초기 30스텝 동안 확률적 행동 선택
        self.max_game_length = 500  # Self-play 안전장치 (무한루프 방지)
        self.num_selfplay_games = 5  # Self-play로 실행할 게임 수

        # 학습 설정
        self.results_path = Path(__file__).resolve().parent / "results" / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        self.save_model = True
        if self.test_mode == 1:
            self.num_iterations = 10
            self.training_steps = 10
        elif self.test_mode == 2:
            self.num_iterations = 1000 # 100 {Self-Play - 학습} 사이클 반복 횟수
            self.training_steps = 1000
        elif self.test_mode == 0:
            self.num_iterations = 100 # 100 {Self-Play - 학습} 사이클 반복 횟수
            self.training_steps = 1000  # 1000. K-Step Unrolling 반복 횟수
        self.unroll_steps = 5  # K-step unrolling 길이 (Episode 의 Step 사이즈보다 작게하는게 좋음. 왜냐하면, 길어질수록 신뢰도 떨어져서. [전체 Episode 커버 - Monte Carlo에 가까운 학습])
        self.n_target_value_bootstrap = 4 # Target Value 생성 시 n-스텝 부트스트랩 길이. (인덱스 0~n-1 까지 Reward 쓰고, n 번째에서 Value 씀. 총 n+1 개임)
        self.batch_size = 128  # 32 -> 128 (GPU 메모리 활용도 최대화)
        self.value_loss_weight = 1
        self.train_on_gpu = torch.cuda.is_available()
        
        # 학습 중 시각화 설정
        self.enable_training_visualization = True  # 학습 중 시각화 생성 여부
        if self.test_mode == 1:
            self.visualization_interval_step = 5  # 테스트 모드에서는 더 자주 시각화
        elif self.test_mode == 0 or self.test_mode == 2:
            self.visualization_interval_step = 50  # training_steps 몇 스텝마다 PNG 시각화 할지
        self.visualization_num_episodes = 3  # PNG 에 시각화할 Episode 수 (최대)
        # self.visualization_min_episode_length = 10  # 시각화할 Episode 의 Step 최소 길이
        self.visualization_dpi = 150  # 이미지 저장 DPI
        self.visualization_figsize = (10, 10)  # 그림 크기 (width, height)

        # pth 체크포인트 저장 설정
        if self.test_mode == 1:
            self.pth_checkpoint_interval_step = 5
        elif self.test_mode == 0 or self.test_mode == 2:
            self.pth_checkpoint_interval_step = 50  # training_steps 몇 스텝마다 체크포인트 저장할지
        self.checkpoint_save_dir = "checkpoints"  # 체크포인트 저장 디렉토리 이름
        self.checkpoint_final_name = "muzero_final.pth"  # 최종 모델 파일명
        self.checkpoint_training_steps_prefix = "muzero_training_steps_"  # 에폭별 체크포인트 파일명 접두사

        # 체크포인트 재개 설정
        self.resume_checkpoint_path = None  # 재개할 체크포인트 경로 (None이면 새로 시작)
        # self.resume_checkpoint_path = "/home/chp/ai_ws/AD/dl-project/models/MuZero/results/2025-09-19--18-34-07/checkpoints/muzero_epoch_1000.pth"
        # "/home/chp/ai_ws/AD/dl-project/models/MuZero/results/2025-09-02--10-03-53/checkpoints/muzero_epoch_15.pth"
        # 예: "models/MuZero/results/2025-09-02--00-11-57/checkpoints/muzero_epoch_5.pth"
        
        # 성능 최적화 설정
        self.use_mixed_precision = True  # Mixed Precision Training (AMP) 활성화
        self.gradient_accumulation_steps = 4  # Gradient Accumulation으로 대용량 배치 시뮬레이션
        self.pin_memory = True  # DataLoader pin_memory 설정
        self.num_data_workers = 4  # CPU 병렬 처리 워커 수

        self.optimizer = "Adam"
        self.weight_decay = 1e-4
        self.momentum = 0.9

        # 학습률 설정 (프로덕션 값으로 복원)
        self.lr_init = 0.001  # 기존값 0.02  
        self.lr_decay_steps = 500  # 5000 스텝마다
        self.lr_decay_rate = 0.95   # 5% 씩 감소
        
        
        # 평가 설정
        self.eval_num_episodes = 100  # 평가할 에피소드 수
        self.eval_use_mcts = True  # 평가 시 MCTS 사용 여부
        self.eval_dataset_type = "validation"  # 평가 데이터셋 ("train", "validation", "test")
        self.eval_save_results = False  # 평가 결과 저장 여부

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        행동 선택이 학습이 진행됨에 따라 더 탐욕적이 되도록 방문 횟수 분포를 변경하는 매개변수.
        값이 작을수록 가장 좋은 행동(즉, 방문 횟수가 가장 높은)이 선택될 가능성이 높아집니다.

        Returns:
            양수 실수.
        """
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25
