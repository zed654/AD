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
        # 네트워크 설정
        self.seed = 0
        self.max_num_gpus = None

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
        self.max_episode_step_length = 50  # 에피소드 최대 스텝 수 (약 10초, 5Hz 기준)
        self.collision_threshold = 0.5  # 충돌 판정 거리
        self.goal_threshold = 1.0  # 목표 도달 판정 거리
        self.cell_size = 0.2  # 각 그리드 셀의 크기 (m)
        
        # GT Episode 생성 설정
        self.gt_num_episodes = 50  # GT 데이터에서 생성할 에피소드 수 (20 → 50)
        self.gt_episode_overlap = 10 #25  # 에피소드 간 겹침 스텝 수
        
        # Observation Provider 설정
        self.observation_provider_type = "gt"  # "mock", "gt", "ros2" 중 선택
        self.gt_data_path = Path("data/db3_parser/results/IO_GT_matching_sensor_data_171626_95.jsonl")  # GT 데이터 경로
        self.use_obstacle_channels = True  # False: 방향 벡터만 사용, True: 장애물 채널도 사용

        # Self-Play 설정
        self.num_workers = 1
        self.selfplay_on_gpu = False
        self.num_simulations = 30  # 50 → 30 (밤새 더 많은 Episode 처리)
        self.discount = 0.997
        self.temperature_threshold = None
        self.num_stochastic_steps = 30  # 초기 30스텝 동안 확률적 행동 선택
        self.max_game_length = 500  # Self-play 안전장치 (무한루프 방지)
        self.num_selfplay_games = 5  # Self-play로 실행할 게임 수

        # PUCB 하이퍼 파라미터 설정
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # 네트워크 설정
        self.network = "fullyconnected"  # "resnet" 또는 "fullyconnected" 또는 "fullyconnected_test"
        self.support_size = 10
        
        # Fully Connected Network 설정
        self.encoding_size = 64  # 8 -> 64로 증가
        self.fc_representation_layers = [128, 64]  # 히든 레이어 추가
        self.fc_dynamics_layers = [64, 32]  # 16 -> 더 큰 네트워크
        self.fc_reward_layers = [32, 16]
        self.fc_value_layers = [32, 16]
        self.fc_policy_layers = [32, 16]

        # 학습 설정
        self.results_path = Path(__file__).resolve().parent / "results" / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        self.save_model = True
        self.training_steps = 1000  # 500 -> 1000 (밤새 충분한 학습)
        self.batch_size = 32  # 16 -> 32 (안정적인 학습)
        self.unroll_steps = 49  # K-step unrolling 길이 (전체 Episode 커버 - Monte Carlo에 가까운 학습)
        self.checkpoint_interval = 5  # 10 → 5로 변경 (5 에폭마다 시각화)
        self.value_loss_weight = 1
        self.train_on_gpu = torch.cuda.is_available()

        self.optimizer = "Adam"
        self.weight_decay = 1e-4
        self.momentum = 0.9

        # 학습률 설정
        self.lr_init = 0.001  # 0.02 -> 0.001로 감소
        self.lr_decay_rate = 0.9  # 0.8 -> 0.9로 더 천천히 감소
        self.lr_decay_steps = 100  # 1000 -> 100으로 더 자주 감소
        
        # 추가 학습 설정 (개선된 인터페이스용)
        self.num_training_epochs = 200  # 5 → 20 (밤새 학습)
        self.replay_buffer_size = 200  # 100 → 200 (더 많은 데이터 보관)
        
        # 체크포인트 저장 설정
        self.checkpoint_save_dir = "checkpoints"  # 체크포인트 저장 디렉토리 이름
        self.checkpoint_final_name = "muzero_final.pth"  # 최종 모델 파일명
        self.checkpoint_epoch_prefix = "muzero_epoch_"  # 에폭별 체크포인트 파일명 접두사
        
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
