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
        self.action_space = list(range(Action.get_action_space_size()))  # Position(8) x StopAndGo(2) = 16개 액션
        self.players = list(range(1))  # 플레이어 목록
        self.stacked_observations = 0  # 현재 관찰에 추가할 이전 관찰 및 이전 액션 수

        # 환경 설정 (DrivingEnvironment에서 사용)
        self.observation_shape = (1, 1, 4)  # 게임 관찰의 차원 (channel, height, width)
        self.env_max_steps = 10  # 에피소드 최대 스텝 수
        self.collision_threshold = 0.5  # 충돌 판정 거리
        self.goal_threshold = 1.0  # 목표 도달 판정 거리
        
        # Observation Provider 설정
        self.observation_provider_type = "mock"  # "mock", "gt", "ros2" 중 선택
        self.gt_data_path = Path("data/db3_parser/results/matched_data.jsonl")  # GT 데이터 경로

        # Self-Play 설정
        self.num_workers = 1
        self.selfplay_on_gpu = False
        self.num_simulations = 10 #50
        self.discount = 0.997
        self.temperature_threshold = None
        self.num_stochastic_steps = 30  # 초기 30스텝 동안 확률적 행동 선택
        self.max_game_length = 500  # Self-play 안전장치 (무한루프 방지)
        self.num_selfplay_games = 5  # Self-play로 실행할 게임 수

        # UCB 공식
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # 네트워크 설정
        self.network = "fullyconnected"  # "resnet" 또는 "fullyconnected"
        self.support_size = 10
        
        # Fully Connected Network 설정
        self.encoding_size = 8
        self.fc_representation_layers = []
        self.fc_dynamics_layers = [16]
        self.fc_reward_layers = [16]
        self.fc_value_layers = [16]
        self.fc_policy_layers = [16]

        # 학습 설정
        self.results_path = Path(__file__).resolve().parent / "results" / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        self.save_model = True
        self.training_steps = 30 #10000
        self.batch_size = 8 #128
        self.checkpoint_interval = 10
        self.value_loss_weight = 1
        self.train_on_gpu = torch.cuda.is_available()

        self.optimizer = "Adam"
        self.weight_decay = 1e-4
        self.momentum = 0.9

        # 학습률 설정
        self.lr_init = 0.02
        self.lr_decay_rate = 0.8
        self.lr_decay_steps = 1000
        
        # 추가 학습 설정 (개선된 인터페이스용)
        self.num_training_epochs = 10 # 100  # 전체 학습 에폭 수
        self.replay_buffer_size = 100  # Replay Buffer 크기
        
        # 체크포인트 저장 설정
        self.checkpoint_save_dir = "checkpoints"  # 체크포인트 저장 디렉토리 이름
        self.checkpoint_final_name = "muzero_final.pth"  # 최종 모델 파일명
        self.checkpoint_epoch_prefix = "muzero_epoch_"  # 에폭별 체크포인트 파일명 접두사

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
