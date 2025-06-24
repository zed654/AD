from pathlib import Path
import datetime
import torch

# MuZero 설정 클래스
class MuZeroConfig:
    def __init__(self):
        # 네트워크 설정
        self.seed = 0
        self.max_num_gpus = None

        # 게임 관련 설정
        self.observation_shape = (1, 1, 4)  # 게임 관찰의 차원 (channel, height, width)
        self.action_space = list(range(2))  # 가능한 모든 액션 목록
        self.players = list(range(1))  # 플레이어 목록
        self.stacked_observations = 0  # 현재 관찰에 추가할 이전 관찰 및 이전 액션 수

        # Self-Play 설정
        self.num_workers = 1
        self.selfplay_on_gpu = False
        self.max_moves = 500
        self.num_simulations = 50
        self.discount = 0.997
        self.temperature_threshold = None

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
        self.training_steps = 10000
        self.batch_size = 128
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
