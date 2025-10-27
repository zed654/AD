"""
MuZero Neural Network Models
실제 신경망 구현체들
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import numpy as np
from .network import AbstractNetwork
from enum import Enum

# 임시로 AgentType 정의 (추후 사용하지 않으면 제거)
class AgentTypeNeuralNetwork(Enum):
    EGO = "ego"
    
class AgentTypeKinematic(Enum):
    PEDESTRIAN = "pedestrian"
    VEHICLE = "vehicle"


def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict


def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=torch.nn.Identity,
    activation=torch.nn.ELU,
    use_layer_norm=False,
):
    """Multi-Layer Perceptron 생성
    
    Args:
        input_size: 입력 크기
        layer_sizes: 은닉층 크기 리스트
        output_size: 출력 크기
        output_activation: 출력 활성화 함수
        activation: 은닉층 활성화 함수
        use_layer_norm: LayerNorm 사용 여부
    """
    layers = []
    for i, layer_size in enumerate(layer_sizes):
        layers.append(torch.nn.Linear(input_size, layer_size))
        if use_layer_norm:
            layers.append(torch.nn.LayerNorm(layer_size))
        layers.append(activation())
        input_size = layer_size

    layers.append(torch.nn.Linear(input_size, output_size))
    layers.append(output_activation())
    return torch.nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    """ResNet Residual Block with GroupNorm and ReLU"""

    def __init__(self, channels, num_groups=32):
        """
        Args:
            channels: 입출력 채널 수 (동일)
            num_groups: GroupNorm의 그룹 수 (기본 32)
        """
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups, channels)

        # Residual connection의 최종 ReLU
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Args:
            x: (B, channels, H, W)
        Returns:
            out: (B, channels, H, W)
        """
        identity = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.gn2(out)

        # Residual connection
        out += identity
        out = self.relu2(out)

        return out


def compute_edt(occupancy_grid):
    """
    Occupancy grid에서 Euclidean Distance Transform 계산

    Args:
        occupancy_grid: (B, 1, H, W) 점유 그리드 (0/1 값)

    Returns:
        edt: (B, 1, H, W) EDT 결과 (0~1 정규화)

    Note:
        현재 구현은 CPU 기반 scipy 사용으로 GPU 병목 가능성 있음
        향후 GPU 기반 EDT 구현 고려 필요
    """
    batch_size, _, height, width = occupancy_grid.shape
    device = occupancy_grid.device

    edt_result = torch.zeros_like(occupancy_grid)

    # 성능 최적화: 전체 배치를 CPU로 이동 (반복적 이동 방지)
    occupancy_cpu = occupancy_grid.cpu().numpy()

    for b in range(batch_size):
        # 점유된 셀(1)을 배경(0)으로, 빈 셀(0)을 전경(1)으로 변환
        binary_mask = (occupancy_cpu[b, 0] == 0).astype(np.float32)

        # EDT 계산 (빈 공간에서 가장 가까운 장애물까지의 거리)
        edt = distance_transform_edt(binary_mask)

        # 정규화: 최대값이 1이 되도록
        if edt.max() > 0:
            edt = edt / edt.max()

        edt_result[b, 0] = torch.tensor(edt, device=device, dtype=torch.float32)

    return edt_result


class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise Separable Convolution for efficient CNN"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        # GroupNorm: 8 groups (out_channels should be divisible by 8)
        num_groups = min(8, out_channels) if out_channels >= 8 else out_channels
        self.gn = nn.GroupNorm(num_groups, out_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.gn(x)
        x = self.relu(x)
        return x


class CNNLiteBackbone(nn.Module):
    """CNN-lite backbone for efficient processing"""
    
    def __init__(self, in_channels, use_coordconv=False):
        super().__init__()
        self.use_coordconv = use_coordconv
        
        # CoordConv 사용 시 x, y 좌표 채널 2개 추가
        if use_coordconv:
            in_channels += 2
        
        # kernel_size 는 명시되지 않으면 기본값 3
        # Stem: 99x99 -> 99x99
        self.stem = DepthwiseSeparableConv2d(in_channels, 32, stride=1)
        
        # Block0: 99x99 -> 99x99 (stride=1) ... Depth 늘려 수용력 향상
        # 참고) 해상도? 높히려면 dilation 을 1로 해야함. 다만 Receptive Field 가 줄어듦. 따라서, 보완을 위해 Depth 늘려야함 야Ga
        self.block0_0 = DepthwiseSeparableConv2d(32, 32, stride=1, padding=2, dilation=2)  # 조밀한 정보
        self.block0_1 = DepthwiseSeparableConv2d(32, 32, stride=1, padding=2, dilation=2)  # 넓은 수용 영역
        self.block0_2 = DepthwiseSeparableConv2d(32, 32, stride=1, padding=2, dilation=2)  # 넓은 수용 영역
        
        # Block1: 99x99 -> 50x50 (stride=2로 축소)
        self.block1_0 = DepthwiseSeparableConv2d(32, 48, stride=2, padding=1)
        self.block1_1 = DepthwiseSeparableConv2d(48, 48, stride=1, padding=2, dilation=2)
        
        # Block2: 50x50 -> 25x25 (stride=2로 축소)
        self.block2 = DepthwiseSeparableConv2d(48, 64, stride=2, padding=1)
        
        # 1. Projection: 64 channels -> 256 channels
        self.dynamics_resnet_projection = nn.Sequential(
            nn.Conv2d(32, 256, kernel_size=1, bias=False),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True)
        )

        # 2. ResNet Blocks: 16 blocks with 256 channels
        self.dynamics_resnet_blocks = nn.Sequential(
            *[ResidualBlock(256, num_groups=32) for _ in range(16)]
        )

        # 3. Back-projection: 256 channels -> 64 channels
        self.dynamics_resnet_backprojection = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True)
        )

        # 좌표 그리드 생성 (CoordConv용)
        self.register_buffer('coord_grid', None)
    
    def _get_coord_grid(self, batch_size, height, width, device):
        """CoordConv용 좌표 그리드 생성"""
        if self.coord_grid is None or self.coord_grid.shape[-2:] != (height, width):
            x_coords = torch.linspace(-1, 1, width, device=device).view(1, 1, 1, width).expand(1, 1, height, width)
            y_coords = torch.linspace(-1, 1, height, device=device).view(1, 1, height, 1).expand(1, 1, height, width)
            self.coord_grid = torch.cat([x_coords, y_coords], dim=1)  # (1, 2, H, W)
        
        return self.coord_grid.expand(batch_size, -1, -1, -1)
    
    def forward(self, x):
        batch_size, _, height, width = x.shape

        # CoordConv 적용
        if self.use_coordconv:
            coord_grid = self._get_coord_grid(batch_size, height, width, x.device)
            x = torch.cat([x, coord_grid], dim=1)

        x = self.stem(x)         # 99x99 -> 99x99
        x = self.block0_0(x)     # 99x99 -> 99x99 (dilation=1)
        x = self.block0_1(x)     # 99x99 -> 99x99 (dilation=2)
        x = self.block0_2(x)     # 99x99 -> 99x99 (dilation=2)
        x = self.block1_0(x)     # 99x99 -> 50x50 (stride=2, dilation=2)
        x = self.block1_1(x)     # 50x50 -> 50x50 (dilation=2) ← 여기 문제!
        x = self.block2(x)       # 50x50 -> 25x25

        # x = self.stem(x)         # 99x99 -> 99x99
        # x = self.dynamics_resnet_projection(x)      # 99x99 -> 99x99 (dilation=1)
        # x = self.dynamics_resnet_blocks(x)          # 99x99 -> 99x99 (dilation=2)
        # x = self.dynamics_resnet_backprojection(x)  # 99x99 -> 99x99 (dilation=2)

        return x


class FiLM(nn.Module):
    """Feature-wise Linear Modulation for Action Conditioning with Stability"""
    
    def __init__(self, action_dim, feature_dim, hidden_dim=64, scale_factor=0.1):
        super().__init__()
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * feature_dim)  # gamma와 beta
        )
        self.feature_dim = feature_dim
        self.scale_factor = scale_factor  # FiLM 효과 크기 조절
        
        # 안정화를 위한 작은 초기값 설정
        with torch.no_grad():
            # gamma 초기값: 작은 랜덤값 (평균 0, 표준편차 0.02)
            self.action_mlp[-1].weight[:feature_dim] *= 0.02
            self.action_mlp[-1].bias[:feature_dim].fill_(0.0)  # gamma 초기값 0
            
            # beta 초기값: 작은 랜덤값 (평균 0, 표준편차 0.01)  
            self.action_mlp[-1].weight[feature_dim:] *= 0.01
            self.action_mlp[-1].bias[feature_dim:].fill_(0.0)  # beta 초기값 0
    
    def forward(self, features, action_one_hot):
        """
        Args:
            features: (B, feature_dim, H, W) feature maps
            action_one_hot: (B, action_dim) one-hot action vector
        Returns:
            modulated features: (B, feature_dim, H, W)
        """
        B = features.shape[0]
        
        # Feature 정규화 (코사인 정규화 유사)
        features_norm = F.normalize(features, dim=1, p=2)  # L2 normalize along channel dim
        
        # Action → FiLM parameters
        film_params = self.action_mlp(action_one_hot)  # (B, 2 * feature_dim)
        gamma, beta = film_params.chunk(2, dim=1)      # (B, feature_dim) each
        
        # 안정화된 스케일링
        gamma = torch.tanh(gamma) * self.scale_factor  # Bound gamma to [-scale_factor, +scale_factor]
        beta = torch.tanh(beta) * self.scale_factor    # Bound beta to [-scale_factor, +scale_factor]
        
        # Reshape for broadcasting
        gamma = gamma.view(B, self.feature_dim, 1, 1)  # (B, feature_dim, 1, 1)
        beta = beta.view(B, self.feature_dim, 1, 1)    # (B, feature_dim, 1, 1)
        
        # 비감쇠 보장: (1 + k·gamma) 형태
        return features_norm * (1 + gamma) + beta


class ConvGRU(nn.Module):
    """Convolutional GRU for spatial-temporal consistency"""
    
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Gates: reset and update (3층으로 깊게)
        # 입력: (B, input_dim + hidden_dim, H, W) = (B, 128, 25, 25)
        # 출력: (B, 2 * hidden_dim, H, W) = (B, 128, 25, 25)
        self.conv_gates = nn.Sequential(
            nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 2 * hidden_dim, kernel_size, padding=kernel_size//2)
        )

        # Candidate hidden state (3층으로 깊게)
        # 입력: (B, input_dim + hidden_dim, H, W) = (B, 128, 25, 25)
        # 출력: (B, hidden_dim, H, W) = (B, 64, 25, 25)
        self.conv_candidate = nn.Sequential(
            nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        )
    
    def forward(self, x, h):
        """
        Args:
            x: (B, input_dim, H, W) input features
            h: (B, hidden_dim, H, W) previous hidden state
        Returns:
            new hidden state with residual: (B, hidden_dim, H, W)
        """
        combined = torch.cat([x, h], dim=1)
        
        # Compute gates
        gates = torch.sigmoid(self.conv_gates(combined))
        reset_gate, update_gate = gates.chunk(2, dim=1)
        
        # Candidate hidden state
        reset_combined = torch.cat([x, reset_gate * h], dim=1)
        candidate = torch.tanh(self.conv_candidate(reset_combined))
        
        # Update hidden state
        h_new = (1 - update_gate) * h + update_gate * candidate
        
        # Residual connection: h_new + h (드리프트/소실 완화)
        return h_new + h


class DirectionalGating(nn.Module):
    """Goal 기반 방향 내적 게이팅 - 근접/최종 목적지 독립 처리"""

    def __init__(self, c=64, hidden=64, alpha_init=0.5):
        super().__init__()
        # 근접 목적지 (g_near)와 최종 목적지 (g_final)를 독립적으로 처리
        self.near_mlp = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, c)
        )
        self.final_mlp = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, c)
        )
        # 각각 독립적인 alpha 파라미터
        self.alpha_near_raw = nn.Parameter(torch.tensor(alpha_init).log())
        self.alpha_final_raw = nn.Parameter(torch.tensor(alpha_init).log())
        self.softplus = nn.Softplus()
        self.c = c

    def forward(self, z, g):
        """
        Args:
            z: [B, C, H, W] feature maps
            g: [B, 4] goal vector (near_direction[2] + final_direction[2])
        Returns:
            gated feature maps [B, C, H, W]
        """
        B, C, H, W = z.shape

        # 목적지 방향 벡터 분리
        g_near = g[:, :2]   # [B, 2] - 근접 목적지 방향
        g_final = g[:, 2:]  # [B, 2] - 최종 목적지 방향

        # 각 목적지에 대한 embedding
        e_near = self.near_mlp(g_near) / (self.c ** 0.5)   # [B, C]
        e_final = self.final_mlp(g_final) / (self.c ** 0.5)  # [B, C]

        # 각각 독립적인 내적 계산
        p_near = z * e_near.view(B, C, 1, 1)   # [B, C, H, W]
        p_final = z * e_final.view(B, C, 1, 1)  # [B, C, H, W]

        s_near = p_near.sum(dim=1)   # [B, H, W]
        s_final = p_final.sum(dim=1)  # [B, H, W]

        # 독립적인 시그모이드 마스크
        m_near = torch.sigmoid(s_near)   # [B, H, W]
        m_final = torch.sigmoid(s_final)  # [B, H, W]

        # 독립적인 alpha 값
        alpha_near = self.softplus(self.alpha_near_raw)
        alpha_final = self.softplus(self.alpha_final_raw)

        # 비감쇠 게이팅: 두 목적지의 영향을 독립적으로 더함
        return z * (1 + alpha_near * m_near.unsqueeze(1) + alpha_final * m_final.unsqueeze(1))  # [B, C, H, W]


class MuZeroFullyConnectedNetwork(AbstractNetwork):
    """Repulsion/Attraction 분기 + Directional Gating을 사용하는 MuZero Network"""
    
    def __init__(
        self,
        observation_shape,
        stacked_observations,  # 사용하지 않지만 호환성 위해 유지
        action_space_size,
        encoding_size,  # 사용하지 않음 (2D 공간 구조 고정)
        fc_reward_layers,
        fc_value_layers,
        fc_policy_layers,
        fc_representation_layers,  # 사용하지 않음
        fc_dynamics_layers,
        num_categorical_distribution_atoms,
        # 새로운 설정들
        use_coordconv=False,
        use_goal_map=False,
        use_film=False,
        alpha_init=0.5,
        use_resnet_dynamics=False,
    ):
        super().__init__()
        self.action_space_size = action_space_size
        # num_categorical_distribution_atoms 직접 사용
        self.num_categorical_distribution_atoms = num_categorical_distribution_atoms
        self.observation_shape = observation_shape

        # 새로운 구조 설정
        self.use_coordconv = use_coordconv
        self.use_goal_map = use_goal_map
        self.use_film = use_film
        self.use_resnet_dynamics = use_resnet_dynamics
        
        # 2D latent 구조 (고정)
        self.latent_channels = 64
        self.latent_height = 25
        self.latent_width = 25
        
        # Repulsion 분기 (장애물/정적환경): Channel 0, 1 + EDT
        repulsion_channels = 1  # obstacles + static_environment -> occupancy_grid + edt
        self.repulsion_encoder = CNNLiteBackbone(repulsion_channels, use_coordconv)
        
        # Attraction 분기 (지면/주행가능): Channel 2 (ground_class_id를 원핫으로 변환)
        # 간단히 하기 위해 class_id를 정규화해서 사용 (원핫 변환은 추후 필요시 구현)
        attraction_channels = 1  # ground_class_id (정규화)
        self.attraction_encoder = CNNLiteBackbone(attraction_channels, use_coordconv)
        
        # Fusion: 128 -> 64 채널로 축소
        self.fuse_1x1 = nn.Conv2d(128, 64, kernel_size=1)
        
        # Directional Gating
        self.dir_gate = DirectionalGating(c=64, alpha_init=alpha_init)
        
        # FiLM for Action Conditioning
        self.action_film = FiLM(
            action_dim=self.action_space_size,
            feature_dim=self.latent_channels,
            hidden_dim=64
        )
        
        # ConvGRU for Temporal Consistency
        self.dynamics_gru = ConvGRU(
            input_dim=self.latent_channels,  # FiLM 적용된 latent
            hidden_dim=self.latent_channels,  # 64
            kernel_size=3
        )

        # Final refinement conv (optional)
        self.dynamics_refine = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),  # 8 groups for 64 channels
            nn.ReLU6(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),  # 8 groups for 64 channels
            nn.ReLU6(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),  # 8 groups for 64 channels
            nn.ReLU6()
        )

        # ========== ResNet Alternative for Dynamics (16 blocks, 256 channels) ==========
        # Dynamics ResNet: 64 -> 256 -> 64 with 16 residual blocks
        # 1. Projection: 64 channels -> 256 channels
        self.dynamics_resnet_projection = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, bias=False),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True)
        )

        # 2. ResNet Blocks: 16 blocks with 256 channels
        self.dynamics_resnet_blocks = nn.Sequential(
            *[ResidualBlock(256, num_groups=32) for _ in range(16)]
        )

        # 3. Back-projection: 256 channels -> 64 channels
        self.dynamics_resnet_backprojection = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True)
        )
        # ===============================================================================
        
        # Reward 네트워크 (latent + action -> support_size)
        reward_input_size = 64 + 64 + self.action_space_size  # prev_latent + next_latent + action
        self.reward_input_norm = nn.LayerNorm(reward_input_size)  # 입력 정규화
        self.dynamics_reward_network = mlp(
            reward_input_size, fc_reward_layers, self.num_categorical_distribution_atoms, use_layer_norm=True
        )
        
        # Prediction 네트워크들
        # Policy: Conv layers로 공간 정보 보존하며 압축
        # Input: (B, 64, 25, 25)
        self.prediction_policy_network = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (B, 64, 25, 25) -> (B, 128, 13, 13)
            nn.ReLU(),
            nn.LayerNorm([128, 13, 13]),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # (B, 128, 13, 13) -> (B, 256, 7, 7)
            nn.ReLU(),
            nn.LayerNorm([256, 7, 7]),
            nn.AdaptiveAvgPool2d(3),  # (B, 256, 7, 7) -> (B, 256, 3, 3)
            nn.Flatten(),  # (B, 2304)
            *[layer for layer in mlp(2304, fc_policy_layers, self.action_space_size, use_layer_norm=True)]
        )

        # Value: Policy와 동일한 구조로 강화 (공간 정보 보존)
        # Input: (B, 64, 25, 25)
        self.prediction_value_network = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (B, 64, 25, 25) -> (B, 128, 13, 13)
            nn.ReLU(),
            nn.LayerNorm([128, 13, 13]),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # (B, 128, 13, 13) -> (B, 256, 7, 7)
            nn.ReLU(),
            nn.LayerNorm([256, 7, 7]),
            nn.AdaptiveAvgPool2d(3),  # (B, 256, 7, 7) -> (B, 256, 3, 3)
            nn.Flatten(),  # (B, 2304)
            *[layer for layer in mlp(2304, fc_value_layers, self.num_categorical_distribution_atoms, use_layer_norm=True)]
        )

    def prediction(self, latent):
        """Prediction function: f(s) -> (p, v)
        
        Args:
            latent: latent state representation (B, 64, 25, 25) 또는 (64, 25, 25)
            
        Returns:
            policy_logits: action에 대한 logits (softmax 전)
            value: 상태 가치 (categorical distribution)
        """
        # Latent 차원 처리 (MCTS에서 3차원으로 올 수 있음)
        if len(latent.shape) == 3:  # (64, 25, 25)
            latent = latent.unsqueeze(0)  # (1, 64, 25, 25)
            
        policy_logits = self.prediction_policy_network(latent)
        value = self.prediction_value_network(latent)
        return policy_logits, value

    def representation(self, observation):
        """
        Repulsion/Attraction 분기 + Directional Gating으로 latent representation 생성
        
        Args:
            observation: (B, 6, 99, 99) 
                - Channel 0: obstacles (동적 객체)
                - Channel 1: static_environment (정적 환경)
                - Channel 2: ground_class_id (지면 의미론적 정보)
                - Channel 3: near_destination_direction
                - Channel 4: final_destination_direction
                - Channel 5: simulated_ego_pos
        
        Returns:
            latent: (B, 64, 25, 25) 2D 공간 구조 유지
        """
        batch_size = observation.shape[0]
        device = observation.device
        
        # 1. 방향 벡터 추출 (4차원: cosθ_local, sinθ_local, cosθ_final, sinθ_final)
        goal_vec = torch.zeros((batch_size, 4), device=device)
        if observation.shape[1] >= 5:  # 5채널 이상인 경우
            near_x = observation[:, 3, 0, 0]  # near_x
            near_y = observation[:, 3, 0, 1]  # near_y
            final_x = observation[:, 4, 0, 0]  # final_x
            final_y = observation[:, 4, 0, 1]  # final_y
            
            # 벡터 정규화 및 cos/sin 계산
            near_norm = torch.sqrt(near_x**2 + near_y**2 + 1e-8)    # FIXME: 이미 단위벡터로 정규화되어있음.. 결과값 똑같음
            final_norm = torch.sqrt(final_x**2 + final_y**2 + 1e-8) # FIXME: 이미 단위벡터로 정규화되어있음.. 결과값 똑같음
            
            goal_vec[:, 0] = near_x / near_norm   # cosθ_local
            goal_vec[:, 1] = near_y / near_norm   # sinθ_local
            goal_vec[:, 2] = final_x / final_norm # cosθ_final
            goal_vec[:, 3] = final_y / final_norm # sinθ_final
        
        # 2. Occupancy grid 생성 (Channel 0 + Channel 1 합치기)
        # occupancy_grid = torch.clamp(observation[:, 0:1] + observation[:, 1:2], 0, 1)  # (B, 1, 99, 99)
        
        # 2. Occupancy grid 생성 (Channel 0 + Channel 1 합치기) 
        occupancy_sum = observation[:, 0:1] + observation[:, 1:2]
        # occupancy_grid = torch.clamp(occupancy_sum, 0, 1)  # 연속값 유지 (0~1 범위로 치환) FIXME: 5번 처럼 객체구분위해 가장 큰 거로 나눠야함.
        binary_occupancy = (occupancy_sum > 0).float()     # EDT용 이진 마스크 (0보다 크면 1, 아니면 0)

        # 3. EDT 계산
        edt = compute_edt(binary_occupancy)  # (B, 1, 99, 99)
        
        # 4. Repulsion 분기 입력 준비 (occupancy_grid 는 장애물 종류 구분 정보 / edt 는 장애물 간의 거리 정보)
        # repulsion_input = torch.cat([occupancy_grid, edt], dim=1)  # (B, 2, 99, 99) 
        repulsion_input = edt  # (B, 1, 99, 99) 
        
        # 5. Attraction 분기 입력 준비 (ground_class_id 정규화)
        ground_class_id = observation[:, 2:3]  # (B, 1, 99, 99)
        # class_id를 0~1로 정규화 (최대값으로 나누기)
        max_class_id = 25.0  # rosetta.py에서 확인된 최대 class_id
        attraction_input = ground_class_id / max_class_id  # (B, 1, 99, 99)
        
        # 6. 분기별 인코딩 (Depthwise Separable Conv 각각 수행)
        z_rep = self.repulsion_encoder(repulsion_input)    # (B, 64, 25, 25)
        z_att = self.attraction_encoder(attraction_input)  # (B, 64, 25, 25)
        
        # 7. Fusion ... torch.cat 에 의해 (B, 128, 25, 25) 가 되고, 1x1 Conv로 (B, 64, 25, 25) 로 축소
        z_fuse = self.fuse_1x1(torch.cat([z_rep, z_att], dim=1))  # (B, 64, 25, 25)
        
        # 8. Directional Gating (단일, Fusion 뒤 1회)
        # z_fuse: 99x99 를 압축표현한 텐서
        # goal_vec: 목표 방향벡터 (B, 4)
        z0 = self.dir_gate(z_fuse, goal_vec)  # (B, 64, 25, 25)
        
        return z0

    def _normalize_action_input(self, action, batch_size, device):
        """Action 입력을 (B,) LongTensor로 엄격하게 정규화"""
        if len(action.shape) == 0:  # 스칼라인 경우
            action = action.unsqueeze(0)  # (1,)
        if len(action.shape) == 2:  # (B, 1) 형태인 경우
            action = action.squeeze(-1)  # (B,)
        if len(action.shape) == 1:
            if action.shape[0] == 1 and batch_size > 1:  # 단일 action을 배치로 확장
                action = action.expand(batch_size)
        
        # LongTensor 강제 변환 및 범위 체크
        action = action.long().to(device)
        assert action.shape == (batch_size,), f"Expected shape ({batch_size},), got {action.shape}"
        assert torch.all((action >= 0) & (action < self.action_space_size)), \
            f"Action values must be in [0, {self.action_space_size}), got {action}"
        
        return action
    
    def _create_action_one_hot(self, action, device):
        """안전한 one-hot 생성"""
        return F.one_hot(action, num_classes=self.action_space_size).float().to(device)
    
    def dynamics(self, latent, action):
        """
        Enhanced Dynamics with FiLM + (ConvGRU or ResNet)
        g(s_t, a_t) -> (r_t, s_{t+1})

        Args:
            latent: 현재 latent state (B, 64, 25, 25) 또는 (64, 25, 25)
            action: 수행할 action (B,) or (B, 1) 또는 스칼라

        Returns:
            next_latent: 다음 latent state (B, 64, 25, 25)
            reward: 예측된 보상 (categorical distribution)
        """
        # Latent 차원 정규화
        if len(latent.shape) == 3:  # (64, 25, 25)
            latent = latent.unsqueeze(0)  # (1, 64, 25, 25)

        batch_size = latent.shape[0]
        device = latent.device

        # Action 입력 엄격화 및 안전한 one-hot 생성 (B, 9)
        action = self._normalize_action_input(action, batch_size, device)
        action_one_hot = self._create_action_one_hot(action, device) # ex. [0,0,0,1,0,0,0,0,0], (B, 9)

        # 1단계: FiLM을 통한 안정화된 Action Conditioning
        # Action 별 Latent 변화 유도 (Action이 latent feature를 채널별로 modulate (안정화된 스케일링)
        action_modulated = self.action_film(latent, action_one_hot)  # (B, 64, 25, 25) -> (B, 64, 25, 25)

        if self.use_resnet_dynamics:
            # ========== ResNet 경로 (16 blocks, 256 channels) ==========
            # 2-A: ResNet으로 강력한 특징 추출
            # 64 -> 256 -> 16 blocks -> 64
            next_latent = self.dynamics_resnet_projection(action_modulated)  # (B, 64, 25, 25) -> (B, 256, 25, 25)
            next_latent = self.dynamics_resnet_blocks(next_latent)  # (B, 256, 25, 25) -> (B, 256, 25, 25)
            next_latent = self.dynamics_resnet_backprojection(next_latent)  # (B, 256, 25, 25) -> (B, 64, 25, 25)
        else:
            # ========== ConvGRU 경로 (기존 방식) ==========
            # 2-B: ConvGRU를 통한 Temporal Evolution with Residual
            # 부모의 상태를 기억(ConvGRU) (순수함수 보장: 항상 인자로 받은 latent만 hidden state로 사용)
            # 매개변수 latent 가 residual 개념으로 들어가는 것
            next_latent = self.dynamics_gru(action_modulated, latent)  # (B, 64, 25, 25) + residual
            # (B, 64, 25, 25) -> (B, 64, 25, 25)

            # 3단계: Final Refinement
            # 특징 추출 강화
            next_latent = self.dynamics_refine(next_latent)  # (B, 64, 25, 25) -> (B, 64, 25, 25)

        # 4단계: Enhanced Reward Prediction with LayerNorm
        # 세 가지 정보 모두 활용: prev + next + action
        prev_features = F.adaptive_avg_pool2d(latent, 1).flatten(1)      # (B, 64)
        next_features = F.adaptive_avg_pool2d(next_latent, 1).flatten(1) # (B, 64)

        reward_input = torch.cat([
            prev_features,  # 현재 상태 정보
            next_features,  # 다음 상태 정보
            action_one_hot  # 수행한 action 정보
        ], dim=1)  # (B, 64+64+9)

        # LayerNorm으로 스케일 민감도 완화
        reward_input_normalized = self.reward_input_norm(reward_input)
        reward = self.dynamics_reward_network(reward_input_normalized)  # (B, support_size)

        return next_latent, reward

    def initial_inference(self, observation):
        """
            초기 추론: 관측값으로부터 가치, 보상, 정책, 은닉 상태를 예측
        """
        latent = self.representation(observation)
        policy_logits, value = self.prediction(latent)
        
        # reward equal to 0 for consistency
        # 중앙값(0)에 해당하는 logits 생성 (log-softmax 이전 상태)
        # 큰 음수로 초기화하고 중앙 인덱스만 0으로 설정
        reward = torch.full((len(observation), self.num_categorical_distribution_atoms), -1e8, device=observation.device)
        reward[:, self.num_categorical_distribution_atoms // 2] = 0  # 중앙값(0)에 높은 확률 할당

        return (
            value,
            reward,
            policy_logits,
            latent,
        )

    def recurrent_inference(self, latent, action):
        """
            재귀적 추론: 초기추론 이후 은닉 상태와 액션으로부터 다음 상태 예측
        """
        next_latent, reward = self.dynamics(latent, action)
        policy_logits, value = self.prediction(next_latent)
        return value, reward, policy_logits, next_latent 