"""
MuZero Neural Network Models
실제 신경망 구현체들
"""
import torch
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


class MuZeroFullyConnectedNetwork(AbstractNetwork):
    """방향 벡터만 사용하는 테스트용 MuZero Network"""
    
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        action_space_size,
        encoding_size,
        fc_reward_layers,
        fc_value_layers,
        fc_policy_layers,
        fc_representation_layers,
        fc_dynamics_layers,
        support_size,
    ):
        super().__init__()
        self.action_space_size = action_space_size
        self.full_support_size = 2 * support_size + 1
        self.observation_shape = observation_shape
        
        # 입력 크기: 방향 벡터(4) + 전체 observation (3 * 99 * 99)
        direction_size = 4
        observation_size = observation_shape[0] * observation_shape[1] * observation_shape[2]
        total_input_size = direction_size + observation_size
        
        self.representation_network = torch.nn.DataParallel(
            mlp(
                total_input_size,  # 방향 벡터 + 전체 observation
                fc_representation_layers,
                encoding_size,
                use_layer_norm=True  # LayerNorm 적용
            )
        )

        self.dynamics_latent_network = torch.nn.DataParallel(
            mlp(
                encoding_size + self.action_space_size,
                fc_dynamics_layers,
                encoding_size,
                use_layer_norm=True  # LayerNorm 적용
            )
        )
        self.dynamics_reward_network = torch.nn.DataParallel(
            mlp(
                encoding_size, 
                fc_reward_layers, 
                self.full_support_size,
                use_layer_norm=True  # LayerNorm 적용
            )
        )

        self.prediction_policy_network = torch.nn.DataParallel(
            mlp(
                encoding_size, 
                fc_policy_layers, 
                self.action_space_size,
                use_layer_norm=True  # LayerNorm 적용
            )
        )
        self.prediction_value_network = torch.nn.DataParallel(
            mlp(
                encoding_size, 
                fc_value_layers, 
                self.full_support_size,
                use_layer_norm=True  # LayerNorm 적용
            )
        )

    def prediction(self, latent):
        """Prediction function: f(s) -> (p, v)
        
        Args:
            latent: latent state representation
            
        Returns:
            policy_logits: action에 대한 logits (softmax 전)
            value: 상태 가치 (categorical distribution)
        """
        policy_logits = self.prediction_policy_network(latent)
        value = self.prediction_value_network(latent)
        return policy_logits, value

    def representation(self, observation):
        """observation에서 방향 벡터와 환경 정보를 추출하여 latent representation 생성"""
        batch_size = observation.shape[0]
        device = observation.device
        
        # 1. 방향 벡터 추출 (4차원)
        # Channel 3: Near destination direction (0,0)=x, (0,1)=y
        # Channel 4: Final destination direction (0,0)=x, (0,1)=y
        direction_vectors = torch.zeros((batch_size, 4), device=device) # (batch_size, 4) 사이즈 생성
        if observation.shape[1] >= 5:  # 5채널 이상 구조인 경우
            direction_vectors[:, 0] = observation[:, 3, 0, 0]  # near_x
            direction_vectors[:, 1] = observation[:, 3, 0, 1]  # near_y
            direction_vectors[:, 2] = observation[:, 4, 0, 0]  # final_x
            direction_vectors[:, 3] = observation[:, 4, 0, 1]  # final_y
        
        # 2. 전체 6채널 데이터를 flatten (6 * 99 * 99 = 58,806차원)
        flattened_observation = observation.view(batch_size, -1)
        
        # 3. 방향 벡터와 전체 observation을 결합 (4 + 58,806 = 58,810차원)
        combined_input = torch.cat([direction_vectors, flattened_observation], dim=1)
        
        # Representation network 통과
        latent = self.representation_network(combined_input)
        
        # 정규화 없이 반환 (LayerNorm이 정규화 담당)
        return latent

    def dynamics(self, latent, action):
        """Dynamics function: g(s_t, a_t) -> (r_t, s_{t+1})
        
        Args:
            latent: 현재 latent state
            action: 수행할 action
            
        Returns:
            next_latent: 다음 latent state
            reward: 예측된 보상 (categorical distribution)
        """
        # Action을 one-hot 인코딩
        action_one_hot = torch.zeros(
            (action.shape[0], self.action_space_size),
            device=action.device,
            dtype=torch.float32
        )
        
        # action이 1차원이면 2차원으로 변환
        if len(action.shape) == 1:
            action = action.unsqueeze(1)
        
        action_one_hot.scatter_(1, action.long(), 1.0)
        
        # Latent와 action 결합
        x = torch.cat([latent, action_one_hot], dim=1)

        # Dynamics network를 통해 다음 latent state 예측
        next_latent = self.dynamics_latent_network(x)
        
        # 다음 latent state로부터 reward 예측
        reward = self.dynamics_reward_network(next_latent)

        # 정규화 없이 반환 (네트워크 내부의 LayerNorm이 정규화 담당)
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
        reward = torch.full((len(observation), self.full_support_size), -1e8, device=observation.device)
        reward[:, self.full_support_size // 2] = 0  # 중앙값(0)에 높은 확률 할당

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