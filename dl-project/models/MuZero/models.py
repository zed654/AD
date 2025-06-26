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
):
    """Multi-Layer Perceptron 생성"""
    layers = []
    for i, layer_size in enumerate(layer_sizes):
        layers.append(torch.nn.Linear(input_size, layer_size))
        layers.append(activation())
        input_size = layer_size

    layers.append(torch.nn.Linear(input_size, output_size))
    layers.append(output_activation())
    return torch.nn.Sequential(*layers)


class MuZeroFullyConnectedNetwork(AbstractNetwork):
    """Fully Connected MuZero Network"""
    
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
        
        self.representation_network = torch.nn.DataParallel(
            mlp(
                observation_shape[0]
                * observation_shape[1]
                * observation_shape[2]
                * (stacked_observations + 1)
                + stacked_observations * observation_shape[1] * observation_shape[2],
                fc_representation_layers,
                encoding_size,
            )
        )

        self.dynamics_latent_network = torch.nn.DataParallel(
            mlp(
                encoding_size + self.action_space_size,
                fc_dynamics_layers,
                encoding_size,
            )
        )
        self.dynamics_reward_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_reward_layers, self.full_support_size)
        )

        self.prediction_policy_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_policy_layers, self.action_space_size)
        )
        self.prediction_value_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_value_layers, self.full_support_size)
        )

    def prediction(self, latent):
        policy_logits = self.prediction_policy_network(latent)
        value = self.prediction_value_network(latent)
        return policy_logits, value

    def representation(self, observation):
        latent = self.representation_network(
            observation.view(observation.shape[0], -1)
        )
        # Scale latent between [0, 1] (See appendix paper Training)
        min_latent = latent.min(1, keepdim=True)[0]
        max_latent = latent.max(1, keepdim=True)[0]
        scale_latent = max_latent - min_latent
        scale_latent[scale_latent < 1e-5] += 1e-5
        latent_normalized = (
            latent - min_latent
        ) / scale_latent
        return latent_normalized

    def dynamics(self, latent, action):
        """dynamics_neural_network의 간단한 버전"""
        # Stack latent with a game specific one hot encoded action
        action_one_hot = (
            torch.zeros((action.shape[0], self.action_space_size))
            .to(action.device)
            .float()
        )
        
        # action이 1차원이면 2차원으로 변환
        if len(action.shape) == 1:
            action = action.unsqueeze(1)
        
        action_one_hot.scatter_(1, action.long(), 1.0)
        
        x = torch.cat((latent, action_one_hot), dim=1)

        next_latent = self.dynamics_latent_network(x)
        reward = self.dynamics_reward_network(next_latent)

        # Scale latent between [0, 1]
        min_next_latent = next_latent.min(1, keepdim=True)[0]
        max_next_latent = next_latent.max(1, keepdim=True)[0]
        scale_next_latent = max_next_latent - min_next_latent
        scale_next_latent[scale_next_latent < 1e-5] += 1e-5
        next_latent_normalized = (
            next_latent - min_next_latent
        ) / scale_next_latent

        return next_latent_normalized, reward

    def initial_inference(self, observation):
        latent = self.representation(observation)
        policy_logits, value = self.prediction(latent)
        
        # reward equal to 0 for consistency
        reward = torch.log(
            (
                torch.zeros(1, self.full_support_size)
                .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
                .repeat(len(observation), 1)
                .to(observation.device)
            )
        )

        return (
            value,
            reward,
            policy_logits,
            latent,
        )

    def recurrent_inference(self, latent, action):
        next_latent, reward = self.dynamics(latent, action)
        policy_logits, value = self.prediction(next_latent)
        return value, reward, policy_logits, next_latent 