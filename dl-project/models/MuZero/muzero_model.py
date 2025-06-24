import torch
from .muzero_interface import AbstractNetwork, AgentTypeNeuralNetwork, AgentTypeKinematic

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
    layers = []
    for i, layer_size in enumerate(layer_sizes):
        layers.append(torch.nn.Linear(input_size, layer_size))
        layers.append(activation())
        input_size = layer_size

    layers.append(torch.nn.Linear(input_size, output_size))
    layers.append(output_activation())
    return torch.nn.Sequential(*layers)


##################################
######## Fully Connected #########


class MuZeroFullyConnectedNetwork(AbstractNetwork):
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
        ) # DataParallel 은 기본적으로 모델을 GPU 로 이동시켜놓음
        
        print(f"self.representation_network.device: {next(self.representation_network.parameters()).device}")

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

    def dynamics_neural_network(self, latent, action, agent_type: AgentTypeNeuralNetwork = AgentTypeNeuralNetwork.EGO):
        if agent_type == AgentTypeNeuralNetwork.EGO:
            # Stack latent with a game specific one hot encoded action (See paper appendix Network Architecture)
            action_one_hot = (
                torch.zeros((action.shape[0], self.action_space_size))
                .to(action.device)
                .float()
            )
            action_one_hot.scatter_(1, action.long(), 1.0)
            
            print(f"latent.device: {latent.device}")
            print(f"action_one_hot.device: {action_one_hot.device}")
            x = torch.cat((latent, action_one_hot), dim=1)

            next_latent = self.dynamics_latent_network(x)

            reward = self.dynamics_reward_network(next_latent)

            # Scale latent between [0, 1] (See paper appendix Training)
            min_next_latent = next_latent.min(1, keepdim=True)[0]
            max_next_latent = next_latent.max(1, keepdim=True)[0]
            scale_next_latent = max_next_latent - min_next_latent
            scale_next_latent[scale_next_latent < 1e-5] += 1e-5
            next_latent_normalized = (
                next_latent - min_next_latent
            ) / scale_next_latent

            return next_latent_normalized, reward
        else:
            # 다른 agent_type에 대한 처리 (기본값 반환)
            print("Unknown agent type")
            return None, None
        
        
    def dynamics_kinematic(self, state, action, agent_type: AgentTypeKinematic = AgentTypeKinematic.PEDESTRIAN):
        """
        Kinematic 동역학 모델
        Args:
            state: [batch_size, state_dim] - 에이전트의 현재 상태
            action: [batch_size, action_dim] - 액션
            agent_type: 에이전트 타입 (PEDESTRIAN, VEHICLE)
            dt: 시간 간격 (초)
        """
        dt = 0.1
        if agent_type == AgentTypeKinematic.PEDESTRIAN:
            # 보행자 Kinematic 모델
            # state: [x, y, v, theta] - 위치(x,y), 속력(v), 방향(theta)
            # action: [dv, dtheta] - 속력 변화, 방향 변화
            
            batch_size = state.shape[0]
            
            # 현재 상태 분해
            x = state[:, 0]      # x 위치
            y = state[:, 1]      # y 위치  
            v = state[:, 2]      # 속력
            theta = state[:, 3]  # 방향 (라디안)
            
            # 액션 분해
            dv = action[:, 0]       # 속력 변화 (-1~1 범위로 가정)
            dtheta = action[:, 1]   # 방향 변화 (-1~1 범위로 가정)
            
            # 보행자 제약조건
            max_speed = 2.0         # 최대 속력 (m/s)
            max_accel = 2.0         # 최대 가속도 (m/s²)
            max_angular_vel = 1.0   # 최대 각속도 (rad/s)
            
            # 속력 업데이트 (제약 적용)
            v_new = v + dv * max_accel * dt
            v_new = torch.clamp(v_new, 0.0, max_speed)
            
            # 방향 업데이트 (제약 적용)
            theta_new = theta + dtheta * max_angular_vel * dt
            theta_new = torch.fmod(theta_new, 2 * torch.pi)  # 0~2π 범위로 정규화
            
            # 위치 업데이트 (현재 속력과 방향 사용)
            x_new = x + v_new * torch.cos(theta_new) * dt
            y_new = y + v_new * torch.sin(theta_new) * dt
            
            # 새로운 상태 구성
            state_next = torch.stack([x_new, y_new, v_new, theta_new], dim=1)
            
            return state_next
            
        elif agent_type == AgentTypeKinematic.VEHICLE:
            # 차량 Bicycle 모델
            # state: [x, y, v, theta, delta] - 위치, 속력, 방향, 조향각
            # action: [accel, steering_rate] - 가속도, 조향 변화율
            
            # 현재 상태 분해
            x = state[:, 0]      # x 위치
            y = state[:, 1]      # y 위치
            v = state[:, 2]      # 속력
            theta = state[:, 3]  # 방향
            delta = state[:, 4]  # 조향각
            
            # 액션 분해
            accel = action[:, 0] * 5.0      # 가속도 (-5~5 m/s²)
            steering_rate = action[:, 1] * 0.5  # 조향 변화율 (-0.5~0.5 rad/s)
            
            # 차량 파라미터
            L = 2.8  # 축간거리 (m)
            max_speed = 15.0  # 최대 속력 (m/s)
            max_steering = 0.6  # 최대 조향각 (rad)
            
            # 속력 업데이트
            v_new = v + accel * dt
            v_new = torch.clamp(v_new, 0.0, max_speed)
            
            # 조향각 업데이트
            delta_new = delta + steering_rate * dt
            delta_new = torch.clamp(delta_new, -max_steering, max_steering)
            
            # Bicycle 모델 기반 운동학
            beta = torch.atan(0.5 * torch.tan(delta_new))  # 슬립각
            
            # 위치와 방향 업데이트
            x_new = x + v_new * torch.cos(theta + beta) * dt
            y_new = y + v_new * torch.sin(theta + beta) * dt
            theta_new = theta + (v_new / L) * torch.sin(beta) * dt
            theta_new = torch.fmod(theta_new, 2 * torch.pi)
            
            # 새로운 상태 구성
            state_next = torch.stack([x_new, y_new, v_new, theta_new, delta_new], dim=1)
            
            return state_next
            
        else:
            # 다른 agent_type에 대한 처리 (기본값 반환)
            print("Unknown agent type")
            return None

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
        next_latent, reward = self.dynamics_neural_network(latent, action, AgentTypeNeuralNetwork.EGO)
        # next_state_pedestrian = self.dynamics_kinematic(latent, action, AgentTypeKinematic.PEDESTRIAN)
        # next_state_vehicle = self.dynamics_kinematic(latent, action, AgentTypeKinematic.VEHICLE)
        policy_logits, value = self.prediction(next_latent)
        return value, reward, policy_logits, next_latent


###### End Fully Connected #######
##################################


# ##################################
# ############# ResNet #############


# def conv3x3(in_channels, out_channels, stride=1):
#     return torch.nn.Conv2d(
#         in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
#     )


# # Residual block
# class ResidualBlock(torch.nn.Module):
#     def __init__(self, num_channels, stride=1):
#         super().__init__()
#         self.conv1 = conv3x3(num_channels, num_channels, stride)
#         self.bn1 = torch.nn.BatchNorm2d(num_channels)
#         self.conv2 = conv3x3(num_channels, num_channels)
#         self.bn2 = torch.nn.BatchNorm2d(num_channels)

#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = torch.nn.functional.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out += x
#         out = torch.nn.functional.relu(out)
#         return out


# # Downsample observations before representation network (See paper appendix Network Architecture)
# class DownSample(torch.nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv1 = torch.nn.Conv2d(
#             in_channels,
#             out_channels // 2,
#             kernel_size=3,
#             stride=2,
#             padding=1,
#             bias=False,
#         )
#         self.resblocks1 = torch.nn.ModuleList(
#             [ResidualBlock(out_channels // 2) for _ in range(2)]
#         )
#         self.conv2 = torch.nn.Conv2d(
#             out_channels // 2,
#             out_channels,
#             kernel_size=3,
#             stride=2,
#             padding=1,
#             bias=False,
#         )
#         self.resblocks2 = torch.nn.ModuleList(
#             [ResidualBlock(out_channels) for _ in range(3)]
#         )
#         self.pooling1 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
#         self.resblocks3 = torch.nn.ModuleList(
#             [ResidualBlock(out_channels) for _ in range(3)]
#         )
#         self.pooling2 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

#     def forward(self, x):
#         x = self.conv1(x)
#         for block in self.resblocks1:
#             x = block(x)
#         x = self.conv2(x)
#         for block in self.resblocks2:
#             x = block(x)
#         x = self.pooling1(x)
#         for block in self.resblocks3:
#             x = block(x)
#         x = self.pooling2(x)
#         return x


# class DownsampleCNN(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, h_w):
#         super().__init__()
#         mid_channels = (in_channels + out_channels) // 2
#         self.features = torch.nn.Sequential(
#             torch.nn.Conv2d(
#                 in_channels, mid_channels, kernel_size=h_w[0] * 2, stride=4, padding=2
#             ),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.MaxPool2d(kernel_size=3, stride=2),
#             torch.nn.Conv2d(mid_channels, out_channels, kernel_size=5, padding=2),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.MaxPool2d(kernel_size=3, stride=2),
#         )
#         self.avgpool = torch.nn.AdaptiveAvgPool2d(h_w)

#     def forward(self, x):
#         x = self.features(x)
#         x = self.avgpool(x)
#         return x


# class RepresentationNetwork(torch.nn.Module):
#     def __init__(
#         self,
#         observation_shape,
#         stacked_observations,
#         num_blocks,
#         num_channels,
#         downsample,
#     ):
#         super().__init__()
#         self.downsample = downsample
#         if self.downsample:
#             if self.downsample == "resnet":
#                 self.downsample_net = DownSample(
#                     observation_shape[0] * (stacked_observations + 1)
#                     + stacked_observations,
#                     num_channels,
#                 )
#             elif self.downsample == "CNN":
#                 self.downsample_net = DownsampleCNN(
#                     observation_shape[0] * (stacked_observations + 1)
#                     + stacked_observations,
#                     num_channels,
#                     (
#                         math.ceil(observation_shape[1] / 16),
#                         math.ceil(observation_shape[2] / 16),
#                     ),
#                 )
#             else:
#                 raise NotImplementedError('downsample should be "resnet" or "CNN".')
#         self.conv = conv3x3(
#             observation_shape[0] * (stacked_observations + 1) + stacked_observations,
#             num_channels,
#         )
#         self.bn = torch.nn.BatchNorm2d(num_channels)
#         self.resblocks = torch.nn.ModuleList(
#             [ResidualBlock(num_channels) for _ in range(num_blocks)]
#         )

#     def forward(self, x):
#         if self.downsample:
#             x = self.downsample_net(x)
#         else:
#             x = self.conv(x)
#             x = self.bn(x)
#             x = torch.nn.functional.relu(x)

#         for block in self.resblocks:
#             x = block(x)
#         return x


# class DynamicsNetwork(torch.nn.Module):
#     def __init__(
#         self,
#         num_blocks,
#         num_channels,
#         reduced_channels_reward,
#         fc_reward_layers,
#         full_support_size,
#         block_output_size_reward,
#     ):
#         super().__init__()
#         self.conv = conv3x3(num_channels, num_channels - 1)
#         self.bn = torch.nn.BatchNorm2d(num_channels - 1)
#         self.resblocks = torch.nn.ModuleList(
#             [ResidualBlock(num_channels - 1) for _ in range(num_blocks)]
#         )

#         self.conv1x1_reward = torch.nn.Conv2d(
#             num_channels - 1, reduced_channels_reward, 1
#         )
#         self.block_output_size_reward = block_output_size_reward
#         self.fc = mlp(
#             self.block_output_size_reward,
#             fc_reward_layers,
#             full_support_size,
#         )

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = torch.nn.functional.relu(x)
#         for block in self.resblocks:
#             x = block(x)
#         state = x
#         x = self.conv1x1_reward(x)
#         x = x.view(-1, self.block_output_size_reward)
#         reward = self.fc(x)
#         return state, reward


# class PredictionNetwork(torch.nn.Module):
#     def __init__(
#         self,
#         action_space_size,
#         num_blocks,
#         num_channels,
#         reduced_channels_value,
#         reduced_channels_policy,
#         fc_value_layers,
#         fc_policy_layers,
#         full_support_size,
#         block_output_size_value,
#         block_output_size_policy,
#     ):
#         super().__init__()
#         self.resblocks = torch.nn.ModuleList(
#             [ResidualBlock(num_channels) for _ in range(num_blocks)]
#         )

#         self.conv1x1_value = torch.nn.Conv2d(num_channels, reduced_channels_value, 1)
#         self.conv1x1_policy = torch.nn.Conv2d(num_channels, reduced_channels_policy, 1)
#         self.block_output_size_value = block_output_size_value
#         self.block_output_size_policy = block_output_size_policy
#         self.fc_value = mlp(
#             self.block_output_size_value, fc_value_layers, full_support_size
#         )
#         self.fc_policy = mlp(
#             self.block_output_size_policy,
#             fc_policy_layers,
#             action_space_size,
#         )

#     def forward(self, x):
#         for block in self.resblocks:
#             x = block(x)
#         value = self.conv1x1_value(x)
#         policy = self.conv1x1_policy(x)
#         value = value.view(-1, self.block_output_size_value)
#         policy = policy.view(-1, self.block_output_size_policy)
#         value = self.fc_value(value)
#         policy = self.fc_policy(policy)
#         return policy, value


# class MuZeroResidualNetwork(AbstractNetwork):
#     def __init__(
#         self,
#         observation_shape,
#         stacked_observations,
#         action_space_size,
#         num_blocks,
#         num_channels,
#         reduced_channels_reward,
#         reduced_channels_value,
#         reduced_channels_policy,
#         fc_reward_layers,
#         fc_value_layers,
#         fc_policy_layers,
#         support_size,
#         downsample,
#     ):
#         super().__init__()
#         self.action_space_size = action_space_size
#         self.full_support_size = 2 * support_size + 1
#         block_output_size_reward = (
#             (
#                 reduced_channels_reward
#                 * math.ceil(observation_shape[1] / 16)
#                 * math.ceil(observation_shape[2] / 16)
#             )
#             if downsample
#             else (reduced_channels_reward * observation_shape[1] * observation_shape[2])
#         )

#         block_output_size_value = (
#             (
#                 reduced_channels_value
#                 * math.ceil(observation_shape[1] / 16)
#                 * math.ceil(observation_shape[2] / 16)
#             )
#             if downsample
#             else (reduced_channels_value * observation_shape[1] * observation_shape[2])
#         )

#         block_output_size_policy = (
#             (
#                 reduced_channels_policy
#                 * math.ceil(observation_shape[1] / 16)
#                 * math.ceil(observation_shape[2] / 16)
#             )
#             if downsample
#             else (reduced_channels_policy * observation_shape[1] * observation_shape[2])
#         )

#         self.representation_network = torch.nn.DataParallel(
#             RepresentationNetwork(
#                 observation_shape,
#                 stacked_observations,
#                 num_blocks,
#                 num_channels,
#                 downsample,
#             )
#         )

#         self.dynamics_network = torch.nn.DataParallel(
#             DynamicsNetwork(
#                 num_blocks,
#                 num_channels + 1,
#                 reduced_channels_reward,
#                 fc_reward_layers,
#                 self.full_support_size,
#                 block_output_size_reward,
#             )
#         )

#         self.prediction_network = torch.nn.DataParallel(
#             PredictionNetwork(
#                 action_space_size,
#                 num_blocks,
#                 num_channels,
#                 reduced_channels_value,
#                 reduced_channels_policy,
#                 fc_value_layers,
#                 fc_policy_layers,
#                 self.full_support_size,
#                 block_output_size_value,
#                 block_output_size_policy,
#             )
#         )

#     def prediction(self, latent):
#         policy, value = self.prediction_network(latent)
#         return policy, value

#     def representation(self, observation):
#         latent = self.representation_network(observation)

#         # Scale latent between [0, 1] (See appendix paper Training)
#         min_latent = (
#             latent.view(
#                 -1,
#                 latent.shape[1],
#                 latent.shape[2] * latent.shape[3],
#             )
#             .min(2, keepdim=True)[0]
#             .unsqueeze(-1)
#         )
#         max_latent = (
#             latent.view(
#                 -1,
#                 latent.shape[1],
#                 latent.shape[2] * latent.shape[3],
#             )
#             .max(2, keepdim=True)[0]
#             .unsqueeze(-1)
#         )
#         scale_latent = max_latent - min_latent
#         scale_latent[scale_latent < 1e-5] += 1e-5
#         latent_normalized = (
#             latent - min_latent
#         ) / scale_latent
#         return latent_normalized

#     def dynamics(self, latent, action, agent_type: AgentType = AgentType.EGO):
#         if agent_type == AgentType.EGO:
#             # Stack latent with a game specific one hot encoded action (See paper appendix Network Architecture)
#             action_one_hot = (
#                 torch.ones(
#                     (
#                         latent.shape[0],
#                         1,
#                         latent.shape[2],
#                         latent.shape[3],
#                     )
#                 )
#                 .to(action.device)
#                 .float()
#             )
#             action_one_hot = (
#                 action[:, :, None, None] * action_one_hot / self.action_space_size
#             )
#             x = torch.cat((latent, action_one_hot), dim=1)
#             next_latent, reward = self.dynamics_network(x)

#             # Scale encoded state between [0, 1] (See paper appendix Training)
#             min_next_latent = (
#                 next_latent.view(
#                     -1,
#                     next_latent.shape[1],
#                     next_latent.shape[2] * next_latent.shape[3],
#                 )
#                 .min(2, keepdim=True)[0]
#                 .unsqueeze(-1)
#             )
#             max_next_latent = (
#                 next_latent.view(
#                     -1,
#                     next_latent.shape[1],
#                     next_latent.shape[2] * next_latent.shape[3],
#                 )
#                 .max(2, keepdim=True)[0]
#                 .unsqueeze(-1)
#             )
#             scale_next_latent = max_next_latent - min_next_latent
#             scale_next_latent[scale_next_latent < 1e-5] += 1e-5
#             next_latent_normalized = (
#                 next_latent - min_next_latent
#             ) / scale_next_latent
#             return next_latent_normalized, reward
#         elif agent_type == AgentType.PEDESTRIAN:
#             # Kinematic Dynamics 적용
#             return latent, torch.zeros(latent.shape[0], self.full_support_size).to(latent.device)
#         elif agent_type == AgentType.VEHICLE:
#             # Kinematic Dynamics 적용
#             return latent, torch.zeros(latent.shape[0], self.full_support_size).to(latent.device)
#         else:
#             # 다른 agent_type에 대한 처리 (기본값 반환)
#             return latent, torch.zeros(latent.shape[0], self.full_support_size).to(latent.device)

#     def initial_inference(self, observation):
#         latent = self.representation(observation)
#         policy_logits, value = self.prediction(latent)
#         # reward equal to 0 for consistency
#         reward = torch.log(
#             (
#                 torch.zeros(1, self.full_support_size)
#                 .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
#                 .repeat(len(observation), 1)
#                 .to(observation.device)
#             )
#         )
#         return (
#             value,
#             reward,
#             policy_logits,
#             latent,
#         )

#     def recurrent_inference(self, latent, action):
#         next_latent, reward = self.dynamics(latent, action, AgentType.EGO)
#         next_state_pedestrian = self.dynamics(latent, action, AgentType.PEDESTRIAN)
#         next_state_vehicle = self.dynamics(latent, action, AgentType.VEHICLE)
#         policy_logits, value = self.prediction(next_latent)
#         return value, reward, policy_logits, next_latent


# ########### End ResNet ###########
# ##################################