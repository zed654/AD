import torch
from enum import Enum

class AgentTypeNeuralNetwork(Enum):
    EGO = "ego"
    
class AgentTypeKinematic(Enum):
    PEDESTRIAN = "pedestrian"
    VEHICLE = "vehicle"

# MuZero 네트워크 구현 가져오기
class AbstractNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def initial_inference(self, observation):
        pass
    
    def recurrent_inference(self, latent, action):
        pass
    
    def dynamics_neural_network(self, latent, action, agent_type: AgentTypeNeuralNetwork = AgentTypeNeuralNetwork.EGO):
        pass
    
    def dynamics_kinematic(self, state, action, agent_type: AgentTypeKinematic = AgentTypeKinematic.PEDESTRIAN):
        pass
    
    def prediction(self, latent):
        pass
    
    def representation(self, observation):
        pass
    
    def get_weights(self):
        return dict_to_cpu(self.state_dict())
    
    def set_weights(self, weights):
        self.load_state_dict(weights)
