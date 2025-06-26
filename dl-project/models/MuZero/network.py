"""
MuZero Network Interface
신경망의 추상 인터페이스 정의
"""
import torch
from abc import ABC, abstractmethod
from typing import Tuple


class AbstractNetwork(torch.nn.Module, ABC):
    """MuZero 네트워크의 추상 기본 클래스"""
    
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def initial_inference(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        초기 추론: 관측값으로부터 가치, 보상, 정책, 은닉 상태를 예측
        
        Args:
            observation: 현재 관측값
            
        Returns:
            value: 예측된 가치
            reward: 예측된 보상 (초기값은 0)
            policy_logits: 정책 로짓
            hidden_state: 은닉 상태 표현
        """
        pass
    
    @abstractmethod
    def recurrent_inference(self, hidden_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        재귀적 추론: 은닉 상태와 액션으로부터 다음 상태 예측
        
        Args:
            hidden_state: 현재 은닉 상태
            action: 선택된 액션
            
        Returns:
            value: 예측된 가치
            reward: 예측된 보상
            policy_logits: 정책 로짓
            next_hidden_state: 다음 은닉 상태
        """
        pass
    
    @abstractmethod
    def representation(self, observation: torch.Tensor) -> torch.Tensor:
        """관측값을 은닉 상태로 인코딩"""
        pass
    
    @abstractmethod
    def prediction(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """은닉 상태로부터 정책과 가치 예측"""
        pass
    
    def get_weights(self):
        """네트워크 가중치를 CPU로 이동하여 반환"""
        return {key: value.cpu() for key, value in self.state_dict().items()}
    
    def set_weights(self, weights):
        """네트워크 가중치 설정"""
        self.load_state_dict(weights) 