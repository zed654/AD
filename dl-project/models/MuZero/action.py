"""
MuZero Action 정의
Position과 StopAndGo의 조합으로 액션을 표현
"""
from enum import Enum
from typing import Tuple


class Action:
    """액션 관련 모든 기능을 담은 클래스"""
    
    class Position(Enum):
        """액션 위치 타입"""
        POS0 = "pos0"
        POS1 = "pos1"
        POS2 = "pos2"
        POS3 = "pos3"
        POS4 = "pos4"
        POS5 = "pos5"
        POS6 = "pos6"
        POS7 = "pos7"
    
    class StopAndGo(Enum):
        """액션 정지/이동 타입"""
        STOP = "stop"
        GO = "go"
    
    POSITIONS = list(Position)
    STOP_GO = list(StopAndGo)
    
    def __init__(self, index: int = None, position: 'Action.Position' = None, stopgo: 'Action.StopAndGo' = None):
        """
        액션 객체 생성
        
        Args:
            index: 액션 인덱스 (0~15)
            position: Action.Position (index가 None일 때 사용)
            stopgo: Action.StopAndGo (index가 None일 때 사용)
        """
        if index is not None:
            self.index = index
            self.position, self.stopgo = self.decode(index)
        elif position is not None and stopgo is not None:
            self.position = position
            self.stopgo = stopgo
            self.index = self.encode(position, stopgo)
        else:
            raise ValueError("Either index or (position, stopgo) must be provided")
    
    @classmethod
    def get_action_space_size(cls) -> int:
        """전체 액션 공간 크기 반환 (Position 8개 x StopAndGo 2개 = 16)"""
        return len(cls.POSITIONS) * len(cls.STOP_GO)
    
    @classmethod
    def decode(cls, action_index: int) -> Tuple['Action.Position', 'Action.StopAndGo']:
        """액션 인덱스를 Position과 StopAndGo 조합으로 디코딩"""
        position_idx = action_index // len(cls.STOP_GO)
        stopgo_idx = action_index % len(cls.STOP_GO)
        
        return cls.POSITIONS[position_idx], cls.STOP_GO[stopgo_idx]
    
    @classmethod
    def encode(cls, position: 'Action.Position', stopgo: 'Action.StopAndGo') -> int:
        """Position과 StopAndGo 조합을 액션 인덱스로 인코딩"""
        position_idx = cls.POSITIONS.index(position)
        stopgo_idx = cls.STOP_GO.index(stopgo)
        
        return position_idx * len(cls.STOP_GO) + stopgo_idx
    
    @classmethod
    def from_index(cls, action_index: int):
        """인덱스로부터 Action 객체 생성"""
        return cls(index=action_index)
    
    @classmethod
    def from_components(cls, position: 'Action.Position', stopgo: 'Action.StopAndGo'):
        """Position과 StopAndGo로부터 Action 객체 생성"""
        return cls(position=position, stopgo=stopgo)
    
    def to_string(self) -> str:
        """액션을 읽기 쉬운 문자열로 변환"""
        return f"{self.position.value.upper()}_{self.stopgo.value.upper()}"
    
    def __str__(self) -> str:
        return self.to_string()
    
    def __repr__(self) -> str:
        return f"Action(index={self.index}, {self.to_string()})" 