"""
MuZero Action 정의
9개의 위치로만 액션을 표현 (StopAndGo 제거)
"""
from enum import Enum
from typing import Tuple


class Action:
    """액션 관련 모든 기능을 담은 클래스"""
    
    class Position(Enum):
        """액션 위치 타입"""
        POS0 = "pos0" # 현재위치
        POS1 = "pos1" # 좌측상단
        POS2 = "pos2" # 상단
        POS3 = "pos3" # 우측상단
        POS4 = "pos4" # 우측
        POS5 = "pos5" # 우측하단
        POS6 = "pos6" # 하단
        POS7 = "pos7" # 좌측 하단
        POS8 = "pos8" # 좌측
    
    POSITIONS = list(Position)
    
    def __init__(self, index: int = None, position: 'Action.Position' = None):
        """
        액션 객체 생성
        
        Args:
            index: 액션 인덱스 (0~8)
            position: Action.Position (index가 None일 때 사용)
        """
        if index is not None:
            self.index = index
            self.position = self.POSITIONS[index]
        elif position is not None:
            self.position = position
            self.index = self.POSITIONS.index(position)
        else:
            raise ValueError("Either index or position must be provided")
    
    @classmethod
    def get_action_space_size(cls) -> int:
        """전체 액션 공간 크기 반환 (Position 9개)"""
        return len(cls.POSITIONS)
    
    @classmethod
    def decode(cls, action_index: int) -> 'Action.Position':
        """액션 인덱스를 Position으로 디코딩"""
        return cls.POSITIONS[action_index]
    
    @classmethod
    def encode(cls, position: 'Action.Position') -> int:
        """Position을 액션 인덱스로 인코딩"""
        return cls.POSITIONS.index(position)
    
    @classmethod
    def from_index(cls, action_index: int):
        """인덱스로부터 Action 객체 생성"""
        return cls(index=action_index)
    
    @classmethod
    def from_position(cls, position: 'Action.Position'):
        """Position으로부터 Action 객체 생성"""
        return cls(position=position)
    
    def to_string(self) -> str:
        """액션을 읽기 쉬운 문자열로 변환"""
        return self.position.value.upper()
    
    def __str__(self) -> str:
        return self.to_string()
    
    def __repr__(self) -> str:
        return f"Action(index={self.index}, {self.to_string()})"
    
    # 위치별 이동 벡터 (0.2m 그리드 기준) (x, y) 이며, x 가 전방이며 상단이 +, y 가 우측으로 +
    POSITION_VECTORS = {
        Position.POS0: (0, 0),   # 현재 위치
        Position.POS1: (1, -1),  # 좌측상단
        Position.POS2: (1, 0),   # 상단
        Position.POS3: (1, 1),   # 우측상단
        Position.POS4: (0, 1),   # 우측
        Position.POS5: (-1, 1),  # 우측하단
        Position.POS6: (-1, 0),  # 하단
        Position.POS7: (-1, -1), # 좌측하단
        Position.POS8: (0, -1),  # 좌측
    }
    
    def get_movement_vector(self):
        """이 액션의 이동 벡터를 반환 (그리드 단위)"""
        return self.POSITION_VECTORS[self.position]