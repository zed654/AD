"""
ROS2 bag 파일에서 추출한 데이터의 구조와 의미를 정의하는 Rosetta Stone
rosetta.json을 Python 모듈로 변환
"""

from typing import Tuple

# =============================================================================
# Object Class IDs
# =============================================================================
OBJECT_CLASS_IDS = {
    # Unknown
    99: "Unknown",
    98: "Unregistered", 
    
    # Person
    100: "Person",
    
    # Vehicles
    200: "Bicycle",
    201: "Bus",
    202: "Car",
    203: "Motorcycle", 
    204: "Scooter",
    205: "Truck",
    206: "Cart",
    
    # Static/Dynamic Obstacles
    300: "Bollard",
    400: "Neubie",
    
    # Others
    500: "EV Close",
    501: "EV Open",
    502: "Warning Light",
    503: "Traffic Light",
    504: "Traffic Light Green",
    505: "Traffic Light Red",
    506: "Traffic Light Other",
    507: "Barricade",
    508: "Wheelchair Entrance",
    509: "Animal",
    510: "Wheelchair",
    511: "Stroller",
    512: "Tree Trunk",
}

# Object Colors (for visualization)
OBJECT_COLORS = {
    99: "#000000",   # Unknown - Black
    98: "#000001",   # Unregistered - Near Black
    100: "#DBDB8D",  # Person - Light Yellow
    200: "#98DF8A",  # Bicycle - Light Green
    201: "#FFBB78",  # Bus - Light Orange
    202: "#1F77B4",  # Car - Blue
    203: "#FF007C",  # Motorcycle - Pink
    204: "#C5B0D5",  # Scooter - Light Purple
    205: "#031B81",  # Truck - Dark Blue
    206: "#6000C0",  # Cart - Purple
    300: "#FAD4AE",  # Bollard - Light Peach
    400: "#F7ECE6",  # Neubie - Very Light Pink
    500: "#763D74",  # EV Close - Dark Purple
    501: "#C49C94",  # EV Open - Light Brown
    502: "#733380",  # Warning Light - Purple
    503: "#FF6A4D",  # Traffic Light - Orange Red
    504: "#24B353",  # Traffic Light Green - Green
    505: "#CC3366",  # Traffic Light Red - Red
    506: "#FF6A4D",  # Traffic Light Other - Orange Red
    507: "#EF60B3",  # Barricade - Pink
    508: "#FAFA37",  # Wheelchair Entrance - Yellow
    509: "#17BECF",  # Animal - Cyan
    510: "#DDFF33",  # Wheelchair - Bright Yellow Green
    511: "#008000",  # Stroller - Green
    512: "#FF9896",  # Tree Trunk - Light Red
}

# =============================================================================
# LDM (Local Dynamic Map) Definitions
# =============================================================================

# LDM Obstacles Types
LDM_OBSTACLES_TYPES = {
    0: "None",
    1: "Cliff",
    2: "Static Obstacles",
    3: "Missing Points"
}

# LDM Segmentation Class IDs
LDM_SEGMENTATION_IDS = {
    0: "Background",
    1: "Blocks", 
    2: "Colored Road",
    3: "Cement Asphalt",
    4: "Crosswalk",
    5: "Speed Bumps",
    6: "Manhole",
    7: "Curbs",
    8: "Grass",
    9: "Soil/Gravel",
    10: "Grating",
    11: "Guide Blocks",
    12: "Rock",
    13: "Unidentified",
    14: "Wood",
    15: "Iron",
    16: "Artificial Turf",
    17: "Water",
    18: "Mat",
    19: "Unlabeled",
    20: "Restricted",
    21: "Lane Straight White",
    22: "Lane Straight Yellow",
    23: "Lane Dashed White",
    24: "Lane Dashed Yellow",
    25: "Guide Blocks Stop",
    99: "Other/Unclassified"
}

# LDM Segmentation Colors
LDM_SEGMENTATION_COLORS = {
    0: "#000000",   # Background - Black
    1: "#3D3DF5",   # Blocks - Blue
    2: "#CC3366",   # Colored Road - Red
    3: "#F078F0",   # Cement Asphalt - Light Purple
    4: "#008000",   # Crosswalk - Green
    5: "#733380",   # Speed Bumps - Purple
    6: "#FF6A4D",   # Manhole - Orange
    7: "#5986B3",   # Curbs - Blue Gray
    8: "#00FF00",   # Grass - Bright Green
    9: "#CC9933",   # Soil/Gravel - Brown
    10: "#34D1B7",  # Grating - Turquoise
    11: "#FFCC33",  # Guide Blocks - Yellow
    12: "#DBDB8D",  # Rock - Light Yellow
    13: "#FF007C",  # Unidentified - Pink
    14: "#942701",  # Wood - Dark Brown
    15: "#AEC7E8",  # Iron - Light Blue
    16: "#A7FFD8",  # Artificial Turf - Light Green
    17: "#00498C",  # Water - Dark Blue
    18: "#000000",  # Mat - Black
    19: "#F7ECE6",  # Unlabeled - Very Light Pink
    20: "#6000C0",  # Restricted - Purple
    21: "#FFFFFF",  # Lane Straight White - White
    22: "#F59331",  # Lane Straight Yellow - Orange
    23: "#DBDB8D",  # Lane Dashed White - Light Yellow
    24: "#000000",  # Lane Dashed Yellow - Black
    25: "#FCE7A7",  # Guide Blocks Stop - Light Yellow
}

# LDM Obstacles Type Colors (simplified for MuZero visualization)
LDM_OBSTACLES_COLORS = {
    1: "lime",      # Cliff - Lime Green
    2: "yellow",    # Static Obstacles - Yellow
    3: "purple"     # Missing Points - Purple
}

# =============================================================================
# MuZero Grid Values and Colors
# =============================================================================

# Grid Values for Channel 0 (Dynamic Objects) and Channel 1 (Static Environment)
GRID_VALUE_EMPTY = 0.0
GRID_VALUE_PEDESTRIAN = 0.3
GRID_VALUE_OTHER_OBJECT = 0.4
GRID_VALUE_VEHICLE = 0.6
GRID_VALUE_EGO = 1.0

# Static Environment Grid Values
GRID_VALUE_CLIFF = 0.3  # obstacles_type == 1
GRID_VALUE_STATIC_OBSTACLE = 0.6  # obstacles_type == 2
GRID_VALUE_MISSING_POINTS = 0.9  # obstacles_type == 3

# =============================================================================
# Object Categories
# =============================================================================
OBJECT_CATEGORIES = {
    "unknown": {
        "ids": [98, 99],
        "description": "알 수 없는 객체"
    },
    "person": {
        "ids": [100],
        "description": "사람"
    },
    "vehicle": {
        "ids": [200, 201, 202, 203, 204, 205, 206],
        "description": "차량류 (자전거, 버스, 자동차, 오토바이, 스쿠터, 트럭, 카트)"
    },
    "static_obstacles": {
        "ids": [300],
        "description": "정적 장애물 (볼라드)"
    },
    "dynamic_obstacles": {
        "ids": [400],
        "description": "동적 장애물 (뉴비)"
    },
    "unclassified": {
        "ids": [500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512],
        "description": "미분류 객체 (신호등, 바리케이드, 동물, 휠체어 등)"
    }
}

# =============================================================================
# Utility Functions
# =============================================================================

def get_object_name(class_id: int) -> str:
    """객체 class_id에 대한 이름을 반환합니다."""
    return OBJECT_CLASS_IDS.get(class_id, f"Unknown_{class_id}")


def get_object_color(class_id: int) -> str:
    """객체 class_id에 대한 색상을 반환합니다."""
    return OBJECT_COLORS.get(class_id, "#000000")


def get_ldm_segmentation_name(class_id: int) -> str:
    """LDM segmentation class_id에 대한 이름을 반환합니다."""
    return LDM_SEGMENTATION_IDS.get(class_id, f"LDM_{class_id}")


def get_ldm_segmentation_color(class_id: int) -> str:
    """LDM segmentation class_id에 대한 색상을 반환합니다."""
    return LDM_SEGMENTATION_COLORS.get(class_id, "#FFFFFF")


def get_ldm_obstacles_type_name(obstacles_type: int) -> str:
    """LDM obstacles_type에 대한 이름을 반환합니다."""
    return LDM_OBSTACLES_TYPES.get(obstacles_type, "Unknown")


def get_ldm_obstacles_type_color(obstacles_type: int) -> str:
    """LDM obstacles_type에 대한 색상을 반환합니다."""
    return LDM_OBSTACLES_COLORS.get(obstacles_type, "gray")


def get_object_category(class_id: int) -> str:
    """객체 class_id가 속하는 카테고리를 반환합니다."""
    for category, info in OBJECT_CATEGORIES.items():
        if class_id in info["ids"]:
            return category
    return "unknown"


def is_vehicle(class_id: int) -> bool:
    """주어진 class_id가 차량류인지 확인합니다."""
    return class_id in OBJECT_CATEGORIES["vehicle"]["ids"]


def is_person(class_id: int) -> bool:
    """주어진 class_id가 사람인지 확인합니다."""
    return class_id in OBJECT_CATEGORIES["person"]["ids"]


def get_grid_value_from_class_id(class_id: int) -> float:
    """
    객체 클래스 ID를 MuZero 그리드 값으로 변환합니다.
    
    Args:
        class_id: 객체 클래스 ID
        
    Returns:
        그리드 값 (0.0 ~ 1.0)
    """
    if class_id == 100:  # 보행자
        return GRID_VALUE_PEDESTRIAN
    elif class_id in [202, 204, 205]:  # 차량류 (차량, 트럭, 버스)
        return GRID_VALUE_VEHICLE
    else:  # 기타 객체
        return GRID_VALUE_OTHER_OBJECT


def get_grid_value_info(value: float) -> Tuple[str, str]:
    """
    MuZero 그리드 값에 대한 정보를 반환합니다.
    
    Returns:
        (객체 타입 이름, 색상 코드)
    """
    if value == GRID_VALUE_EMPTY:
        return "Empty", "black"
    elif value == GRID_VALUE_PEDESTRIAN:
        return "Pedestrian", get_object_color(100)
    elif value == GRID_VALUE_OTHER_OBJECT:
        return "Other Object", get_object_color(300)
    elif value == GRID_VALUE_VEHICLE:
        return "Vehicle", get_object_color(202)
    elif value == GRID_VALUE_EGO:
        return "Ego Vehicle", "white"
    else:
        return f"Unknown ({value})", "gray"