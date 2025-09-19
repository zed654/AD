"""MuZero 설정 관리 유틸리티"""
import json
from pathlib import Path
from typing import Dict, Any, Optional
from .config import MuZeroConfig


def load_config(config_path: Optional[Path] = None) -> MuZeroConfig:
    """JSON 파일에서 설정 로드
    
    Args:
        config_path: 설정 파일 경로. None이면 기본 설정 사용
        
    Returns:
        MuZeroConfig 인스턴스
    """
    config = MuZeroConfig()
    
    if config_path and config_path.exists():
        print(f"📄 설정 파일 로드: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
            
            # 설정값 적용
            for key, value in config_dict.items():
                if hasattr(config, key):
                    # Path 타입 복원
                    if key.endswith('_path') and isinstance(value, str):
                        value = Path(value)
                    setattr(config, key, value)
                else:
                    print(f"⚠️  알 수 없는 설정 키: {key}")
    
    return config


def save_config(config: MuZeroConfig, save_path: Path):
    """설정을 JSON 파일로 저장
    
    Args:
        config: 저장할 설정 객체
        save_path: 저장 경로
    """
    # 딕셔너리로 변환
    config_dict = {}
    for key in dir(config):
        if not key.startswith('_'):
            value = getattr(config, key)
            # 메서드 제외
            if not callable(value):
                config_dict[key] = value
    
    # Path 객체는 문자열로 변환
    config_dict = _convert_paths_to_str(config_dict)
    
    # JSON으로 저장
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, default=str, ensure_ascii=False)
    
    print(f"💾 설정 저장: {save_path}")


def merge_config(base_config: MuZeroConfig, updates: Dict[str, Any]) -> MuZeroConfig:
    """기본 설정에 업데이트 적용
    
    Args:
        base_config: 기본 설정
        updates: 업데이트할 설정 딕셔너리
        
    Returns:
        업데이트된 설정 객체
    """
    for key, value in updates.items():
        if hasattr(base_config, key):
            setattr(base_config, key, value)
        else:
            print(f"⚠️  알 수 없는 설정 키: {key}")
    
    return base_config


def _convert_paths_to_str(obj: Any) -> Any:
    """Path 객체를 재귀적으로 문자열로 변환"""
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: _convert_paths_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_paths_to_str(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_paths_to_str(item) for item in obj)
    else:
        return obj


def print_config(config: MuZeroConfig, title: str = "현재 설정"):
    """설정 내용을 보기 좋게 출력
    
    Args:
        config: 출력할 설정 객체
        title: 출력 제목
    """
    print(f"\n{'='*60}")
    print(f"⚙️  {title}")
    print(f"{'='*60}")
    
    # 카테고리별로 정리
    categories = {
        "🎮 게임 설정": ["observation_shape", "action_space", "max_game_length", "discount"],
        "🧠 네트워크 설정": ["network", "encoding_size", "support_size", "fc_representation_layers", 
                        "fc_dynamics_layers", "fc_reward_layers", "fc_value_layers", "fc_policy_layers"],
        "📚 학습 설정": ["training_steps", "batch_size", "num_epochs", "num_selfplay_games", 
                     "learning_rate", "optimizer", "weight_decay"],
        "💾 저장 설정": ["results_path", "pth_checkpoint_interval_step", "checkpoint_save_dir", 
                     "checkpoint_final_name"],
        "🔧 기타 설정": ["observation_provider_type", "gt_data_path", "num_simulations", 
                     "num_stochastic_steps"]
    }
    
    for category, keys in categories.items():
        print(f"\n{category}")
        for key in keys:
            if hasattr(config, key):
                value = getattr(config, key)
                print(f"  - {key}: {value}")
    
    print(f"\n{'='*60}")