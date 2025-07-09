"""MuZero 로깅 시스템"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """컬러 출력을 위한 포맷터"""
    
    # ANSI 색상 코드
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # 로그 레벨에 따라 색상 적용
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: str = "INFO",
    use_color: bool = True
) -> logging.Logger:
    """로거 설정
    
    Args:
        name: 로거 이름
        log_file: 로그 파일 경로 (None이면 파일 출력 안함)
        level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        use_color: 콘솔 출력에 색상 사용 여부
        
    Returns:
        설정된 로거
    """
    logger = logging.getLogger(name)
    
    # 이미 핸들러가 있으면 제거 (중복 방지)
    logger.handlers.clear()
    
    # 로그 레벨 설정
    logger.setLevel(getattr(logging, level.upper()))
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # 포맷 설정
    if use_color:
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러 (옵션)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, level.upper()))
        
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


class MuZeroLogger:
    """MuZero 전용 로거 래퍼"""
    
    def __init__(self, name: str = "MuZero", log_dir: Optional[Path] = None):
        """
        Args:
            name: 로거 이름
            log_dir: 로그 디렉토리 (None이면 파일 로깅 안함)
        """
        log_file = None
        if log_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"muzero_{timestamp}.log"
        
        self.logger = setup_logger(name, log_file)
        
    def epoch_start(self, epoch: int, total_epochs: int):
        """에폭 시작 로그"""
        self.logger.info(f"📍 Epoch {epoch}/{total_epochs} 시작")
        
    def epoch_end(self, epoch: int, stats: dict):
        """에폭 종료 로그"""
        self.logger.info(
            f"✅ Epoch {epoch} 완료 - "
            f"평균 보상: {stats.get('avg_reward', 0):.2f}, "
            f"평균 길이: {stats.get('avg_length', 0):.1f}"
        )
        
    def training_start(self, config: dict):
        """학습 시작 로그"""
        self.logger.info("🚀 MuZero 학습 시작")
        self.logger.info(f"설정: {config}")
        
    def training_end(self, total_time: float):
        """학습 종료 로그"""
        self.logger.info(f"🎉 학습 완료! 총 소요 시간: {total_time:.2f}초")
        
    def game_result(self, game_id: int, reward: float, length: int):
        """게임 결과 로그"""
        self.logger.debug(f"게임 {game_id}: 보상={reward:.2f}, 길이={length}")
        
    def checkpoint_saved(self, path: Path):
        """체크포인트 저장 로그"""
        self.logger.info(f"💾 체크포인트 저장: {path}")
        
    def warning(self, message: str):
        """경고 메시지"""
        self.logger.warning(f"⚠️  {message}")
        
    def error(self, message: str, exc_info: bool = False):
        """에러 메시지"""
        self.logger.error(f"❌ {message}", exc_info=exc_info)
        
    def info(self, message: str):
        """일반 정보 메시지"""
        self.logger.info(message)
        
    def debug(self, message: str):
        """디버그 메시지"""
        self.logger.debug(message)


# 전역 로거 인스턴스
_global_logger = None


def get_logger(name: str = "MuZero", log_dir: Optional[Path] = None) -> MuZeroLogger:
    """전역 로거 가져오기 또는 생성
    
    Args:
        name: 로거 이름
        log_dir: 로그 디렉토리
        
    Returns:
        MuZeroLogger 인스턴스
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = MuZeroLogger(name, log_dir)
    return _global_logger