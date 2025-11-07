"""표준화된 로깅 시스템"""
import logging
import sys
from typing import Optional


# 로거 싱글톤 딕셔너리
_loggers = {}


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    표준화된 로거 생성

    Args:
        name: 로거 이름 (일반적으로 __name__ 사용)
        level: 로그 레벨 (기본값: INFO)

    Returns:
        logging.Logger 객체

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("서버 시작")
        >>> logger.warning("캐시 미스")
        >>> logger.error("DB 연결 실패", exc_info=True)
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)

    # 레벨 설정
    if level is None:
        level = logging.INFO
    logger.setLevel(level)

    # 기존 핸들러 제거 (중복 방지)
    logger.handlers.clear()

    # 콘솔 핸들러 생성
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # 포맷 설정
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    # 부모 로거로 전파 방지 (중복 출력 방지)
    logger.propagate = False

    _loggers[name] = logger
    return logger


# 전역 로거 (편의용)
default_logger = get_logger("verisafe")
