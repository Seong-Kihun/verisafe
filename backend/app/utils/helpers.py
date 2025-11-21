"""유틸리티 헬퍼 함수들"""
import uuid
from contextlib import contextmanager
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from fastapi import HTTPException, status
from app.utils.logger import get_logger

logger = get_logger(__name__)


@contextmanager
def db_transaction(db: Session, operation_name: str = "DB operation"):
    """
    안전한 DB 트랜잭션 컨텍스트 매니저
    자동으로 commit/rollback 처리

    Args:
        db: SQLAlchemy 세션
        operation_name: 작업 이름 (로깅용)

    Yields:
        Session 객체

    Raises:
        HTTPException: DB 작업 실패 시 500 에러

    Example:
        >>> with db_transaction(db, "사용자 생성") as session:
        >>>     user = User(username="test")
        >>>     session.add(user)
        >>> # 자동 commit, 에러 시 자동 rollback
    """
    try:
        yield db
        db.commit()
        logger.debug(f"{operation_name} 완료 (commit 성공)")
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"{operation_name} 실패 (rollback): {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{operation_name} 중 데이터베이스 오류가 발생했습니다"
        )
    except Exception as e:
        db.rollback()
        logger.error(f"{operation_name} 예상치 못한 오류 (rollback): {e}", exc_info=True)
        raise


def parse_uuid(uuid_str: str, field_name: str = "ID") -> uuid.UUID:
    """
    문자열을 UUID로 변환하고 유효성 검사

    Args:
        uuid_str: UUID 문자열
        field_name: 필드 이름 (에러 메시지용)

    Returns:
        uuid.UUID 객체

    Raises:
        HTTPException: UUID 형식이 잘못된 경우 400 에러

    Example:
        >>> feature_id = parse_uuid(feature_id_str, "Feature ID")
    """
    try:
        return uuid.UUID(uuid_str)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid {field_name} format"
        )
