"""유틸리티 헬퍼 함수들"""
import uuid
from fastapi import HTTPException, status


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
