"""공통 입력 검증 유틸리티"""
from typing import Optional


def validate_latitude(lat: float) -> float:
    """
    위도 검증 (-90 ~ 90)

    Args:
        lat: 위도 값

    Returns:
        검증된 위도 값

    Raises:
        ValueError: 범위를 벗어난 경우
    """
    if not isinstance(lat, (int, float)):
        raise ValueError(f"위도는 숫자여야 합니다: {type(lat)}")

    if not -90 <= lat <= 90:
        raise ValueError(f"위도는 -90~90 범위여야 합니다: {lat}")

    return float(lat)


def validate_longitude(lng: float) -> float:
    """
    경도 검증 (-180 ~ 180)

    Args:
        lng: 경도 값

    Returns:
        검증된 경도 값

    Raises:
        ValueError: 범위를 벗어난 경우
    """
    if not isinstance(lng, (int, float)):
        raise ValueError(f"경도는 숫자여야 합니다: {type(lng)}")

    if not -180 <= lng <= 180:
        raise ValueError(f"경도는 -180~180 범위여야 합니다: {lng}")

    return float(lng)


def validate_hazard_type(hazard_type: str, allowed_types: set) -> str:
    """
    위험 유형 검증

    Args:
        hazard_type: 위험 유형 문자열
        allowed_types: 허용된 위험 유형 집합

    Returns:
        검증된 위험 유형

    Raises:
        ValueError: 허용되지 않은 유형인 경우
    """
    if not isinstance(hazard_type, str):
        raise ValueError(f"위험 유형은 문자열이어야 합니다: {type(hazard_type)}")

    hazard_type = hazard_type.strip().lower()

    if hazard_type not in allowed_types:
        raise ValueError(
            f"허용되지 않은 위험 유형입니다: {hazard_type}. "
            f"허용된 유형: {', '.join(sorted(allowed_types))}"
        )

    return hazard_type


def validate_radius(radius: float, min_value: float = 0.1, max_value: float = 100.0) -> float:
    """
    반경 검증 (km)

    Args:
        radius: 반경 값 (km)
        min_value: 최소값 (기본 0.1km = 100m)
        max_value: 최대값 (기본 100km)

    Returns:
        검증된 반경 값

    Raises:
        ValueError: 범위를 벗어난 경우
    """
    if not isinstance(radius, (int, float)):
        raise ValueError(f"반경은 숫자여야 합니다: {type(radius)}")

    if not min_value <= radius <= max_value:
        raise ValueError(f"반경은 {min_value}~{max_value}km 범위여야 합니다: {radius}")

    return float(radius)


def validate_severity(severity: str) -> str:
    """
    심각도 검증

    Args:
        severity: 심각도 (low, medium, high, critical)

    Returns:
        검증된 심각도

    Raises:
        ValueError: 허용되지 않은 값인 경우
    """
    allowed_severities = {'low', 'medium', 'high', 'critical'}

    if not isinstance(severity, str):
        raise ValueError(f"심각도는 문자열이어야 합니다: {type(severity)}")

    severity = severity.strip().lower()

    if severity not in allowed_severities:
        raise ValueError(
            f"허용되지 않은 심각도입니다: {severity}. "
            f"허용된 값: {', '.join(sorted(allowed_severities))}"
        )

    return severity


def validate_photo_urls(photos: Optional[list]) -> list:
    """
    사진 URL 목록 검증

    Args:
        photos: 사진 URL 리스트

    Returns:
        검증된 사진 URL 리스트

    Raises:
        ValueError: 잘못된 형식인 경우
    """
    if photos is None:
        return []

    if not isinstance(photos, list):
        raise ValueError(f"사진은 리스트여야 합니다: {type(photos)}")

    # 최대 10개로 제한
    if len(photos) > 10:
        raise ValueError(f"사진은 최대 10개까지 업로드할 수 있습니다: {len(photos)}개")

    # 각 항목이 문자열인지 확인
    for i, photo in enumerate(photos):
        if not isinstance(photo, str):
            raise ValueError(f"사진 URL은 문자열이어야 합니다 (인덱스 {i}): {type(photo)}")

        # 빈 문자열 체크
        if not photo.strip():
            raise ValueError(f"빈 사진 URL이 포함되어 있습니다 (인덱스 {i})")

    return photos


def validate_description(description: Optional[str], max_length: int = 1000) -> Optional[str]:
    """
    설명 텍스트 검증

    Args:
        description: 설명 텍스트
        max_length: 최대 길이 (기본 1000자)

    Returns:
        검증된 설명 텍스트

    Raises:
        ValueError: 너무 긴 경우
    """
    if description is None:
        return None

    if not isinstance(description, str):
        raise ValueError(f"설명은 문자열이어야 합니다: {type(description)}")

    description = description.strip()

    if len(description) > max_length:
        raise ValueError(f"설명은 최대 {max_length}자까지 입력할 수 있습니다: {len(description)}자")

    return description if description else None
