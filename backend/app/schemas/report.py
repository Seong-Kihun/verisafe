"""제보 스키마"""
from pydantic import BaseModel, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID

from app.utils.validators import (
    validate_latitude,
    validate_longitude,
    validate_hazard_type,
    validate_description,
    validate_photo_urls,
    validate_severity
)

# 허용된 위험 유형 (route.py와 동기화)
ALLOWED_HAZARD_TYPES = {
    'armed_conflict',
    'protest_riot',
    'checkpoint',
    'road_damage',
    'natural_disaster',
    'flood',
    'landslide',
    'safe_haven',
    'other'
}


class ReportCreate(BaseModel):
    hazard_type: str
    description: Optional[str] = None
    latitude: float
    longitude: float
    image_url: Optional[str] = None
    # 새 필드들 (선택적)
    severity: Optional[str] = 'medium'
    photos: Optional[List[str]] = None
    accuracy: Optional[float] = None
    conditional_data: Optional[Dict[str, Any]] = None

    @field_validator('latitude')
    @classmethod
    def check_latitude(cls, v):
        """위도 검증 (-90 ~ 90)"""
        return validate_latitude(v)

    @field_validator('longitude')
    @classmethod
    def check_longitude(cls, v):
        """경도 검증 (-180 ~ 180)"""
        return validate_longitude(v)

    @field_validator('hazard_type')
    @classmethod
    def check_hazard_type(cls, v):
        """위험 유형 검증"""
        return validate_hazard_type(v, ALLOWED_HAZARD_TYPES)

    @field_validator('description')
    @classmethod
    def check_description(cls, v):
        """설명 검증 (최대 1000자)"""
        return validate_description(v, max_length=1000)

    @field_validator('severity')
    @classmethod
    def check_severity(cls, v):
        """심각도 검증 (low, medium, high, critical)"""
        if v is None:
            return 'medium'  # 기본값
        return validate_severity(v)

    @field_validator('photos')
    @classmethod
    def check_photos(cls, v):
        """사진 URL 검증 (최대 10개)"""
        return validate_photo_urls(v)

    @field_validator('accuracy')
    @classmethod
    def check_accuracy(cls, v):
        """위치 정확도 검증 (미터 단위, 0 이상)"""
        if v is not None and v < 0:
            raise ValueError(f"정확도는 0 이상이어야 합니다: {v}")
        return v


class ReportResponse(BaseModel):
    id: UUID
    hazard_type: str
    description: Optional[str] = None
    latitude: float
    longitude: float
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class ReportListResponse(BaseModel):
    reports: list[ReportResponse]
