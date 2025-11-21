"""안전 체크인 스키마"""
from pydantic import BaseModel, field_validator
from typing import Optional
from datetime import datetime

from app.utils.validators import validate_latitude, validate_longitude


class SafetyCheckinCreate(BaseModel):
    """안전 체크인 생성 스키마"""
    user_id: int
    route_id: Optional[int] = None
    estimated_arrival_time: datetime
    destination_lat: Optional[float] = None
    destination_lon: Optional[float] = None

    @field_validator('destination_lat')
    @classmethod
    def check_latitude(cls, v):
        """위도 검증 (-90 ~ 90)"""
        if v is None:
            return v
        return validate_latitude(v)

    @field_validator('destination_lon')
    @classmethod
    def check_longitude(cls, v):
        """경도 검증 (-180 ~ 180)"""
        if v is None:
            return v
        return validate_longitude(v)


class SafetyCheckinResponse(BaseModel):
    """안전 체크인 응답 스키마"""
    success: bool
    checkin_id: int
    created_at: datetime
    message: str


class SafetyCheckinConfirmRequest(BaseModel):
    """안전 체크인 확인 요청 스키마"""
    user_id: int


class SafetyCheckinConfirmResponse(BaseModel):
    """안전 체크인 확인 응답 스키마"""
    success: bool
    message: str
