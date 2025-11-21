"""긴급 상황 스키마"""
from pydantic import BaseModel, field_validator
from typing import Optional
from datetime import datetime

from app.utils.validators import validate_latitude, validate_longitude


class SOSEventCreate(BaseModel):
    """SOS 이벤트 생성 스키마"""
    user_id: int
    latitude: float
    longitude: float
    message: Optional[str] = "Emergency SOS activated"

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


class SOSEventResponse(BaseModel):
    """SOS 이벤트 응답 스키마"""
    id: int
    user_id: int
    latitude: float
    longitude: float
    message: Optional[str] = None
    status: str
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class SOSTriggerResponse(BaseModel):
    """SOS 발동 응답 스키마"""
    success: bool
    sos_id: int
    timestamp: datetime
    nearest_safe_haven: Optional[dict] = None
    message: str


class SOSCancelRequest(BaseModel):
    """SOS 취소 요청 스키마"""
    user_id: int


class SOSCancelResponse(BaseModel):
    """SOS 취소 응답 스키마"""
    success: bool
    message: str
