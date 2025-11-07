"""긴급 상황 스키마"""
from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class SOSEventCreate(BaseModel):
    """SOS 이벤트 생성 스키마"""
    user_id: int
    latitude: float
    longitude: float
    message: Optional[str] = "Emergency SOS activated"


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
