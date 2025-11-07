"""안전 체크인 스키마"""
from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class SafetyCheckinCreate(BaseModel):
    """안전 체크인 생성 스키마"""
    user_id: int
    route_id: Optional[int] = None
    estimated_arrival_time: datetime
    destination_lat: Optional[float] = None
    destination_lon: Optional[float] = None


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
