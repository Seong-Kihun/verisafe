"""안전 대피처 스키마"""
from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class SafeHavenBase(BaseModel):
    """안전 대피처 기본 스키마"""
    name: str
    category: str  # embassy, hospital, un, police, hotel, shelter
    latitude: float
    longitude: float
    address: Optional[str] = None
    phone: Optional[str] = None
    hours: Optional[str] = None
    capacity: Optional[int] = None
    verified: bool = False
    notes: Optional[str] = None


class SafeHavenCreate(SafeHavenBase):
    """안전 대피처 생성 스키마"""
    pass


class SafeHavenResponse(SafeHavenBase):
    """안전 대피처 응답 스키마"""
    id: int
    distance: Optional[float] = None  # 거리 (meters)
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class SafeHavensListResponse(BaseModel):
    """안전 대피처 목록 응답 스키마"""
    success: bool
    count: int
    data: list[SafeHavenResponse]


class NearestSafeHavenResponse(BaseModel):
    """가장 가까운 안전 대피처 응답 스키마"""
    success: bool
    data: SafeHavenResponse
