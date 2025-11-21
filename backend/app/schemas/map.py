"""지도 데이터 스키마"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from uuid import UUID
from datetime import datetime

from app.utils.validators import validate_latitude, validate_longitude


class LandmarkResponse(BaseModel):
    id: UUID
    name: str
    category: Optional[str] = None
    latitude: float
    longitude: float
    description: Optional[str] = None

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

    class Config:
        from_attributes = True


class HazardResponse(BaseModel):
    id: UUID
    hazard_type: str
    risk_score: int
    latitude: float
    longitude: float
    radius: float
    country: Optional[str] = None
    description: Optional[str] = None
    verified: bool
    source: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    created_at: Optional[datetime] = None

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

    @field_validator('risk_score')
    @classmethod
    def check_risk_score(cls, v):
        """위험 점수 검증 (0-100)"""
        if not 0 <= v <= 100:
            raise ValueError(f"위험 점수는 0-100 범위여야 합니다: {v}")
        return v

    @field_validator('radius')
    @classmethod
    def check_radius(cls, v):
        """반경 검증 (0.1-100 km)"""
        if not 0.1 <= v <= 100:
            raise ValueError(f"반경은 0.1-100km 범위여야 합니다: {v}")
        return v

    class Config:
        from_attributes = True


class MapBoundsResponse(BaseModel):
    landmarks: List[LandmarkResponse]
    hazards: List[HazardResponse]


class AutocompleteResponse(BaseModel):
    id: str
    name: str
    address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    importance: Optional[float] = Field(None, description="OpenStreetMap importance 점수 (0.0-1.0)")

    @field_validator('latitude')
    @classmethod
    def check_latitude(cls, v):
        """위도 검증 (-90 ~ 90)"""
        if v is None:
            return v
        return validate_latitude(v)

    @field_validator('longitude')
    @classmethod
    def check_longitude(cls, v):
        """경도 검증 (-180 ~ 180)"""
        if v is None:
            return v
        return validate_longitude(v)

    @field_validator('importance')
    @classmethod
    def check_importance(cls, v):
        """중요도 검증 (0.0-1.0)"""
        if v is not None and not 0.0 <= v <= 1.0:
            raise ValueError(f"중요도는 0.0-1.0 범위여야 합니다: {v}")
        return v

    class Config:
        from_attributes = True
