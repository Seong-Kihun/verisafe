"""지도 데이터 스키마"""
from pydantic import BaseModel, Field
from typing import Optional, List
from uuid import UUID
from datetime import datetime


class LandmarkResponse(BaseModel):
    id: UUID
    name: str
    category: Optional[str] = None
    latitude: float
    longitude: float
    description: Optional[str] = None

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

    class Config:
        from_attributes = True
