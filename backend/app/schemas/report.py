"""제보 스키마"""
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from uuid import UUID


class ReportCreate(BaseModel):
    hazard_type: str
    description: Optional[str] = None
    latitude: float
    longitude: float
    image_url: Optional[str] = None


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
