"""AI 감지 지리 정보 스키마"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


class DetectedFeatureBase(BaseModel):
    """DetectedFeature 기본 스키마"""
    feature_type: str = Field(..., description="특징 유형 (building, bridge, hospital 등)")
    latitude: float = Field(..., description="위도")
    longitude: float = Field(..., description="경도")
    confidence: float = Field(..., ge=0.0, le=1.0, description="신뢰도 (0-1)")
    detection_source: str = Field(..., description="감지 소스")
    name: Optional[str] = Field(None, description="이름")
    description: Optional[str] = Field(None, description="설명")
    properties: Optional[Dict[str, Any]] = Field(None, description="추가 속성")


class DetectedFeatureCreate(DetectedFeatureBase):
    """DetectedFeature 생성 스키마"""
    pass


class DetectedFeatureResponse(DetectedFeatureBase):
    """DetectedFeature 응답 스키마"""
    id: str
    verified: bool
    verification_count: int
    satellite_image_url: Optional[str]
    user_photo_url: Optional[str]
    created_at: datetime
    updated_at: datetime
    last_verified_at: Optional[datetime]

    class Config:
        from_attributes = True


class DetectedFeatureVerify(BaseModel):
    """사용자 검증 요청"""
    is_correct: bool = Field(..., description="올바른 감지인지 여부")
    comment: Optional[str] = Field(None, description="검증 코멘트")


class DetectedFeatureListResponse(BaseModel):
    """DetectedFeature 목록 응답"""
    features: list[DetectedFeatureResponse]
    total: int
    page: int
    page_size: int


class DetectedFeatureSummary(BaseModel):
    """DetectedFeature 요약 통계"""
    total_features: int
    by_type: Dict[str, int]
    verified_count: int
    pending_verification: int
    high_confidence_count: int
