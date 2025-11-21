"""AI 감지 지리 정보 스키마"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid

from app.utils.validators import validate_latitude, validate_longitude


class GeoJSONGeometry(BaseModel):
    """GeoJSON Geometry"""
    type: str = Field(..., description="Geometry 타입 (Point, LineString, Polygon)")
    coordinates: List = Field(..., description="좌표 배열")


class DetectedFeatureBase(BaseModel):
    """DetectedFeature 기본 스키마"""
    feature_type: str = Field(..., description="특징 유형 (building, bridge, hospital 등)")
    latitude: float = Field(..., description="위도")
    longitude: float = Field(..., description="경도")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="신뢰도 (0-1)")
    detection_source: str = Field(..., description="감지 소스")
    name: Optional[str] = Field(None, description="이름")
    description: Optional[str] = Field(None, description="설명")
    properties: Optional[Dict[str, Any]] = Field(None, description="추가 속성")
    geometry_type: Optional[str] = Field(None, description="Geometry 타입 (point, line, polygon)")
    geometry_data: Optional[Dict[str, Any]] = Field(None, description="GeoJSON 형태")

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


class DetectedFeatureCreate(DetectedFeatureBase):
    """DetectedFeature 생성 스키마 (AI 자동 감지용)"""
    pass


class MapperFeatureCreate(BaseModel):
    """매퍼가 수동으로 생성하는 지리 정보"""
    feature_type: str = Field(..., description="특징 유형 (building, road, bridge 등)")
    name: Optional[str] = Field(None, description="이름")
    description: Optional[str] = Field(None, description="설명")
    properties: Optional[Dict[str, Any]] = Field(None, description="추가 속성 (층수, 재질 등)")
    geometry_type: str = Field(..., description="Geometry 타입 (point, line, polygon)")
    geometry_data: Dict[str, Any] = Field(..., description="GeoJSON 형태")


class MapperFeatureUpdate(BaseModel):
    """매퍼가 기존 항목 수정"""
    name: Optional[str] = None
    description: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None
    geometry_type: Optional[str] = None
    geometry_data: Optional[Dict[str, Any]] = None


class DetectedFeatureResponse(DetectedFeatureBase):
    """DetectedFeature 응답 스키마"""
    id: str
    verified: bool
    verification_count: int
    review_status: str
    created_by_user_id: Optional[str] = None
    reviewed_by_user_id: Optional[str] = None
    review_comment: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    satellite_image_url: Optional[str] = None
    user_photo_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    last_verified_at: Optional[datetime] = None

    @field_validator('id', 'created_by_user_id', 'reviewed_by_user_id', mode='before')
    @classmethod
    def convert_uuid_to_str(cls, v):
        """UUID를 문자열로 변환"""
        if v is None:
            return None
        if isinstance(v, uuid.UUID):
            return str(v)
        return v

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


class ReviewAction(BaseModel):
    """검수 액션 (승인/거부)"""
    comment: Optional[str] = Field(None, description="검수 코멘트")


class ReviewerDashboardResponse(BaseModel):
    """검수자 대시보드 응답"""
    pending_count: int
    under_review_count: int
    approved_today: int
    rejected_today: int
    ai_detected_pending: int
    mapper_created_pending: int


class MapperContributionSummary(BaseModel):
    """매퍼 기여 요약"""
    total_contributions: int
    pending: int
    approved: int
    rejected: int
    by_type: Dict[str, int]
