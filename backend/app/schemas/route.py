"""경로 계산 스키마"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from uuid import UUID

from app.utils.validators import validate_latitude, validate_longitude


class RoutePoint(BaseModel):
    lat: float
    lng: float

    @field_validator('lat')
    @classmethod
    def check_latitude(cls, v):
        """위도 검증 (-90 ~ 90)"""
        return validate_latitude(v)

    @field_validator('lng')
    @classmethod
    def check_longitude(cls, v):
        """경도 검증 (-180 ~ 180)"""
        return validate_longitude(v)


class Waypoint(BaseModel):
    name: str
    lat: float
    lng: float
    risk: str

    @field_validator('lat')
    @classmethod
    def check_latitude(cls, v):
        """위도 검증 (-90 ~ 90)"""
        return validate_latitude(v)

    @field_validator('lng')
    @classmethod
    def check_longitude(cls, v):
        """경도 검증 (-180 ~ 180)"""
        return validate_longitude(v)


class RouteRequest(BaseModel):
    start: RoutePoint
    end: RoutePoint
    preference: str = "safe"  # safe or fast
    transportation_mode: str = "car"  # car, walking, bicycle
    excluded_hazard_types: List[str] = []  # 경로 계산에서 제외할 위험 유형 리스트


class RouteResponse(BaseModel):
    id: str
    type: str  # safe, fast, balanced, scenic, alternative
    label: str = Field(default="경로", description="사용자 친화적 경로 라벨")
    distance: float = Field(..., description="거리 (km)")
    distance_meters: float = Field(..., description="거리 (m)")
    duration: int = Field(..., description="소요 시간 (분)")
    duration_seconds: int = Field(..., description="소요 시간 (초)")
    risk_score: int = Field(..., ge=0, le=10, description="위험 점수 (0-10)")
    hazard_count: int = Field(default=0, description="경로 근방 위험 정보 개수")
    transportation_mode: str = Field(..., description="이동 수단 (car, walking, bicycle)")
    waypoints: List[List[float]] = Field(..., description="경로 좌표 배열 [[lat, lng], ...]")
    polyline: List[List[float]] = Field(..., description="폴리라인 좌표 (더밀한 좌표)")

    @field_validator('distance', 'distance_meters')
    @classmethod
    def ensure_distance_numeric(cls, v):
        """거리 값을 항상 float으로 보장 (None이나 문자열 방지)"""
        if v is None:
            return 0.0
        try:
            return float(v)
        except (ValueError, TypeError):
            return 0.0

    @field_validator('duration', 'duration_seconds', 'hazard_count')
    @classmethod
    def ensure_duration_numeric(cls, v):
        """시간/개수 값을 항상 int로 보장 (None이나 문자열 방지)"""
        if v is None:
            return 0
        try:
            return int(v)
        except (ValueError, TypeError):
            return 0

    @field_validator('risk_score')
    @classmethod
    def ensure_risk_score_valid(cls, v):
        """위험 점수를 0-10 범위로 보장"""
        if v is None:
            return 0
        try:
            score = int(v)
            return max(0, min(10, score))  # 0-10 범위로 클램핑
        except (ValueError, TypeError):
            return 0


class CalculateRouteResponse(BaseModel):
    routes: List[RouteResponse]


class RouteHazardInfo(BaseModel):
    """경로 상의 위험 정보"""
    hazard_id: UUID
    hazard_type: str
    risk_score: int
    latitude: float
    longitude: float
    distance_from_route: float  # 경로로부터의 거리 (m)
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


class RouteHazardBriefing(BaseModel):
    """경로 위험 정보 브리핑"""
    route_id: str
    hazards: List[RouteHazardInfo] = Field(..., description="경로 근방 위험 정보")
    hazards_by_type: dict = Field(..., description="위험 유형별 그룹화")
    summary: dict = Field(..., description="요약 정보")
