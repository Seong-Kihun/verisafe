"""경로 계산 스키마"""
from pydantic import BaseModel, Field
from typing import List, Optional
from uuid import UUID


class RoutePoint(BaseModel):
    lat: float
    lng: float


class Waypoint(BaseModel):
    name: str
    lat: float
    lng: float
    risk: str


class RouteRequest(BaseModel):
    start: RoutePoint
    end: RoutePoint
    preference: str = "safe"  # safe or fast
    transportation_mode: str = "car"  # car, walking, bicycle


class RouteResponse(BaseModel):
    id: str
    type: str  # safe or fast
    distance: float = Field(..., description="거리 (km)")
    distance_meters: float = Field(..., description="거리 (m)")
    duration: int = Field(..., description="소요 시간 (분)")
    duration_seconds: int = Field(..., description="소요 시간 (초)")
    risk_score: int = Field(..., ge=0, le=10, description="위험 점수 (0-10)")
    transportation_mode: str = Field(..., description="이동 수단 (car, walking, bicycle)")
    waypoints: List[List[float]] = Field(..., description="경로 좌표 배열 [[lat, lng], ...]")
    polyline: List[List[float]] = Field(..., description="폴리라인 좌표 (더밀한 좌표)")


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


class RouteHazardBriefing(BaseModel):
    """경로 위험 정보 브리핑"""
    route_id: str
    hazards: List[RouteHazardInfo] = Field(..., description="경로 근방 위험 정보")
    hazards_by_type: dict = Field(..., description="위험 유형별 그룹화")
    summary: dict = Field(..., description="요약 정보")
