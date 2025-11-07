"""안전 대피처 API"""
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from typing import Optional
import math

from app.database import get_db
from app.models.safe_haven import SafeHaven
from app.schemas.safe_haven import SafeHavensListResponse, NearestSafeHavenResponse, SafeHavenResponse
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    두 좌표 간의 거리 계산 (Haversine formula)
    반환: meters
    """
    # Earth radius in meters
    R = 6371000

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


@router.get("/", response_model=SafeHavensListResponse)
async def get_safe_havens(
    lat: float = Query(..., description="위도"),
    lon: float = Query(..., description="경도"),
    radius: int = Query(5000, description="검색 반경 (meters)", ge=100, le=50000),
    category: Optional[str] = Query(None, description="카테고리: embassy, hospital, un, police, hotel, shelter"),
    db: Session = Depends(get_db)
):
    """
    주변 안전 대피처 조회

    Parameters:
    - lat: 중심 위도
    - lon: 중심 경도
    - radius: 검색 반경 (기본: 5000m = 5km)
    - category: 필터링할 카테고리 (선택)

    Returns:
    - 안전 대피처 목록 (거리순 정렬, 최대 50개)
    """
    try:
        # Base query
        query = db.query(SafeHaven)

        # Category filter
        if category:
            query = query.filter(SafeHaven.category == category)

        # Get all potential havens
        havens = query.all()

        # Calculate distances and filter by radius
        results = []
        for haven in havens:
            distance = calculate_distance(lat, lon, haven.latitude, haven.longitude)
            if distance <= radius:
                # Convert to dict and add distance
                haven_dict = {
                    "id": haven.id,
                    "name": haven.name,
                    "category": haven.category,
                    "latitude": haven.latitude,
                    "longitude": haven.longitude,
                    "address": haven.address,
                    "phone": haven.phone,
                    "hours": haven.hours,
                    "capacity": haven.capacity,
                    "verified": haven.verified,
                    "notes": haven.notes,
                    "distance": round(distance, 2),
                    "created_at": haven.created_at,
                    "updated_at": haven.updated_at,
                }
                results.append(haven_dict)

        # Sort by distance
        results.sort(key=lambda x: x["distance"])

        # Limit to 50
        results = results[:50]

        logger.info(f"Found {len(results)} safe havens within {radius}m of ({lat}, {lon})")

        return {
            "success": True,
            "count": len(results),
            "data": results
        }

    except Exception as e:
        logger.error(f"Error fetching safe havens: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch safe havens: {str(e)}")


@router.get("/nearest", response_model=NearestSafeHavenResponse)
async def get_nearest_safe_haven(
    lat: float = Query(..., description="위도"),
    lon: float = Query(..., description="경도"),
    db: Session = Depends(get_db)
):
    """
    가장 가까운 안전 대피처 조회

    Parameters:
    - lat: 현재 위도
    - lon: 현재 경도

    Returns:
    - 가장 가까운 안전 대피처 (카테고리 무관)
    """
    try:
        # Get all havens
        havens = db.query(SafeHaven).all()

        if not havens:
            raise HTTPException(status_code=404, detail="No safe havens found")

        # Find nearest
        nearest = None
        min_distance = float('inf')

        for haven in havens:
            distance = calculate_distance(lat, lon, haven.latitude, haven.longitude)
            if distance < min_distance:
                min_distance = distance
                nearest = haven

        if nearest is None:
            raise HTTPException(status_code=404, detail="No safe havens found")

        # Convert to dict and add distance
        result = {
            "id": nearest.id,
            "name": nearest.name,
            "category": nearest.category,
            "latitude": nearest.latitude,
            "longitude": nearest.longitude,
            "address": nearest.address,
            "phone": nearest.phone,
            "hours": nearest.hours,
            "capacity": nearest.capacity,
            "verified": nearest.verified,
            "notes": nearest.notes,
            "distance": round(min_distance, 2),
            "created_at": nearest.created_at,
            "updated_at": nearest.updated_at,
        }

        logger.info(f"Nearest safe haven: {nearest.name} at {min_distance:.2f}m")

        return {
            "success": True,
            "data": result
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching nearest safe haven: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch nearest safe haven: {str(e)}")
