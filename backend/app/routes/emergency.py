"""긴급 상황 API"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import math

from app.database import get_db
from app.models.sos_event import SOSEvent
from app.models.safe_haven import SafeHaven
from app.schemas.emergency import (
    SOSEventCreate,
    SOSTriggerResponse,
    SOSCancelRequest,
    SOSCancelResponse
)
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    두 좌표 간의 거리 계산 (Haversine formula)
    반환: meters
    """
    R = 6371000  # Earth radius in meters

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


@router.post("/sos", response_model=SOSTriggerResponse)
async def trigger_sos(
    sos_data: SOSEventCreate,
    db: Session = Depends(get_db)
):
    """
    SOS 긴급 알림 발동

    Parameters:
    - user_id: 사용자 ID
    - latitude: 현재 위도
    - longitude: 현재 경도
    - message: 긴급 메시지 (선택)

    Returns:
    - SOS 이벤트 ID
    - 가장 가까운 안전 대피처 정보
    """
    try:
        # 1. SOS 이벤트 저장
        sos_event = SOSEvent(
            user_id=sos_data.user_id,
            latitude=sos_data.latitude,
            longitude=sos_data.longitude,
            message=sos_data.message,
            status='active'
        )

        db.add(sos_event)
        db.commit()
        db.refresh(sos_event)

        logger.warning(f"[SOS] User {sos_data.user_id} activated SOS at ({sos_data.latitude}, {sos_data.longitude})")

        # 2. 가장 가까운 안전 대피처 찾기
        havens = db.query(SafeHaven).all()
        nearest_haven = None
        min_distance = float('inf')

        for haven in havens:
            distance = calculate_distance(
                sos_data.latitude,
                sos_data.longitude,
                haven.latitude,
                haven.longitude
            )
            if distance < min_distance:
                min_distance = distance
                nearest_haven = haven

        # 3. 응답 데이터 준비
        nearest_safe_haven = None
        if nearest_haven:
            nearest_safe_haven = {
                "id": nearest_haven.id,
                "name": nearest_haven.name,
                "category": nearest_haven.category,
                "latitude": nearest_haven.latitude,
                "longitude": nearest_haven.longitude,
                "address": nearest_haven.address,
                "phone": nearest_haven.phone,
                "distance": round(min_distance, 2)
            }

        logger.info(f"[SOS] SOS event {sos_event.id} created. Nearest haven: {nearest_haven.name if nearest_haven else 'None'}")

        return {
            "success": True,
            "sos_id": sos_event.id,
            "timestamp": sos_event.created_at,
            "nearest_safe_haven": nearest_safe_haven,
            "message": "SOS alert sent successfully"
        }

    except Exception as e:
        logger.error(f"Error processing SOS: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to process SOS: {str(e)}")


@router.post("/sos/{sos_id}/cancel", response_model=SOSCancelResponse)
async def cancel_sos(
    sos_id: int,
    cancel_data: SOSCancelRequest,
    db: Session = Depends(get_db)
):
    """
    SOS 취소

    Parameters:
    - sos_id: SOS 이벤트 ID
    - user_id: 사용자 ID (본인 확인)

    Returns:
    - 취소 성공 여부
    """
    try:
        # SOS 이벤트 조회
        sos_event = db.query(SOSEvent).filter(
            SOSEvent.id == sos_id,
            SOSEvent.user_id == cancel_data.user_id,
            SOSEvent.status == 'active'
        ).first()

        if not sos_event:
            raise HTTPException(
                status_code=404,
                detail="SOS not found or already resolved"
            )

        # 상태 업데이트
        sos_event.status = 'cancelled'
        db.commit()

        logger.info(f"[SOS] User {cancel_data.user_id} cancelled SOS {sos_id}")

        return {
            "success": True,
            "message": "SOS cancelled successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling SOS: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to cancel SOS: {str(e)}")
