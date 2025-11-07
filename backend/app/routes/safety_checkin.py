"""안전 체크인 API"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.safety_checkin import SafetyCheckin
from app.schemas.safety_checkin import (
    SafetyCheckinCreate,
    SafetyCheckinResponse,
    SafetyCheckinConfirmRequest,
    SafetyCheckinConfirmResponse
)
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/register", response_model=SafetyCheckinResponse)
async def register_checkin(
    checkin_data: SafetyCheckinCreate,
    db: Session = Depends(get_db)
):
    """
    안전 체크인 등록

    Parameters:
    - user_id: 사용자 ID
    - route_id: 경로 ID (선택)
    - estimated_arrival_time: 예상 도착 시간
    - destination_lat: 목적지 위도 (선택)
    - destination_lon: 목적지 경도 (선택)

    Returns:
    - 체크인 ID 및 생성 시간
    """
    try:
        checkin = SafetyCheckin(
            user_id=checkin_data.user_id,
            route_id=checkin_data.route_id,
            estimated_arrival_time=checkin_data.estimated_arrival_time,
            destination_lat=checkin_data.destination_lat,
            destination_lon=checkin_data.destination_lon,
            status='active'
        )

        db.add(checkin)
        db.commit()
        db.refresh(checkin)

        logger.info(f"[Safety Check-in] Registered for user {checkin_data.user_id}, ETA: {checkin_data.estimated_arrival_time}")

        return {
            "success": True,
            "checkin_id": checkin.id,
            "created_at": checkin.created_at,
            "message": "Safety check-in registered"
        }

    except Exception as e:
        logger.error(f"Error registering safety check-in: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to register safety check-in: {str(e)}")


@router.post("/{checkin_id}/confirm", response_model=SafetyCheckinConfirmResponse)
async def confirm_checkin(
    checkin_id: int,
    confirm_data: SafetyCheckinConfirmRequest,
    db: Session = Depends(get_db)
):
    """
    안전 도착 확인

    Parameters:
    - checkin_id: 체크인 ID
    - user_id: 사용자 ID (본인 확인)

    Returns:
    - 확인 성공 여부
    """
    try:
        checkin = db.query(SafetyCheckin).filter(
            SafetyCheckin.id == checkin_id,
            SafetyCheckin.user_id == confirm_data.user_id,
            SafetyCheckin.status == 'active'
        ).first()

        if not checkin:
            raise HTTPException(
                status_code=404,
                detail="Check-in not found or already confirmed"
            )

        checkin.status = 'confirmed'
        from datetime import datetime
        checkin.confirmed_at = datetime.now()

        db.commit()

        logger.info(f"[Safety Check-in] User {confirm_data.user_id} confirmed arrival (check-in {checkin_id})")

        return {
            "success": True,
            "message": "Arrival confirmed"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error confirming check-in: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to confirm check-in: {str(e)}")
