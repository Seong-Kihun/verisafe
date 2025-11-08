"""검수자 라우터 - 지리정보 승인/거부 API"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import Optional

from app.database import get_db
from app.models.user import User
from app.services.auth_service import get_current_user
from app.services.review_service import ReviewService
from app.schemas.detected_feature import (
    DetectedFeatureResponse,
    DetectedFeatureListResponse,
    ReviewAction,
    ReviewerDashboardResponse
)
from app.utils.logger import get_logger
from app.utils.helpers import parse_uuid

router = APIRouter()
logger = get_logger(__name__)


def require_reviewer_role(current_user: User = Depends(get_current_user)) -> User:
    """검수자 역할 검증 (admin만 가능)"""
    if current_user.role != 'admin':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Reviewer (admin) role required"
        )
    return current_user


@router.get("/pending", response_model=DetectedFeatureListResponse)
async def get_pending_reviews(
    source: Optional[str] = Query(None, description="출처 필터: ai, mapper, all"),
    feature_type: Optional[str] = Query(None, description="특징 유형 필터"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_reviewer_role)
):
    """
    검수 대기 목록 조회

    - AI 감지 결과 + 매퍼 입력 통합
    - 출처별 필터링 가능
    - 오래된 것부터 (FIFO)
    """
    try:
        features, total = ReviewService.get_pending_reviews(
            db=db,
            source_filter=source,
            feature_type=feature_type,
            page=page,
            page_size=page_size
        )

        return DetectedFeatureListResponse(
            features=[DetectedFeatureResponse.from_orm(f) for f in features],
            total=total,
            page=page,
            page_size=page_size
        )

    except Exception as e:
        logger.error(f"검수 대기 목록 조회 오류: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get pending reviews: {str(e)}"
        )


@router.post("/{feature_id}/approve", response_model=DetectedFeatureResponse)
async def approve_feature(
    feature_id: str,
    action: ReviewAction,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_reviewer_role)
):
    """
    지리정보 승인

    - 승인된 항목은 실제 지도에 반영됨
    - verified=True로 설정됨
    """
    feature_uuid = parse_uuid(feature_id, "Feature ID")

    try:
        approved_feature = ReviewService.approve_feature(
            db=db,
            feature_id=feature_uuid,
            reviewer_id=current_user.id,
            comment=action.comment
        )

        if not approved_feature:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Feature not found"
            )

        return DetectedFeatureResponse.from_orm(approved_feature)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"승인 처리 오류: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to approve feature: {str(e)}"
        )


@router.post("/{feature_id}/reject", response_model=DetectedFeatureResponse)
async def reject_feature(
    feature_id: str,
    action: ReviewAction,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_reviewer_role)
):
    """
    지리정보 거부

    - 거부 사유 필수
    - 거부된 항목은 지도에 반영되지 않음
    """
    feature_uuid = parse_uuid(feature_id, "Feature ID")

    if not action.comment:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Rejection reason required"
        )

    try:
        rejected_feature = ReviewService.reject_feature(
            db=db,
            feature_id=feature_uuid,
            reviewer_id=current_user.id,
            reason="Manual rejection",
            comment=action.comment
        )

        if not rejected_feature:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Feature not found"
            )

        return DetectedFeatureResponse.from_orm(rejected_feature)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"거부 처리 오류: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reject feature: {str(e)}"
        )


@router.get("/area", response_model=dict)
async def get_area_features(
    bounds: str = Query(..., description="지도 영역 (south,west,north,east)"),
    status: Optional[str] = Query(None, description="상태 필터"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_reviewer_role)
):
    """
    특정 영역의 지리정보 조회 (검수용 지도 뷰)

    - AI + 매퍼 결과 통합 조회
    - 중복 항목 자동 검사
    """
    try:
        south, west, north, east = map(float, bounds.split(','))
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid bounds format. Use: south,west,north,east"
        )

    try:
        result = ReviewService.get_area_features(
            db=db,
            bounds=(south, west, north, east),
            status_filter=status
        )

        return {
            "features": [DetectedFeatureResponse.from_orm(f) for f in result["features"]],
            "duplicates": result["duplicates"]
        }

    except Exception as e:
        logger.error(f"영역 조회 오류: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get area features: {str(e)}"
        )


@router.get("/dashboard", response_model=ReviewerDashboardResponse)
async def get_dashboard_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_reviewer_role)
):
    """
    검수자 대시보드 통계

    - 대기 중 개수
    - 오늘 승인/거부 건수
    - AI vs 매퍼 출처별 통계
    """
    try:
        stats = ReviewService.get_dashboard_stats(db=db)
        return ReviewerDashboardResponse(**stats)

    except Exception as e:
        logger.error(f"대시보드 통계 조회 오류: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get dashboard stats: {str(e)}"
        )


@router.post("/{feature_id}/start-review", response_model=DetectedFeatureResponse)
async def start_review(
    feature_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_reviewer_role)
):
    """
    검토 시작 (under_review 상태로 변경)

    - 다른 검수자가 동시 작업 방지
    """
    feature_uuid = parse_uuid(feature_id, "Feature ID")

    try:
        feature = ReviewService.mark_under_review(
            db=db,
            feature_id=feature_uuid,
            reviewer_id=current_user.id
        )

        if not feature:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Feature not found"
            )

        return DetectedFeatureResponse.from_orm(feature)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"검토 시작 오류: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start review: {str(e)}"
        )
