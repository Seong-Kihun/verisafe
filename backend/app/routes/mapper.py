"""매퍼 라우터 - 크라우드소싱 지도 편집 API"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import Optional

from app.database import get_db
from app.models.user import User
from app.models.detected_feature import DetectedFeature
from app.services.auth_service import get_current_user
from app.services.mapper_service import MapperService
from app.schemas.detected_feature import (
    MapperFeatureCreate,
    MapperFeatureUpdate,
    DetectedFeatureResponse,
    DetectedFeatureListResponse,
    MapperContributionSummary
)
from app.utils.logger import get_logger
from app.utils.helpers import parse_uuid

router = APIRouter()
logger = get_logger(__name__)


def require_mapper_role(current_user: User = Depends(get_current_user)) -> User:
    """매퍼 역할 검증"""
    if current_user.role not in ['mapper', 'admin']:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Mapper role required"
        )
    return current_user


@router.post("/features", response_model=DetectedFeatureResponse, status_code=status.HTTP_201_CREATED)
async def create_feature(
    feature: MapperFeatureCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_mapper_role)
):
    """
    매퍼가 지리정보 생성

    - Leaflet.draw 등으로 그린 도형을 GeoJSON 형태로 전송
    - Geometry로부터 자동으로 중심점(lat/lon) 계산
    - 검수 대기 상태로 생성됨
    """
    try:
        # 중복 검사
        if feature.geometry_data and feature.geometry_data.get("type") == "Point":
            coords = feature.geometry_data.get("coordinates", [])
            if len(coords) >= 2:
                lon, lat = coords[0], coords[1]
                duplicates = MapperService.check_duplicate(
                    db, lat, lon, feature.feature_type, threshold_meters=50.0
                )
                if duplicates:
                    logger.warning(f"중복 가능성: {len(duplicates)}개 항목 발견")
                    # 경고만 하고 진행 (검수자가 최종 판단)

        # 생성
        db_feature = MapperService.create_feature(
            db=db,
            feature_data=feature,
            user_id=current_user.id
        )

        return DetectedFeatureResponse.from_orm(db_feature)

    except ValueError as e:
        logger.error(f"Geometry 계산 오류: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid geometry: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Feature 생성 오류: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create feature: {str(e)}"
        )


@router.put("/features/{feature_id}", response_model=DetectedFeatureResponse)
async def update_feature(
    feature_id: str,
    feature_data: MapperFeatureUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_mapper_role)
):
    """
    매퍼가 지리정보 수정

    - 자신이 생성한 항목만 수정 가능
    - AI 감지 결과도 수정 가능 (hybrid로 변경됨)
    """
    feature_uuid = parse_uuid(feature_id, "Feature ID")

    try:
        updated_feature = MapperService.update_feature(
            db=db,
            feature_id=feature_uuid,
            feature_data=feature_data,
            user_id=current_user.id
        )

        if not updated_feature:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Feature not found or no permission to edit"
            )

        return DetectedFeatureResponse.from_orm(updated_feature)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feature 수정 오류: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update feature: {str(e)}"
        )


@router.get("/my-contributions", response_model=DetectedFeatureListResponse)
async def get_my_contributions(
    status: Optional[str] = Query(None, description="상태 필터: pending, approved, rejected"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_mapper_role)
):
    """
    내 기여 목록 조회

    - 내가 생성한 지리정보 목록
    - 상태별 필터링 가능
    """
    try:
        features, total = MapperService.get_my_contributions(
            db=db,
            user_id=current_user.id,
            status_filter=status,
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
        logger.error(f"기여 목록 조회 오류: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get contributions: {str(e)}"
        )


@router.get("/my-summary", response_model=MapperContributionSummary)
async def get_my_summary(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_mapper_role)
):
    """
    내 기여 요약 통계

    - 총 기여 수
    - 상태별 개수 (pending, approved, rejected)
    - 유형별 개수
    """
    try:
        summary = MapperService.get_contribution_summary(
            db=db,
            user_id=current_user.id
        )

        return MapperContributionSummary(**summary)

    except Exception as e:
        logger.error(f"요약 통계 조회 오류: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get summary: {str(e)}"
        )


@router.get("/features/{feature_id}", response_model=DetectedFeatureResponse)
async def get_feature(
    feature_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_mapper_role)
):
    """특정 지리정보 상세 조회"""
    feature_uuid = parse_uuid(feature_id, "Feature ID")
    feature = MapperService.get_feature(db, feature_uuid)

    if not feature:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Feature not found"
        )

    return DetectedFeatureResponse.from_orm(feature)


@router.delete("/features/{feature_id}")
async def delete_feature(
    feature_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_mapper_role)
):
    """
    지리정보 삭제

    - 자신이 생성한 항목만 삭제 가능
    - 승인 전 항목만 삭제 가능
    """
    feature_uuid = parse_uuid(feature_id, "Feature ID")

    try:
        success = MapperService.delete_feature(
            db=db,
            feature_id=feature_uuid,
            user_id=current_user.id
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Feature not found"
            )

        return {"message": "Feature deleted successfully", "feature_id": feature_id}

    except ValueError as e:
        error_msg = str(e)
        if "Only the creator" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=error_msg
            )
        elif "Cannot delete approved" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg
            )
    except Exception as e:
        logger.error(f"Feature 삭제 오류: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete feature: {str(e)}"
        )
