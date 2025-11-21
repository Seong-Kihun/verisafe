"""AI 감지 지리 정보 API"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional

from app.schemas.detected_feature import (
    DetectedFeatureResponse,
    DetectedFeatureListResponse,
    DetectedFeatureSummary,
    DetectedFeatureVerify
)
from app.models.detected_feature import DetectedFeature
from app.database import get_db
from app.utils.helpers import db_transaction
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get("/detected-features", response_model=DetectedFeatureListResponse)
async def get_detected_features(
    bounds: Optional[str] = Query(None, description="지도 영역 (south,west,north,east)"),
    feature_type: Optional[str] = Query(None, description="필터: 특징 유형"),
    min_confidence: float = Query(0.7, ge=0.0, le=1.0, description="최소 신뢰도"),
    verified_only: bool = Query(False, description="검증된 항목만"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db)
):
    """
    AI가 감지한 지리 정보 조회

    - bounds: 지도 영역 필터
    - feature_type: building, bridge, hospital, school 등
    - min_confidence: 신뢰도 필터 (0.0~1.0)
    - verified_only: 사용자 검증된 항목만
    """
    try:
        # 기본 쿼리
        query = db.query(DetectedFeature)

        # 신뢰도 필터
        query = query.filter(DetectedFeature.confidence >= min_confidence)

        # 지도 영역 필터
        if bounds:
            try:
                south, west, north, east = map(float, bounds.split(','))
                query = query.filter(
                    DetectedFeature.latitude.between(south, north),
                    DetectedFeature.longitude.between(west, east)
                )
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid bounds format. Use: south,west,north,east")

        # 특징 유형 필터
        if feature_type:
            query = query.filter(DetectedFeature.feature_type == feature_type)

        # 검증 상태 필터
        if verified_only:
            query = query.filter(DetectedFeature.verified == True)

        # 총 개수
        total = query.count()

        # 페이지네이션
        offset = (page - 1) * page_size
        features = query.order_by(DetectedFeature.confidence.desc()).offset(offset).limit(page_size).all()

        logger.info(f"감지된 지리 정보 조회: {len(features)}개 (총 {total}개)")

        return DetectedFeatureListResponse(
            features=[DetectedFeatureResponse.from_orm(f) for f in features],
            total=total,
            page=page,
            page_size=page_size
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"감지 정보 조회 오류: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/detected-features/summary", response_model=DetectedFeatureSummary)
async def get_detected_features_summary(
    db: Session = Depends(get_db)
):
    """
    AI 감지 지리 정보 요약 통계

    - 총 개수
    - 유형별 개수
    - 검증 상태
    """
    try:
        # 총 개수
        total_features = db.query(DetectedFeature).count()

        # 유형별 개수
        by_type_query = db.query(
            DetectedFeature.feature_type,
            func.count(DetectedFeature.id)
        ).group_by(DetectedFeature.feature_type).all()

        by_type = {feature_type: count for feature_type, count in by_type_query}

        # 검증 상태
        verified_count = db.query(DetectedFeature).filter(
            DetectedFeature.verified == True
        ).count()

        pending_verification = db.query(DetectedFeature).filter(
            DetectedFeature.verified == False,
            DetectedFeature.confidence < 0.9
        ).count()

        high_confidence_count = db.query(DetectedFeature).filter(
            DetectedFeature.confidence >= 0.8
        ).count()

        return DetectedFeatureSummary(
            total_features=total_features,
            by_type=by_type,
            verified_count=verified_count,
            pending_verification=pending_verification,
            high_confidence_count=high_confidence_count
        )

    except Exception as e:
        logger.error(f"요약 통계 조회 오류: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/detected-features/{feature_id}", response_model=DetectedFeatureResponse)
async def get_detected_feature(
    feature_id: str,
    db: Session = Depends(get_db)
):
    """특정 감지 항목 상세 조회"""
    try:
        feature = db.query(DetectedFeature).filter(
            DetectedFeature.id == feature_id
        ).first()

        if not feature:
            raise HTTPException(status_code=404, detail="Feature not found")

        return DetectedFeatureResponse.from_orm(feature)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"감지 항목 조회 오류: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/detected-features/{feature_id}/verify")
async def verify_detected_feature(
    feature_id: str,
    verification: DetectedFeatureVerify,
    db: Session = Depends(get_db)
):
    """
    사용자가 AI 감지 항목을 검증

    - is_correct: True/False
    - comment: 검증 코멘트 (선택)
    """
    try:
        feature = db.query(DetectedFeature).filter(
            DetectedFeature.id == feature_id
        ).first()

        if not feature:
            raise HTTPException(status_code=404, detail="Feature not found")

        with db_transaction(db, "AI 감지 항목 검증"):
            # 검증 카운트 증가
            feature.verification_count += 1

            # 검증 판정 (3명 이상 검증하면 verified=True)
            if verification.is_correct:
                if feature.verification_count >= 3:
                    feature.verified = True
                    from datetime import datetime
                    feature.last_verified_at = datetime.utcnow()
            else:
                # 틀렸다고 판정하면 신뢰도 낮춤
                feature.confidence = max(0.0, feature.confidence - 0.1)

        logger.info(f"감지 항목 검증: {feature_id}, 올바름={verification.is_correct}")

        return {
            "message": "Verification recorded",
            "feature_id": feature_id,
            "verified": feature.verified,
            "verification_count": feature.verification_count,
            "confidence": feature.confidence
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"검증 처리 오류: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.delete("/detected-features/{feature_id}")
async def delete_detected_feature(
    feature_id: str,
    db: Session = Depends(get_db)
):
    """감지 항목 삭제 (관리자용)"""
    try:
        feature = db.query(DetectedFeature).filter(
            DetectedFeature.id == feature_id
        ).first()

        if not feature:
            raise HTTPException(status_code=404, detail="Feature not found")

        with db_transaction(db, "AI 감지 항목 삭제"):
            db.delete(feature)

        logger.info(f"감지 항목 삭제: {feature_id}")

        return {"message": "Feature deleted", "feature_id": feature_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"삭제 오류: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/detected-features/trigger-detection")
async def trigger_detection(
    db: Session = Depends(get_db)
):
    """
    AI 감지 프로세스 수동 실행 (관리자용)

    건물, 중요 시설, 다리 등을 자동 감지
    """
    try:
        from app.services.ai_detection import BuildingDetector

        detector = BuildingDetector()

        # 주바 중심으로 전체 감지
        stats = await detector.collect_all_features(db)

        logger.info(f"AI 감지 완료: {stats}")

        return {
            "message": "Detection completed",
            "stats": stats,
            "total": sum(stats.values())
        }

    except Exception as e:
        logger.error(f"AI 감지 오류: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
