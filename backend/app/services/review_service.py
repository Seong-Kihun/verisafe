"""검수자 서비스 - 지리정보 승인/거부 비즈니스 로직"""
from typing import Dict, List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
from datetime import datetime, timedelta
import uuid

from app.models.detected_feature import DetectedFeature
from app.models.user import User
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ReviewService:
    """검수자 서비스 - 지리정보 품질 관리"""

    @staticmethod
    def approve_feature(
        db: Session,
        feature_id: uuid.UUID,
        reviewer_id: uuid.UUID,
        comment: Optional[str] = None
    ) -> Optional[DetectedFeature]:
        """
        지리정보 승인

        Args:
            db: Database session
            feature_id: 승인할 feature ID
            reviewer_id: 검수자 ID
            comment: 검수 코멘트

        Returns:
            승인된 DetectedFeature 객체
        """
        feature = db.query(DetectedFeature).filter(
            DetectedFeature.id == feature_id
        ).first()

        if not feature:
            logger.warning(f"Feature not found for approval: {feature_id}")
            return None

        # 상태 업데이트
        feature.review_status = 'approved'
        feature.reviewed_by_user_id = reviewer_id
        feature.review_comment = comment
        feature.reviewed_at = datetime.utcnow()
        feature.verified = True  # 승인되면 verified도 True
        feature.last_verified_at = datetime.utcnow()

        db.commit()
        db.refresh(feature)

        logger.info(f"검수자 {reviewer_id}가 지리정보 승인: {feature_id} ({feature.feature_type})")

        return feature

    @staticmethod
    def reject_feature(
        db: Session,
        feature_id: uuid.UUID,
        reviewer_id: uuid.UUID,
        reason: str,
        comment: Optional[str] = None
    ) -> Optional[DetectedFeature]:
        """
        지리정보 거부

        Args:
            db: Database session
            feature_id: 거부할 feature ID
            reviewer_id: 검수자 ID
            reason: 거부 사유
            comment: 추가 코멘트

        Returns:
            거부된 DetectedFeature 객체
        """
        feature = db.query(DetectedFeature).filter(
            DetectedFeature.id == feature_id
        ).first()

        if not feature:
            logger.warning(f"Feature not found for rejection: {feature_id}")
            return None

        # 상태 업데이트
        feature.review_status = 'rejected'
        feature.reviewed_by_user_id = reviewer_id
        feature.review_comment = f"[거부 사유: {reason}] {comment or ''}"
        feature.reviewed_at = datetime.utcnow()
        feature.verified = False

        db.commit()
        db.refresh(feature)

        logger.info(f"검수자 {reviewer_id}가 지리정보 거부: {feature_id} (사유: {reason})")

        return feature

    @staticmethod
    def get_pending_reviews(
        db: Session,
        source_filter: Optional[str] = None,  # 'ai', 'mapper', 'all'
        feature_type: Optional[str] = None,
        page: int = 1,
        page_size: int = 50
    ) -> Tuple[List[DetectedFeature], int]:
        """
        검수 대기 목록 조회

        Args:
            db: Database session
            source_filter: 출처 필터 (ai, mapper, all)
            feature_type: 특징 유형 필터
            page: 페이지 번호
            page_size: 페이지 크기

        Returns:
            (대기 목록, 총 개수) 튜플
        """
        query = db.query(DetectedFeature).filter(
            or_(
                DetectedFeature.review_status == 'pending',
                DetectedFeature.review_status == 'under_review'
            )
        )

        # 출처 필터
        if source_filter == 'ai':
            query = query.filter(
                DetectedFeature.detection_source.in_([
                    'satellite_ai', 'yolo_v8', 'microsoft_buildings'
                ])
            )
        elif source_filter == 'mapper':
            query = query.filter(
                DetectedFeature.detection_source.in_(['mapper_created', 'hybrid'])
            )

        # 특징 유형 필터
        if feature_type:
            query = query.filter(DetectedFeature.feature_type == feature_type)

        total = query.count()

        # 오래된 것부터 (FIFO)
        offset = (page - 1) * page_size
        features = query.order_by(
            DetectedFeature.created_at.asc()
        ).offset(offset).limit(page_size).all()

        return features, total

    @staticmethod
    def get_area_features(
        db: Session,
        bounds: Tuple[float, float, float, float],  # (south, west, north, east)
        status_filter: Optional[str] = None
    ) -> Dict:
        """
        특정 영역의 지리정보 조회 (지도 뷰용)

        Args:
            db: Database session
            bounds: 경계 (south, west, north, east)
            status_filter: 상태 필터

        Returns:
            지리정보 목록 + 중복 검사 결과
        """
        south, west, north, east = bounds

        query = db.query(DetectedFeature).filter(
            DetectedFeature.latitude.between(south, north),
            DetectedFeature.longitude.between(west, east)
        )

        if status_filter:
            query = query.filter(DetectedFeature.review_status == status_filter)

        features = query.all()

        # 중복 검사 (같은 위치에 여러 항목)
        duplicates = ReviewService._find_duplicates(features)

        return {
            "features": features,
            "duplicates": duplicates
        }

    @staticmethod
    def _find_duplicates(
        features: List[DetectedFeature],
        threshold_meters: float = 50.0
    ) -> List[Dict]:
        """
        같은 위치의 중복 항목 찾기

        Args:
            features: 검사할 항목 리스트
            threshold_meters: 거리 임계값

        Returns:
            중복 항목 리스트 [{"location": [lat, lon], "items": [id1, id2]}]
        """
        duplicates = []
        checked = set()

        degree_threshold = threshold_meters / 111000.0

        for i, feature1 in enumerate(features):
            if feature1.id in checked:
                continue

            cluster = [feature1.id]

            for j, feature2 in enumerate(features):
                if i == j or feature2.id in checked:
                    continue

                # 간단한 거리 비교
                lat_diff = abs(feature1.latitude - feature2.latitude)
                lon_diff = abs(feature1.longitude - feature2.longitude)

                if lat_diff < degree_threshold and lon_diff < degree_threshold:
                    # 같은 타입인지 확인
                    if feature1.feature_type == feature2.feature_type:
                        cluster.append(feature2.id)
                        checked.add(feature2.id)

            if len(cluster) > 1:
                duplicates.append({
                    "location": [feature1.latitude, feature1.longitude],
                    "items": [str(item_id) for item_id in cluster]
                })
                checked.add(feature1.id)

        return duplicates

    @staticmethod
    def get_dashboard_stats(db: Session) -> Dict:
        """
        검수자 대시보드 통계

        Args:
            db: Database session

        Returns:
            통계 dict
        """
        # 전체 대기 중
        pending_count = db.query(DetectedFeature).filter(
            DetectedFeature.review_status == 'pending'
        ).count()

        # 검토 중
        under_review_count = db.query(DetectedFeature).filter(
            DetectedFeature.review_status == 'under_review'
        ).count()

        # 오늘 승인/거부 건수
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

        approved_today = db.query(DetectedFeature).filter(
            DetectedFeature.review_status == 'approved',
            DetectedFeature.reviewed_at >= today_start
        ).count()

        rejected_today = db.query(DetectedFeature).filter(
            DetectedFeature.review_status == 'rejected',
            DetectedFeature.reviewed_at >= today_start
        ).count()

        # 출처별 대기 건수
        ai_detected_pending = db.query(DetectedFeature).filter(
            DetectedFeature.review_status.in_(['pending', 'under_review']),
            DetectedFeature.detection_source.in_([
                'satellite_ai', 'yolo_v8', 'microsoft_buildings'
            ])
        ).count()

        mapper_created_pending = db.query(DetectedFeature).filter(
            DetectedFeature.review_status.in_(['pending', 'under_review']),
            DetectedFeature.detection_source.in_(['mapper_created', 'hybrid'])
        ).count()

        return {
            "pending_count": pending_count,
            "under_review_count": under_review_count,
            "approved_today": approved_today,
            "rejected_today": rejected_today,
            "ai_detected_pending": ai_detected_pending,
            "mapper_created_pending": mapper_created_pending
        }

    @staticmethod
    def mark_under_review(
        db: Session,
        feature_id: uuid.UUID,
        reviewer_id: uuid.UUID
    ) -> Optional[DetectedFeature]:
        """
        검토 중 상태로 변경 (다른 검수자가 동시 작업 방지)

        Args:
            db: Database session
            feature_id: Feature ID
            reviewer_id: 검수자 ID

        Returns:
            DetectedFeature 객체
        """
        feature = db.query(DetectedFeature).filter(
            DetectedFeature.id == feature_id
        ).first()

        if not feature:
            return None

        if feature.review_status == 'pending':
            feature.review_status = 'under_review'
            feature.reviewed_by_user_id = reviewer_id
            db.commit()
            db.refresh(feature)

        return feature
