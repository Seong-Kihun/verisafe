"""매퍼 서비스 - 지리정보 수동 입력 비즈니스 로직"""
from typing import Dict, List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, timedelta
import uuid

from app.models.detected_feature import DetectedFeature
from app.models.user import User
from app.schemas.detected_feature import MapperFeatureCreate, MapperFeatureUpdate
from app.utils.logger import get_logger

logger = get_logger(__name__)


class MapperService:
    """매퍼 서비스 - 크라우드소싱 지도 편집"""

    @staticmethod
    def calculate_centroid(geometry_data: Dict) -> Tuple[float, float]:
        """
        GeoJSON geometry로부터 중심점 계산

        Args:
            geometry_data: GeoJSON 형태 {"type": "...", "coordinates": [...]}

        Returns:
            (latitude, longitude) 튜플
        """
        geom_type = geometry_data.get("type", "Point")
        coords = geometry_data.get("coordinates", [])

        if geom_type == "Point":
            # Point: [lon, lat]
            return coords[1], coords[0]

        elif geom_type == "LineString":
            # LineString: [[lon, lat], ...] - 중간점 사용
            if not coords:
                raise ValueError("LineString has no coordinates")
            mid_idx = len(coords) // 2
            return coords[mid_idx][1], coords[mid_idx][0]

        elif geom_type == "Polygon":
            # Polygon: [[[lon, lat], ...]] - 첫 번째 ring의 평균
            if not coords or not coords[0]:
                raise ValueError("Polygon has no coordinates")
            ring = coords[0]
            avg_lon = sum(pt[0] for pt in ring) / len(ring)
            avg_lat = sum(pt[1] for pt in ring) / len(ring)
            return avg_lat, avg_lon

        else:
            raise ValueError(f"Unsupported geometry type: {geom_type}")

    @staticmethod
    def create_feature(
        db: Session,
        feature_data: MapperFeatureCreate,
        user_id: uuid.UUID
    ) -> DetectedFeature:
        """
        매퍼가 지리정보 생성

        Args:
            db: Database session
            feature_data: 생성 데이터
            user_id: 매퍼 사용자 ID

        Returns:
            생성된 DetectedFeature 객체
        """
        # Geometry로부터 중심점 계산
        latitude, longitude = MapperService.calculate_centroid(feature_data.geometry_data)

        # DetectedFeature 생성
        db_feature = DetectedFeature(
            feature_type=feature_data.feature_type,
            latitude=latitude,
            longitude=longitude,
            confidence=1.0,  # 매퍼가 직접 입력한 것이므로 신뢰도 1.0
            detection_source="mapper_created",
            review_status="pending",
            created_by_user_id=user_id,
            name=feature_data.name,
            description=feature_data.description,
            properties=feature_data.properties,
            geometry_type=feature_data.geometry_type,
            geometry_data=feature_data.geometry_data,
            verified=False
        )

        db.add(db_feature)
        db.commit()
        db.refresh(db_feature)

        logger.info(f"매퍼 {user_id}가 지리정보 생성: {db_feature.id} ({db_feature.feature_type})")

        return db_feature

    @staticmethod
    def update_feature(
        db: Session,
        feature_id: uuid.UUID,
        feature_data: MapperFeatureUpdate,
        user_id: uuid.UUID
    ) -> Optional[DetectedFeature]:
        """
        매퍼가 기존 지리정보 수정 (자신이 생성한 항목 또는 AI 결과)

        Args:
            db: Database session
            feature_id: 수정할 feature ID
            feature_data: 수정 데이터
            user_id: 매퍼 사용자 ID

        Returns:
            수정된 DetectedFeature 객체 (권한 없으면 None)
        """
        feature = db.query(DetectedFeature).filter(
            DetectedFeature.id == feature_id
        ).first()

        if not feature:
            return None

        # 권한 확인: 자신이 생성한 항목이거나, AI 감지 결과면 수정 가능
        if feature.created_by_user_id and feature.created_by_user_id != user_id:
            logger.warning(f"매퍼 {user_id}가 다른 사용자의 항목 수정 시도: {feature_id}")
            return None

        # 필드 업데이트
        if feature_data.name is not None:
            feature.name = feature_data.name
        if feature_data.description is not None:
            feature.description = feature_data.description
        if feature_data.properties is not None:
            feature.properties = feature_data.properties
        if feature_data.geometry_type is not None:
            feature.geometry_type = feature_data.geometry_type
        if feature_data.geometry_data is not None:
            feature.geometry_data = feature_data.geometry_data
            # 중심점 재계산
            latitude, longitude = MapperService.calculate_centroid(feature_data.geometry_data)
            feature.latitude = latitude
            feature.longitude = longitude

        # AI 결과를 매퍼가 수정한 경우 detection_source 변경
        if feature.detection_source in ['satellite_ai', 'yolo_v8', 'microsoft_buildings']:
            feature.detection_source = 'hybrid'
            feature.created_by_user_id = user_id

        feature.updated_at = datetime.utcnow()

        db.commit()
        db.refresh(feature)

        logger.info(f"매퍼 {user_id}가 지리정보 수정: {feature_id}")

        return feature

    @staticmethod
    def get_my_contributions(
        db: Session,
        user_id: uuid.UUID,
        status_filter: Optional[str] = None,
        page: int = 1,
        page_size: int = 50
    ) -> Tuple[List[DetectedFeature], int]:
        """
        매퍼의 기여 목록 조회

        Args:
            db: Database session
            user_id: 매퍼 ID
            status_filter: 상태 필터 (pending, approved, rejected)
            page: 페이지 번호
            page_size: 페이지 크기

        Returns:
            (기여 목록, 총 개수) 튜플
        """
        query = db.query(DetectedFeature).filter(
            DetectedFeature.created_by_user_id == user_id
        )

        if status_filter:
            query = query.filter(DetectedFeature.review_status == status_filter)

        total = query.count()

        offset = (page - 1) * page_size
        features = query.order_by(
            DetectedFeature.created_at.desc()
        ).offset(offset).limit(page_size).all()

        return features, total

    @staticmethod
    def get_contribution_summary(
        db: Session,
        user_id: uuid.UUID
    ) -> Dict:
        """
        매퍼의 기여 요약 통계

        Args:
            db: Database session
            user_id: 매퍼 ID

        Returns:
            통계 dict
        """
        total = db.query(DetectedFeature).filter(
            DetectedFeature.created_by_user_id == user_id
        ).count()

        pending = db.query(DetectedFeature).filter(
            DetectedFeature.created_by_user_id == user_id,
            DetectedFeature.review_status == 'pending'
        ).count()

        approved = db.query(DetectedFeature).filter(
            DetectedFeature.created_by_user_id == user_id,
            DetectedFeature.review_status == 'approved'
        ).count()

        rejected = db.query(DetectedFeature).filter(
            DetectedFeature.created_by_user_id == user_id,
            DetectedFeature.review_status == 'rejected'
        ).count()

        # 유형별 통계
        by_type_query = db.query(
            DetectedFeature.feature_type,
            func.count(DetectedFeature.id)
        ).filter(
            DetectedFeature.created_by_user_id == user_id
        ).group_by(DetectedFeature.feature_type).all()

        by_type = {ft: count for ft, count in by_type_query}

        return {
            "total_contributions": total,
            "pending": pending,
            "approved": approved,
            "rejected": rejected,
            "by_type": by_type
        }

    @staticmethod
    def check_duplicate(
        db: Session,
        latitude: float,
        longitude: float,
        feature_type: str,
        threshold_meters: float = 50.0
    ) -> List[DetectedFeature]:
        """
        중복 항목 검사 (같은 위치에 유사한 항목 있는지)

        Args:
            db: Database session
            latitude: 위도
            longitude: 경도
            feature_type: 특징 유형
            threshold_meters: 거리 임계값 (미터)

        Returns:
            중복 가능성 있는 항목 리스트
        """
        # 간단한 bounding box 검색 (정확한 거리는 haversine 필요)
        # 1도 ≈ 111km, threshold_meters를 도 단위로 변환
        degree_threshold = threshold_meters / 111000.0

        candidates = db.query(DetectedFeature).filter(
            DetectedFeature.feature_type == feature_type,
            DetectedFeature.latitude.between(
                latitude - degree_threshold,
                latitude + degree_threshold
            ),
            DetectedFeature.longitude.between(
                longitude - degree_threshold,
                longitude + degree_threshold
            ),
            DetectedFeature.review_status != 'rejected'  # 거부된 항목은 제외
        ).all()

        return candidates

    @staticmethod
    def get_feature(
        db: Session,
        feature_id: uuid.UUID
    ) -> Optional[DetectedFeature]:
        """
        특정 지리정보 조회

        Args:
            db: Database session
            feature_id: Feature ID

        Returns:
            DetectedFeature 객체 (없으면 None)
        """
        return db.query(DetectedFeature).filter(
            DetectedFeature.id == feature_id
        ).first()

    @staticmethod
    def delete_feature(
        db: Session,
        feature_id: uuid.UUID,
        user_id: uuid.UUID
    ) -> bool:
        """
        지리정보 삭제 (자신이 생성한 항목만, 승인 전만)

        Args:
            db: Database session
            feature_id: 삭제할 feature ID
            user_id: 사용자 ID

        Returns:
            삭제 성공 여부

        Raises:
            ValueError: 권한 없음 또는 승인된 항목 삭제 시도
        """
        feature = db.query(DetectedFeature).filter(
            DetectedFeature.id == feature_id
        ).first()

        if not feature:
            return False

        # 권한 확인
        if feature.created_by_user_id != user_id:
            raise ValueError("Only the creator can delete this feature")

        # 승인된 항목은 삭제 불가
        if feature.review_status == 'approved':
            raise ValueError("Cannot delete approved feature")

        db.delete(feature)
        db.commit()

        logger.info(f"매퍼 {user_id}가 feature 삭제: {feature_id}")

        return True
