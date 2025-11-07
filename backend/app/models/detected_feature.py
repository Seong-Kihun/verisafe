"""AI 감지 지리 정보 모델 (건물, 다리, 도로 등)"""
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, CheckConstraint, Index, JSON
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

from app.database import Base


class DetectedFeature(Base):
    """AI가 감지한 지리 정보 (건물, 다리, 병원, 학교 등)"""
    __tablename__ = "detected_features"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # 특징 유형
    feature_type = Column(String(50), nullable=False, index=True)
    # 'building', 'bridge', 'hospital', 'school', 'police', 'safe_haven', 'road'

    # 위치
    latitude = Column(Float, nullable=False, index=True)
    longitude = Column(Float, nullable=False, index=True)

    # AI 신뢰도 (0.0 ~ 1.0)
    confidence = Column(Float, nullable=False, default=0.0)

    # 감지 소스
    detection_source = Column(String(50), nullable=False)
    # 'microsoft_buildings', 'yolo_v8', 'user_report', 'osm_missing', 'satellite_ai'

    # 검증 상태
    verified = Column(Boolean, default=False, index=True)
    verification_count = Column(Integer, default=0)  # 사용자 검증 투표 수

    # 추가 정보
    name = Column(String(200), nullable=True)
    description = Column(Text, nullable=True)
    properties = Column(JSON, nullable=True)  # 추가 메타데이터 (면적, 층수 등)

    # 이미지/증거
    satellite_image_url = Column(String(500), nullable=True)
    user_photo_url = Column(String(500), nullable=True)

    # 메타데이터
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_verified_at = Column(DateTime, nullable=True)

    __table_args__ = (
        CheckConstraint('confidence >= 0.0 AND confidence <= 1.0', name='check_confidence_range'),
        # 복합 인덱스: 위치와 유형을 함께 검색
        Index('idx_detected_feature_location_type', 'latitude', 'longitude', 'feature_type'),
        # 검증된 항목만 빠르게 조회
        Index('idx_detected_feature_verified', 'verified', 'feature_type'),
    )

    def __repr__(self):
        return f"<DetectedFeature(id={self.id}, type={self.feature_type}, confidence={self.confidence:.2f})>"

    @property
    def is_high_confidence(self) -> bool:
        """높은 신뢰도 여부 (>= 0.8)"""
        return self.confidence >= 0.8

    @property
    def needs_verification(self) -> bool:
        """사용자 검증이 필요한지 여부"""
        return not self.verified and self.confidence < 0.9
