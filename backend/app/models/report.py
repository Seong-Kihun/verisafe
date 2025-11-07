"""사용자 제보 모델"""
from sqlalchemy import Column, String, Float, DateTime, Text, ForeignKey, Boolean, Integer, JSON
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime

from app.database import Base


class Report(Base):
    """사용자 제보 모델"""
    __tablename__ = "reports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='SET NULL'), nullable=True)

    hazard_type = Column(String(50), nullable=False)
    description = Column(Text)

    # 좌표 (latitude, longitude 사용 - SQLite 호환)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)

    # PostGIS geometry는 PostgreSQL로 전환 시 추가
    # 현재는 latitude/longitude만 사용

    image_url = Column(String(500))
    status = Column(String(20), default='pending', index=True)  # pending, verified, rejected

    # 새로운 필드들 (Phase 1-5 기능)
    severity = Column(String(20), default='medium', index=True)  # low, medium, high
    reported_at = Column(DateTime, default=datetime.utcnow, index=True)  # 실제 발생 시간
    photos = Column(JSON, default=list)  # 사진 URL 배열 (SQLite/PostgreSQL 호환)
    is_draft = Column(Boolean, default=False, index=True)  # 임시 저장 여부
    accuracy = Column(Float)  # GPS 정확도 (미터)
    conditional_data = Column(JSON, default=dict)  # 조건부 질문 데이터 (SQLite/PostgreSQL 호환)
    impact_count = Column(Integer, default=0)  # 영향 받은 사용자 수

    # 검증 정보
    verified_by = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='SET NULL'), nullable=True)
    verified_at = Column(DateTime)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", foreign_keys=[user_id], backref="reports")
    verifier = relationship("User", foreign_keys=[verified_by])

    def __repr__(self):
        return f"<Report(id={self.id}, type={self.hazard_type}, status={self.status})>"
