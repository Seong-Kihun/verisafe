"""위험 정보 모델"""
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, CheckConstraint, Index
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

from app.database import Base


class Hazard(Base):
    """위험 정보 모델 (영역 기반)"""
    __tablename__ = "hazards"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    hazard_type = Column(String(50), nullable=False, index=True)
    risk_score = Column(Integer, nullable=False)

    # 좌표 (latitude, longitude 사용 - SQLite 호환)
    latitude = Column(Float, nullable=False, index=True)  # 위치 검색 최적화
    longitude = Column(Float, nullable=False, index=True)  # 위치 검색 최적화
    radius = Column(Float, nullable=False)  # 영향 반경 (km)

    # 국가 코드 (ISO 3166-1 alpha-2)
    country = Column(String(2), nullable=True, index=True)  # 국가별 필터링용

    # PostGIS geometry는 PostgreSQL로 전환 시 추가
    # 현재는 latitude/longitude만 사용

    # 메타데이터
    source = Column(String(50))  # external_api, user_report, system
    description = Column(Text)
    start_date = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)  # 시간 범위 검색 최적화
    end_date = Column(DateTime, nullable=True, index=True)  # NULL = 영구적, 시간 범위 검색 최적화
    verified = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        CheckConstraint('risk_score >= 0 AND risk_score <= 100', name='check_risk_score_range'),
        # 복합 인덱스: 위치와 날짜를 함께 검색하는 쿼리 최적화
        Index('idx_hazard_location_date', 'latitude', 'longitude', 'start_date'),
        # 활성 위험 정보 검색 최적화
        Index('idx_hazard_active', 'hazard_type', 'end_date'),
    )

    def __repr__(self):
        return f"<Hazard(id={self.id}, type={self.hazard_type}, risk={self.risk_score})>"

    @property
    def is_active(self) -> bool:
        """위험 정보가 현재 활성 상태인지 확인"""
        if self.end_date is None:
            return True
        return datetime.utcnow() < self.end_date


class HazardScoringRule(Base):
    __tablename__ = "hazard_scoring_rules"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    hazard_type = Column(String(50), unique=True, nullable=False)
    
    # 위험도 점수
    base_risk_score = Column(Integer, nullable=False)
    min_risk_score = Column(Integer)
    max_risk_score = Column(Integer)
    
    # 시간 제약
    default_duration_hours = Column(Integer, nullable=False)
    
    # 공간 제약
    default_radius_km = Column(Float, nullable=False)
    
    # 표시
    icon = Column(String(10))
    color = Column(String(20))
    description = Column(String)
    
    created_at = Column(DateTime, default=datetime.utcnow)

