"""랜드마크 모델"""
from sqlalchemy import Column, String, Float, BigInteger, DateTime, Text
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

from app.database import Base


class Landmark(Base):
    """주요 지점 모델"""
    __tablename__ = "landmarks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False, index=True)
    category = Column(String(50), index=True)  # airport, government, hospital 등

    # 좌표 (latitude, longitude 사용 - SQLite 호환)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)

    # PostGIS geometry는 PostgreSQL로 전환 시 추가
    # 현재는 latitude/longitude만 사용

    description = Column(Text)
    address = Column(String(500))
    osm_id = Column(BigInteger)

    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Landmark(id={self.id}, name={self.name}, category={self.category})>"

