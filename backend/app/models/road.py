"""도로 모델"""
from sqlalchemy import Column, String, Float, BigInteger, DateTime
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

from app.database import Base


class Road(Base):
    """도로 네트워크 모델"""
    __tablename__ = "roads"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    osm_id = Column(BigInteger, unique=True, index=True)
    name = Column(String(200), nullable=True)

    # PostGIS geometry는 PostgreSQL로 전환 시 추가
    # 현재는 SQLite 호환을 위해 제거

    road_type = Column(String(50), index=True)  # highway, primary, secondary 등
    length_km = Column(Float)  # 길이 (km)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Road(id={self.id}, name={self.name}, type={self.road_type})>"

