"""안전 체크인 모델"""
from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from app.database import Base


class SafetyCheckin(Base):
    """안전 체크인"""
    __tablename__ = "safety_checkins"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    route_id = Column(Integer)
    estimated_arrival_time = Column(DateTime(timezone=True), nullable=False, index=True)
    destination_lat = Column(Float)
    destination_lon = Column(Float)
    status = Column(String(20), default='active', index=True)  # active, confirmed, missed, cancelled
    confirmed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<SafetyCheckin(id={self.id}, user_id={self.user_id}, status='{self.status}')>"
