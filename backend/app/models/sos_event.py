"""SOS 이벤트 모델"""
from sqlalchemy import Column, Integer, String, Float, Text, DateTime
from sqlalchemy.sql import func
from app.database import Base


class SOSEvent(Base):
    """SOS 긴급 이벤트"""
    __tablename__ = "sos_events"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    message = Column(Text)
    status = Column(String(20), default='active', index=True)  # active, resolved, cancelled
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    resolved_at = Column(DateTime(timezone=True))

    def __repr__(self):
        return f"<SOSEvent(id={self.id}, user_id={self.user_id}, status='{self.status}')>"
