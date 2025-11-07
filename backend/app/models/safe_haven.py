"""안전 대피처 모델"""
from sqlalchemy import Column, Integer, String, Float, Boolean, Text, DateTime
from sqlalchemy.sql import func
from app.database import Base


class SafeHaven(Base):
    """안전 대피처 (대사관, 병원, UN 시설, 경찰서, 안전 호텔, 대피소)"""
    __tablename__ = "safe_havens"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    category = Column(String(50), nullable=False, index=True)  # embassy, hospital, un, police, hotel, shelter
    latitude = Column(Float, nullable=False, index=True)
    longitude = Column(Float, nullable=False, index=True)
    address = Column(Text)
    phone = Column(String(50))
    hours = Column(Text)  # Operating hours
    capacity = Column(Integer)  # Max people for shelters
    verified = Column(Boolean, default=False)  # Verified by admins
    notes = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<SafeHaven(id={self.id}, name='{self.name}', category='{self.category}')>"
