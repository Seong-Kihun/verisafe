"""사용자 포인트 모델"""
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime

from app.database import Base


class UserPoints(Base):
    """사용자 포인트 현황"""
    __tablename__ = "user_points"
    
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), primary_key=True)
    total_points = Column(Integer, default=0)
    level = Column(Integer, default=1)
    
    # 활동별 포인트
    reports_submitted = Column(Integer, default=0)
    reports_verified = Column(Integer, default=0)
    captcha_completed = Column(Integer, default=0)
    contributions = Column(Integer, default=0)
    
    last_activity = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = relationship("User", backref="points")


class PointsHistory(Base):
    """포인트 획득 이력"""
    __tablename__ = "points_history"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    
    action_type = Column(String(50))  # 'report_submit', 'report_verify', 'captcha', etc.
    points = Column(Integer)
    description = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    user = relationship("User", backref="points_history")
