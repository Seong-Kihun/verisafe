"""포인트 관리 시스템"""
from typing import Dict, Optional
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from app.models.user import User


class PointsManager:
    """
    인센티브 포인트 관리 시스템
    
    Phase 5 구현:
    - 활동별 포인트 지급
    - 레벨 시스템
    - 리더보드
    """
    
    # 활동별 포인트 규칙
    POINTS_RULES = {
        "report_submit": 10,      # 제보 등록
        "report_verify": 5,       # 제보 검증 (관리자/매퍼)
        "captcha_complete": 2,    # CAPTCHA 완료
        "contribution": 15,       # 크라우드소싱 기여
        "daily_login": 1,         # 일일 로그인
        "first_report": 50,       # 첫 제보 보너스
    }
    
    # 레벨별 필요 포인트
    LEVEL_THRESHOLDS = [
        0, 100, 300, 600, 1000, 1500, 2100, 2800, 3600, 4500, 5500
    ]
    
    def __init__(self):
        pass
    
    def award_points(self, db: Session, user_id: str, action_type: str, 
                    description: str = "") -> Dict:
        """
        포인트 지급
        
        Args:
            db: Database session
            user_id: 사용자 ID
            action_type: 활동 타입
            description: 설명
        
        Returns:
            포인트 지급 결과
        """
        points = self.POINTS_RULES.get(action_type, 0)
        
        if points == 0:
            return {"status": "error", "message": f"알 수 없는 활동: {action_type}"}
        
        try:
            # 사용자 포인트 모델 (가상)
            # 실제로는 UserPoints 모델 사용
            print(f"[PointsManager] {user_id}에게 {points}포인트 지급 ({action_type})")
            
            # TODO: UserPoints 모델에 포인트 추가
            # user_points = db.query(UserPoints).filter_by(user_id=user_id).first()
            # if not user_points:
            #     user_points = UserPoints(user_id=user_id)
            #     db.add(user_points)
            # user_points.total_points += points
            
            # TODO: PointsHistory에 기록
            # history = PointsHistory(user_id=user_id, action_type=action_type, points=points, description=description)
            # db.add(history)
            
            # db.commit()
            
            return {
                "status": "success",
                "user_id": user_id,
                "action": action_type,
                "points_awarded": points,
                "description": description
            }
            
        except Exception as e:
            print(f"[PointsManager] 포인트 지급 오류: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_user_points(self, db: Session, user_id: str) -> Dict:
        """
        사용자 포인트 조회
        
        Args:
            db: Database session
            user_id: 사용자 ID
        
        Returns:
            포인트 정보
        """
        # TODO: 실제 DB에서 조회
        # user_points = db.query(UserPoints).filter_by(user_id=user_id).first()
        
        # 더미 데이터
        total_points = 350
        level = self.calculate_level(total_points)
        next_level_points = self.LEVEL_THRESHOLDS[level] if level < len(self.LEVEL_THRESHOLDS) - 1 else total_points
        
        return {
            "user_id": user_id,
            "total_points": total_points,
            "level": level,
            "next_level_at": next_level_points,
            "progress_to_next_level": (total_points / next_level_points * 100) if next_level_points > 0 else 100
        }
    
    def calculate_level(self, total_points: int) -> int:
        """
        포인트로부터 레벨 계산
        
        Args:
            total_points: 총 포인트
        
        Returns:
            레벨 (1부터 시작)
        """
        for level, threshold in enumerate(self.LEVEL_THRESHOLDS):
            if total_points < threshold:
                return level
        
        return len(self.LEVEL_THRESHOLDS)
    
    def get_leaderboard(self, db: Session, limit: int = 10) -> List[Dict]:
        """
        리더보드 조회
        
        Args:
            db: Database session
            limit: 상위 N명
        
        Returns:
            리더보드
        """
        # TODO: 실제 DB에서 조회
        # leaderboard = db.query(UserPoints, User).join(User).order_by(UserPoints.total_points.desc()).limit(limit).all()
        
        # 더미 데이터
        dummy_leaderboard = [
            {"rank": 1, "username": "mapper_alice", "points": 2500, "level": 7},
            {"rank": 2, "username": "reporter_bob", "points": 1800, "level": 6},
            {"rank": 3, "username": "helper_charlie", "points": 1200, "level": 5},
        ]
        
        return dummy_leaderboard


# 싱글톤
points_manager = PointsManager()
