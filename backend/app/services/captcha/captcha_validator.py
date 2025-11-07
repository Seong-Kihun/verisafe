"""CAPTCHA 검증 시스템 (크라우드소싱용)"""
from typing import Dict, Optional
from datetime import datetime, timedelta
import hashlib
import json
import secrets

from app.services.redis_manager import redis_manager


class CaptchaValidator:
    """
    CAPTCHA 검증 시스템
    
    Phase 4 구현:
    - 위성사진 기반 CAPTCHA 생성
    - 사용자 응답 검증
    - 크라우드소싱 데이터 수집
    - 스팸 방지
    """
    
    def __init__(self):
        self.captcha_ttl = 300  # 5분
        self.max_attempts = 3
    
    def create_captcha(self, task_data: Dict) -> Dict:
        """
        CAPTCHA 생성
        
        Args:
            task_data: 위성사진 분석 결과 (SatelliteImageAnalyzer)
        
        Returns:
            CAPTCHA 정보 (사용자에게 전달)
        """
        # 고유 ID 생성
        captcha_id = secrets.token_urlsafe(16)
        
        # Redis에 정답 저장 (TTL 5분)
        cache_key = f"captcha:{captcha_id}"
        cache_data = {
            "correct_answer": task_data.get("correct_answer"),
            "created_at": datetime.utcnow().isoformat(),
            "attempts": 0,
            "task_type": task_data.get("task_type")
        }
        
        redis_manager.set(cache_key, cache_data, ttl=self.captcha_ttl)
        
        # 사용자에게 반환 (정답 제외)
        return {
            "captcha_id": captcha_id,
            "question": task_data.get("question"),
            "options": task_data.get("options"),
            "task_type": task_data.get("task_type"),
            "expires_in": self.captcha_ttl
        }
    
    def validate_captcha(self, captcha_id: str, user_answer) -> Dict:
        """
        사용자 응답 검증
        
        Args:
            captcha_id: CAPTCHA ID
            user_answer: 사용자 답변
        
        Returns:
            검증 결과
        """
        cache_key = f"captcha:{captcha_id}"
        captcha_data = redis_manager.get(cache_key)
        
        if not captcha_data:
            return {
                "valid": False,
                "error": "CAPTCHA가 만료되었거나 존재하지 않습니다."
            }
        
        # 시도 횟수 확인
        attempts = captcha_data.get("attempts", 0)
        
        if attempts >= self.max_attempts:
            redis_manager.delete(cache_key)
            return {
                "valid": False,
                "error": "최대 시도 횟수를 초과했습니다."
            }
        
        # 정답 확인
        correct_answer = captcha_data.get("correct_answer")
        
        # 타입에 따라 비교
        is_correct = False
        if isinstance(correct_answer, int):
            is_correct = int(user_answer) == correct_answer
        else:
            is_correct = str(user_answer).lower() == str(correct_answer).lower()
        
        # 시도 횟수 증가
        captcha_data["attempts"] = attempts + 1
        
        if is_correct:
            # 성공 시 캐시 삭제
            redis_manager.delete(cache_key)
            
            return {
                "valid": True,
                "message": "검증 성공",
                "task_type": captcha_data.get("task_type")
            }
        else:
            # 실패 시 시도 횟수 업데이트
            redis_manager.set(cache_key, captcha_data, ttl=self.captcha_ttl)
            
            return {
                "valid": False,
                "error": "답변이 올바르지 않습니다.",
                "attempts_left": self.max_attempts - captcha_data["attempts"]
            }
    
    def generate_verification_token(self, user_id: str) -> str:
        """
        사용자 검증 완료 토큰 생성
        
        CAPTCHA 성공 후 사용자에게 발급
        제보 등록 시 이 토큰 필요
        
        Args:
            user_id: 사용자 ID
        
        Returns:
            검증 토큰
        """
        token = secrets.token_urlsafe(32)
        
        # Redis에 토큰 저장 (1시간 유효)
        cache_key = f"verification_token:{token}"
        redis_manager.set(cache_key, {"user_id": user_id}, ttl=3600)
        
        return token
    
    def verify_token(self, token: str) -> Optional[str]:
        """
        검증 토큰 확인
        
        Args:
            token: 검증 토큰
        
        Returns:
            user_id or None
        """
        cache_key = f"verification_token:{token}"
        data = redis_manager.get(cache_key)
        
        if data:
            return data.get("user_id")
        
        return None
    
    def collect_contribution(self, user_id: str, contribution_data: Dict) -> bool:
        """
        크라우드소싱 기여 데이터 수집
        
        사용자가 위성사진에서 도로/건물을 표시한 데이터 저장
        
        Args:
            user_id: 사용자 ID
            contribution_data: 기여 데이터
        
        Returns:
            성공 여부
        """
        # Redis에 임시 저장 (나중에 DB로 이관)
        cache_key = f"contribution:{user_id}:{datetime.utcnow().timestamp()}"
        
        data = {
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": contribution_data
        }
        
        redis_manager.set(cache_key, data, ttl=86400)  # 24시간
        
        print(f"[CaptchaValidator] 기여 데이터 수집: {user_id}")
        return True


# 싱글톤
captcha_validator = CaptchaValidator()
