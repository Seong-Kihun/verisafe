"""크라우드소싱 신뢰도 평가 시스템"""
from datetime import datetime, timedelta
from typing import Dict, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models.hazard import Hazard
from app.models.user import User


class TrustScorer:
    """
    크라우드소싱 신뢰도 평가 시스템

    기능:
    - 사용자 제보 자동 검증
    - 신뢰도 점수 계산 (0-100)
    - 허위 정보 필터링
    - 사용자 평판 관리

    Phase 3 구현
    """

    # 신뢰도 점수 가중치
    WEIGHTS = {
        "user_reputation": 0.3,      # 사용자 평판
        "data_consistency": 0.25,    # 데이터 일관성
        "cross_validation": 0.20,    # 교차 검증
        "timeliness": 0.15,          # 시의성
        "completeness": 0.10         # 완전성
    }

    def __init__(self):
        """신뢰도 평가 시스템 초기화"""
        print("[TrustScorer] 크라우드소싱 신뢰도 평가 시스템 초기화")

    def calculate_trust_score(
        self,
        hazard_dict: Dict,
        user: Optional[User],
        db: Session
    ) -> int:
        """
        종합 신뢰도 점수 계산 (0-100)

        Args:
            hazard_dict: Hazard 데이터 딕셔너리
            user: 제보한 사용자 (None이면 익명/시스템)
            db: Database session

        Returns:
            신뢰도 점수 (0-100)
        """
        scores = {}

        # 1. 사용자 평판 점수
        scores["user_reputation"] = self._calculate_user_reputation(user, db)

        # 2. 데이터 일관성 점수
        scores["data_consistency"] = self._check_data_consistency(hazard_dict)

        # 3. 교차 검증 점수
        scores["cross_validation"] = self._cross_validate(hazard_dict, db)

        # 4. 시의성 점수
        scores["timeliness"] = self._assess_timeliness(hazard_dict)

        # 5. 완전성 점수
        scores["completeness"] = self._assess_completeness(hazard_dict)

        # 가중 평균 계산
        trust_score = sum(
            scores[key] * self.WEIGHTS[key]
            for key in scores.keys()
        )

        return int(trust_score)

    def is_likely_spam(
        self,
        hazard_dict: Dict,
        user: Optional[User],
        db: Session
    ) -> bool:
        """
        스팸/허위 정보 판별

        Returns:
            True if likely spam, False otherwise
        """
        # 1. 신뢰도 점수가 너무 낮음
        trust_score = self.calculate_trust_score(hazard_dict, user, db)
        if trust_score < 20:
            return True

        # 2. 사용자가 단시간에 너무 많은 제보
        if user and self._is_flooding(user, db):
            return True

        # 3. 의심스러운 패턴
        if self._has_suspicious_pattern(hazard_dict):
            return True

        return False

    def update_user_reputation(
        self,
        user: User,
        hazard: Hazard,
        feedback: str,
        db: Session
    ):
        """
        사용자 평판 업데이트

        Args:
            user: 사용자
            hazard: 사용자가 제보한 위험
            feedback: 피드백 ("confirmed", "disputed", "false")
            db: Database session
        """
        if not user:
            return

        # 피드백에 따라 평판 조정
        reputation_change = {
            "confirmed": +5,    # 검증됨
            "disputed": -1,     # 논쟁의 여지
            "false": -10        # 허위
        }.get(feedback, 0)

        # 사용자 평판 업데이트 (가상 필드, 실제로는 별도 테이블 필요)
        # user.reputation = max(0, min(100, user.reputation + reputation_change))
        # db.commit()

        print(f"[TrustScorer] 사용자 {user.id} 평판 변경: {reputation_change}")

    def _calculate_user_reputation(
        self,
        user: Optional[User],
        db: Session
    ) -> float:
        """
        사용자 평판 점수 계산 (0-100)

        새 사용자: 50점 (중립)
        검증된 사용자: 과거 제보 정확도 기반
        """
        if not user:
            # 익명 제보: 낮은 초기 점수
            return 30.0

        # TODO: 실제로는 User 테이블에 reputation 필드 추가 필요
        # 여기서는 간단한 휴리스틱 사용

        # 사용자의 과거 제보 수
        user_reports_count = db.query(func.count(Hazard.id)).filter(
            Hazard.reported_by == user.id
        ).scalar() or 0

        if user_reports_count == 0:
            # 신규 사용자
            return 50.0

        # 검증된 제보 수
        verified_count = db.query(func.count(Hazard.id)).filter(
            Hazard.reported_by == user.id,
            Hazard.verified == True
        ).scalar() or 0

        # 정확도 계산
        accuracy = verified_count / user_reports_count if user_reports_count > 0 else 0.5

        # 50 + 정확도 * 50 (50-100 범위)
        reputation = 50 + (accuracy * 50)

        # 제보 수 보너스 (경험)
        experience_bonus = min(10, user_reports_count * 0.5)

        return min(100, reputation + experience_bonus)

    def _check_data_consistency(self, hazard_dict: Dict) -> float:
        """
        데이터 일관성 점수 (0-100)

        필드 간 일관성 확인
        """
        score = 100.0

        # 1. 위험도와 위험 유형 일관성
        risk_score = hazard_dict.get("risk_score", 50)
        hazard_type = hazard_dict.get("hazard_type", "other")

        # 분쟁/갈등은 일반적으로 높은 위험도
        if hazard_type == "conflict" and risk_score < 50:
            score -= 20

        # 안전 피난처는 낮은 위험도여야 함
        if hazard_type == "safe_haven" and risk_score > 30:
            score -= 30

        # 2. 위치 타당성 (남수단 범위)
        lat = hazard_dict.get("latitude", 0)
        lon = hazard_dict.get("longitude", 0)

        # 남수단 대략 위도 3-12, 경도 24-36
        if not (3 <= lat <= 12 and 24 <= lon <= 36):
            score -= 40  # 범위 벗어남

        # 3. 설명 길이 (너무 짧거나 없으면 감점)
        description = hazard_dict.get("description", "")
        if len(description) < 10:
            score -= 15

        return max(0, score)

    def _cross_validate(self, hazard_dict: Dict, db: Session) -> float:
        """
        교차 검증 점수 (0-100)

        비슷한 시간/위치의 다른 제보와 비교
        """
        lat = hazard_dict.get("latitude")
        lon = hazard_dict.get("longitude")

        if not lat or not lon:
            return 50.0  # 중립

        # 최근 24시간, 반경 5km 내 유사 제보 검색
        recent_time = datetime.utcnow() - timedelta(hours=24)

        # 간단한 거리 계산 (1도 ≈ 111km)
        lat_delta = 5 / 111.0
        lon_delta = 5 / 111.0

        similar_hazards = db.query(Hazard).filter(
            Hazard.created_at >= recent_time,
            Hazard.latitude.between(lat - lat_delta, lat + lat_delta),
            Hazard.longitude.between(lon - lon_delta, lon + lon_delta),
            Hazard.hazard_type == hazard_dict.get("hazard_type")
        ).count()

        if similar_hazards == 0:
            # 첫 제보 - 중립 점수
            return 50.0
        elif similar_hazards == 1:
            # 1건 더 있음 - 약간 높은 신뢰도
            return 70.0
        elif similar_hazards >= 2:
            # 2건 이상 - 높은 신뢰도 (교차 검증됨)
            return 90.0

        return 50.0

    def _assess_timeliness(self, hazard_dict: Dict) -> float:
        """
        시의성 점수 (0-100)

        얼마나 최신 정보인가
        """
        start_date = hazard_dict.get("start_date")

        if not start_date:
            return 50.0

        # start_date가 datetime 객체인지 확인
        if isinstance(start_date, str):
            try:
                start_date = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
            except:
                return 50.0

        # 현재 시각과 차이
        now = datetime.utcnow()
        if start_date.tzinfo:
            # timezone-aware datetime을 UTC로 변환
            now = now.replace(tzinfo=start_date.tzinfo)

        time_diff = abs((now - start_date).total_seconds())

        # 시간 차이에 따른 점수
        if time_diff < 3600:  # 1시간 이내
            return 100.0
        elif time_diff < 3600 * 6:  # 6시간 이내
            return 90.0
        elif time_diff < 3600 * 24:  # 24시간 이내
            return 70.0
        elif time_diff < 3600 * 72:  # 3일 이내
            return 50.0
        else:
            return 30.0  # 너무 오래됨

    def _assess_completeness(self, hazard_dict: Dict) -> float:
        """
        완전성 점수 (0-100)

        얼마나 많은 정보가 제공되었는가
        """
        score = 0.0

        # 필수 필드 체크
        required_fields = ["hazard_type", "latitude", "longitude", "risk_score"]
        for field in required_fields:
            if field in hazard_dict and hazard_dict[field]:
                score += 15

        # 선택 필드 체크
        optional_fields = ["description", "radius", "start_date", "end_date"]
        for field in optional_fields:
            if field in hazard_dict and hazard_dict[field]:
                score += 10

        return min(100, score)

    def _is_flooding(self, user: User, db: Session) -> bool:
        """
        플러딩 감지 (단시간 과다 제보)

        Returns:
            True if flooding detected
        """
        # 최근 1시간 제보 수
        recent_hour = datetime.utcnow() - timedelta(hours=1)

        report_count = db.query(func.count(Hazard.id)).filter(
            Hazard.reported_by == user.id,
            Hazard.created_at >= recent_hour
        ).scalar() or 0

        # 1시간에 10건 이상 = 플러딩
        return report_count >= 10

    def _has_suspicious_pattern(self, hazard_dict: Dict) -> bool:
        """
        의심스러운 패턴 감지

        Returns:
            True if suspicious
        """
        # 1. 위험도가 너무 극단적 (0 or 100)
        risk_score = hazard_dict.get("risk_score", 50)
        if risk_score in [0, 100]:
            return True

        # 2. 설명이 없거나 너무 짧음
        description = hazard_dict.get("description", "")
        if len(description) < 5:
            return True

        # 3. 반복되는 동일 위치 (소수점 정확히 같음 - 복붙 가능성)
        # TODO: 데이터베이스에서 동일 좌표 빈도 확인

        return False

    def get_trust_breakdown(
        self,
        hazard_dict: Dict,
        user: Optional[User],
        db: Session
    ) -> Dict:
        """
        신뢰도 점수 상세 분석

        각 요소별 점수 반환
        """
        scores = {
            "user_reputation": self._calculate_user_reputation(user, db),
            "data_consistency": self._check_data_consistency(hazard_dict),
            "cross_validation": self._cross_validate(hazard_dict, db),
            "timeliness": self._assess_timeliness(hazard_dict),
            "completeness": self._assess_completeness(hazard_dict)
        }

        # 가중 평균
        total_score = sum(
            scores[key] * self.WEIGHTS[key]
            for key in scores.keys()
        )

        return {
            "total_score": int(total_score),
            "breakdown": {
                key: {
                    "score": round(scores[key], 1),
                    "weight": self.WEIGHTS[key],
                    "contribution": round(scores[key] * self.WEIGHTS[key], 1)
                }
                for key in scores.keys()
            },
            "is_spam": self.is_likely_spam(hazard_dict, user, db)
        }
