"""시간대별 위험도 승수 계산기"""
from datetime import datetime, timedelta
from typing import Dict, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from app.models.hazard import Hazard
from app.services.redis_manager import redis_manager


class TimeMultiplierCalculator:
    """
    시간대별 위험도 승수 계산
    
    Phase 3 구현:
    - 요일별, 시간대별 위험 패턴 분석
    - 과거 데이터 기반 승수 계산
    - 예: 금요일 17시 = 1.5배 위험
    """
    
    def __init__(self):
        self.multipliers: Dict[Tuple[int, int], float] = {}  # (day_of_week, hour) -> multiplier
        self.cache_key = "time_multipliers"
    
    async def calculate_and_cache(self, db: Session, days_back: int = 30) -> Dict[str, float]:
        """
        과거 데이터를 분석하여 시간대별 승수 계산 및 캐싱
        
        Args:
            db: Database session
            days_back: 분석할 과거 기간 (일)
        
        Returns:
            시간대별 승수 딕셔너리
        """
        print(f"[TimeMultiplier] 과거 {days_back}일 데이터 분석 중...")
        
        # 캐시 확인
        cached = redis_manager.get(self.cache_key)
        if cached:
            print("[TimeMultiplier] 캐시에서 로드됨")
            self.multipliers = {eval(k): v for k, v in cached.items()}
            return cached
        
        # 과거 데이터 조회
        start_date = datetime.utcnow() - timedelta(days=days_back)
        
        # 요일별, 시간대별 평균 위험도 계산
        hazards = db.query(
            func.extract('dow', Hazard.start_date).label('day_of_week'),
            func.extract('hour', Hazard.start_date).label('hour'),
            func.avg(Hazard.risk_score).label('avg_risk'),
            func.count(Hazard.id).label('count')
        ).filter(
            Hazard.start_date >= start_date
        ).group_by(
            'day_of_week', 'hour'
        ).all()
        
        if not hazards or len(hazards) == 0:
            print("[TimeMultiplier] 데이터 부족 - 기본 승수 사용")
            return self._get_default_multipliers()
        
        # 전체 평균 위험도 계산
        total_avg = db.query(func.avg(Hazard.risk_score)).filter(
            Hazard.start_date >= start_date
        ).scalar() or 50.0
        
        print(f"[TimeMultiplier] 전체 평균 위험도: {total_avg:.2f}")
        
        # 시간대별 승수 계산
        multipliers = {}
        
        for dow, hour, avg_risk, count in hazards:
            if count < 2:  # 데이터 부족한 시간대 제외
                continue
            
            # 승수 = 시간대 평균 / 전체 평균
            multiplier = (avg_risk or total_avg) / total_avg
            
            # 0.5 ~ 2.0 범위로 제한
            multiplier = max(0.5, min(2.0, multiplier))
            
            key = f"{int(dow)}_{int(hour)}"
            multipliers[key] = round(multiplier, 2)
            
            # 메모리에도 저장
            self.multipliers[(int(dow), int(hour))] = round(multiplier, 2)
        
        # 기본값으로 채우기 (데이터 없는 시간대)
        for dow in range(7):
            for hour in range(24):
                key = f"{dow}_{hour}"
                if key not in multipliers:
                    multipliers[key] = 1.0
                    self.multipliers[(dow, hour)] = 1.0
        
        # Redis에 캐싱 (24시간)
        redis_manager.set(self.cache_key, multipliers, ttl=86400)
        
        print(f"[TimeMultiplier] {len(multipliers)}개 시간대 승수 계산 완료")
        return multipliers
    
    def _get_default_multipliers(self) -> Dict[str, float]:
        """
        데이터 부족 시 사용할 기본 승수 (도메인 지식 기반)
        
        패턴:
        - 금요일 저녁 (17-20시): 1.5배
        - 주말 낮 (10-16시): 0.8배
        - 심야 (22-05시): 1.3배
        - 평일 업무시간 (09-17시): 1.1배
        """
        multipliers = {}
        
        for dow in range(7):  # 0=일, 1=월, ..., 6=토
            for hour in range(24):
                base = 1.0
                
                # 금요일 (5) 저녁
                if dow == 5 and 17 <= hour <= 20:
                    base = 1.5
                # 주말 낮
                elif dow in [0, 6] and 10 <= hour <= 16:
                    base = 0.8
                # 심야
                elif hour >= 22 or hour <= 5:
                    base = 1.3
                # 평일 업무시간
                elif dow in [1, 2, 3, 4, 5] and 9 <= hour <= 17:
                    base = 1.1
                
                key = f"{dow}_{hour}"
                multipliers[key] = base
                self.multipliers[(dow, hour)] = base
        
        # Redis에 캐싱
        redis_manager.set(self.cache_key, multipliers, ttl=86400)
        
        return multipliers
    
    def get_multiplier(self, timestamp: datetime) -> float:
        """
        특정 시간의 위험도 승수 조회
        
        Args:
            timestamp: 조회할 시간
        
        Returns:
            위험도 승수 (0.5 ~ 2.0)
        """
        dow = timestamp.weekday()  # 0=월, 6=일
        hour = timestamp.hour
        
        # PostgreSQL의 dow는 0=일요일, Python의 weekday는 0=월요일
        # 변환: Python weekday -> PostgreSQL dow
        pg_dow = (dow + 1) % 7
        
        multiplier = self.multipliers.get((pg_dow, hour))
        
        if multiplier is None:
            # 캐시에 없으면 기본값 1.0
            return 1.0
        
        return multiplier
    
    def get_multipliers_for_range(self, start: datetime, hours: int = 24) -> list:
        """
        시간 범위의 승수 조회
        
        Args:
            start: 시작 시간
            hours: 조회할 시간 수
        
        Returns:
            [{timestamp, multiplier}, ...]
        """
        results = []
        
        for hour_offset in range(hours):
            timestamp = start + timedelta(hours=hour_offset)
            multiplier = self.get_multiplier(timestamp)
            
            results.append({
                "timestamp": timestamp.isoformat(),
                "hour": timestamp.hour,
                "day_of_week": timestamp.strftime("%A"),
                "multiplier": multiplier
            })
        
        return results
    
    def apply_to_risk_score(self, base_risk: float, timestamp: datetime) -> float:
        """
        기본 위험도에 시간 승수 적용
        
        Args:
            base_risk: 기본 위험도 (0-100)
            timestamp: 시간
        
        Returns:
            조정된 위험도 (0-100)
        """
        multiplier = self.get_multiplier(timestamp)
        adjusted_risk = base_risk * multiplier
        
        return min(100, max(0, adjusted_risk))


# 싱글톤 인스턴스
time_multiplier_calculator = TimeMultiplierCalculator()
