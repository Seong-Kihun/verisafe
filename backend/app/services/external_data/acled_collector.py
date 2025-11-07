"""ACLED API 수집기 - 분쟁 및 폭력 사건 데이터"""
import httpx
from datetime import datetime, timedelta
from typing import List, Optional
from sqlalchemy.orm import Session

from app.models.hazard import Hazard
from app.config import settings


class ACLEDCollector:
    """
    ACLED (Armed Conflict Location & Event Data Project) API 연동
    
    데이터 소스: https://acleddata.com/
    - 정치적 폭력, 시위, 분쟁 사건
    - 남수단 지역 사건 정보
    
    Phase 2 구현
    """
    
    BASE_URL = "https://api.acleddata.com/acled/read"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: ACLED API 키 (settings.acled_api_key 사용 또는 명시적 제공)
        """
        self.api_key = api_key or settings.acled_api_key
        
    async def collect_recent_events(self, db: Session, country: str = "South Sudan", days: int = 7) -> int:
        """
        최근 N일간의 사건을 수집하여 DB에 저장
        
        Args:
            db: Database session
            country: 국가명 (기본값: "South Sudan")
            days: 수집할 일수 (기본값: 7일)
            
        Returns:
            수집된 사건 수
        """
        if not self.api_key:
            print("[ACLEDCollector] 경고: API 키가 설정되지 않았습니다. 더미 데이터를 생성합니다.")
            return await self._create_dummy_data(db)
        
        print(f"[ACLEDCollector] {country} 최근 {days}일 데이터 수집 중...")
        
        # 날짜 범위 계산
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        params = {
            "key": self.api_key,
            "country": country,
            "event_date": f"{start_date.strftime('%Y-%m-%d')}|{end_date.strftime('%Y-%m-%d')}",
            "event_date_where": "BETWEEN",
            "limit": 500
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(self.BASE_URL, params=params)
                response.raise_for_status()
                data = response.json()
            
            if "data" not in data:
                print("[ACLEDCollector] 응답에 데이터가 없습니다.")
                return 0
            
            events = data["data"]
            print(f"[ACLEDCollector] {len(events)}개 사건 발견")
            
            # Hazard 모델로 변환 및 저장
            count = 0
            for event in events:
                try:
                    hazard = self._convert_to_hazard(event)
                    
                    # 중복 확인 (동일 위치, 동일 날짜)
                    existing = db.query(Hazard).filter(
                        Hazard.latitude == hazard.latitude,
                        Hazard.longitude == hazard.longitude,
                        Hazard.source == "acled",
                        Hazard.start_date == hazard.start_date
                    ).first()
                    
                    if not existing:
                        db.add(hazard)
                        count += 1
                        
                except Exception as e:
                    print(f"[ACLEDCollector] 사건 변환 오류: {e}")
                    continue
            
            db.commit()
            print(f"[ACLEDCollector] {count}개 새 사건 저장 완료")
            return count
            
        except httpx.HTTPError as e:
            print(f"[ACLEDCollector] API 오류: {e}")
            return 0
    
    def _convert_to_hazard(self, event: dict) -> Hazard:
        """
        ACLED 사건을 Hazard 모델로 변환
        
        ACLED 이벤트 타입별 위험도 매핑:
        - Battles: 80-100
        - Explosions/Remote violence: 70-90
        - Violence against civilians: 75-95
        - Protests: 30-60
        - Riots: 50-70
        - Strategic developments: 20-40
        """
        event_type = event.get("event_type", "").lower()
        fatalities = int(event.get("fatalities", 0))
        
        # 이벤트 타입별 기본 위험도
        risk_map = {
            "battles": 80,
            "explosions/remote violence": 70,
            "violence against civilians": 75,
            "riots": 50,
            "protests": 30,
            "strategic developments": 20
        }
        
        base_risk = risk_map.get(event_type, 40)
        
        # 사망자 수로 위험도 조정 (+5점 per fatality, 최대 100)
        risk_score = min(100, base_risk + (fatalities * 5))
        
        # hazard_type 매핑
        hazard_type_map = {
            "battles": "conflict",
            "explosions/remote violence": "conflict",
            "violence against civilians": "conflict",
            "riots": "protest",
            "protests": "protest",
            "strategic developments": "other"
        }
        
        hazard_type = hazard_type_map.get(event_type, "other")
        
        # 날짜 파싱
        event_date_str = event.get("event_date", "")
        try:
            event_date = datetime.strptime(event_date_str, "%Y-%m-%d")
        except:
            event_date = datetime.utcnow()
        
        # 지속 기간: 분쟁은 72시간, 시위는 24시간
        duration_hours = 72 if hazard_type == "conflict" else 24
        end_date = event_date + timedelta(hours=duration_hours)
        
        return Hazard(
            hazard_type=hazard_type,
            risk_score=risk_score,
            latitude=float(event.get("latitude", 0)),
            longitude=float(event.get("longitude", 0)),
            radius=2.0 if hazard_type == "conflict" else 1.0,  # km
            source="acled",
            description=f"{event.get('event_type')}: {event.get('notes', '')[:200]}",
            start_date=event_date,
            end_date=end_date,
            verified=True  # ACLED 데이터는 검증된 것으로 간주
        )
    
    async def _create_dummy_data(self, db: Session) -> int:
        """
        API 키가 없을 때 더미 데이터 생성 (개발/테스트용) - 더 현실적인 데이터
        중복 방지: 기존 더미 데이터를 먼저 삭제 후 재생성
        """
        print("[ACLEDCollector] 더미 분쟁 데이터 생성 중...")

        import random

        # 기존 더미 데이터 삭제 (중복 방지)
        existing_dummy = db.query(Hazard).filter(Hazard.source == "acled_dummy").all()
        if existing_dummy:
            for hazard in existing_dummy:
                db.delete(hazard)
            db.commit()
            print(f"[ACLEDCollector] 기존 더미 데이터 {len(existing_dummy)}개 삭제")

        # 주바 주변 좌표 생성 (±0.2도 범위)
        base_lat, base_lng = 4.8594, 31.5713

        event_types = [
            ("conflict", 75, 2.0, "Armed conflict in {area}"),
            ("conflict", 85, 1.5, "Battle reported near {area}"),
            ("protest", 40, 0.8, "Peaceful protest in {area}"),
            ("protest", 55, 1.0, "Demonstration near {area}"),
            ("conflict", 70, 2.5, "Violence against civilians in {area}"),
            ("protest", 50, 1.2, "Riot reported in {area}"),
        ]

        areas = ["northern district", "city center", "marketplace", "border area", "residential zone", "checkpoint"]

        dummy_events = []
        for i, (htype, risk, radius, desc_template) in enumerate(event_types):
            lat_offset = random.uniform(-0.15, 0.15)
            lng_offset = random.uniform(-0.15, 0.15)
            area = random.choice(areas)

            dummy_events.append({
                "latitude": base_lat + lat_offset,
                "longitude": base_lng + lng_offset,
                "hazard_type": htype,
                "risk_score": risk + random.randint(-10, 10),
                "description": f"{desc_template.format(area=area)} (Test data #{i+1})",
                "radius": radius
            })

        count = 0
        for event_data in dummy_events:
            # 최근 7일 내 랜덤 날짜
            hours_ago = random.randint(1, 168)  # 7 days

            hazard = Hazard(
                source="acled_dummy",
                start_date=datetime.utcnow() - timedelta(hours=hours_ago),
                end_date=datetime.utcnow() + timedelta(hours=72 - hours_ago),  # 72시간 지속
                verified=True,  # 더미 데이터를 실제 위험처럼 인식
                **event_data
            )
            db.add(hazard)
            count += 1

        db.commit()
        print(f"[ACLEDCollector] {count}개 더미 사건 생성 완료")
        return count
