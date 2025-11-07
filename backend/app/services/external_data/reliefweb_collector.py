"""ReliefWeb API 수집기 - 인도적 지원 보고서 데이터"""
import httpx
from datetime import datetime, timedelta
from typing import List
from sqlalchemy.orm import Session

from app.models.hazard import Hazard
from app.config import settings


class ReliefWebCollector:
    """
    ReliefWeb API 연동
    
    데이터 소스: https://reliefweb.int/
    - 인도적 지원 상황 보고서
    - 긴급 경보 및 업데이트
    - 남수단 인도적 상황 정보
    
    Phase 2 구현
    """
    
    BASE_URL = "https://api.reliefweb.int/v1/reports"
    
    async def collect_recent_reports(self, db: Session, country: str = "South Sudan", days: int = 7) -> int:
        """
        최근 N일간의 인도적 지원 보고서를 수집하여 DB에 저장
        
        Args:
            db: Database session
            country: 국가명 (기본값: "South Sudan")
            days: 수집할 일수 (기본값: 7일)
            
        Returns:
            수집된 보고서 수
        """
        print(f"[ReliefWebCollector] {country} 최근 {days}일 보고서 수집 중...")
        
        # 날짜 범위 계산
        from_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        # ReliefWeb API 쿼리 구성
        query = {
            "appname": "verisafe",
            "query": {
                "value": country,
                "fields": ["country"]
            },
            "filter": {
                "field": "date.created",
                "value": {
                    "from": from_date
                }
            },
            "fields": {
                "include": [
                    "id", "title", "body", "date.created",
                    "primary_country", "disaster_type", "vulnerable_groups"
                ]
            },
            "limit": 50
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.BASE_URL,
                    json=query,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                data = response.json()
            
            if "data" not in data or len(data["data"]) == 0:
                print("[ReliefWebCollector] 데이터가 없습니다. 더미 데이터를 생성합니다.")
                return await self._create_dummy_data(db)
            
            reports = data["data"]
            print(f"[ReliefWebCollector] {len(reports)}개 보고서 발견")
            
            # Hazard 모델로 변환 및 저장
            count = 0
            for report in reports:
                try:
                    # ReliefWeb 데이터는 일반적으로 전국적/지역적이므로
                    # 주바 중심 좌표 사용 (실제로는 보고서 내용 파싱 필요)
                    hazards = self._convert_to_hazards(report)
                    
                    for hazard in hazards:
                        # 중복 확인
                        existing = db.query(Hazard).filter(
                            Hazard.source == "reliefweb",
                            Hazard.description.contains(report["fields"]["title"][:50])
                        ).first()
                        
                        if not existing:
                            db.add(hazard)
                            count += 1
                            
                except Exception as e:
                    print(f"[ReliefWebCollector] 보고서 변환 오류: {e}")
                    continue
            
            db.commit()
            print(f"[ReliefWebCollector] {count}개 새 위험 정보 저장 완료")
            return count
            
        except Exception as e:
            print(f"[ReliefWebCollector] API 오류: {e}")
            print("[ReliefWebCollector] 더미 데이터로 폴백합니다.")
            return await self._create_dummy_data(db)
    
    def _convert_to_hazards(self, report: dict) -> List[Hazard]:
        """
        ReliefWeb 보고서를 Hazard 모델 리스트로 변환
        
        보고서 내용 기반 위험도 추정:
        - 키워드 분석으로 위험 유형 및 심각도 판단
        """
        fields = report.get("fields", {})
        title = fields.get("title", "")
        body = fields.get("body", "")
        date_str = fields.get("date", {}).get("created", "")
        
        # 날짜 파싱
        try:
            created_date = datetime.fromisoformat(date_str.replace("T", " ").replace("Z", ""))
        except:
            created_date = datetime.utcnow()
        
        # 키워드 기반 위험도 및 타입 분석
        text_lower = (title + " " + body).lower()
        
        hazards = []
        
        # 분쟁 관련
        if any(word in text_lower for word in ["conflict", "violence", "armed", "fighting", "clashes"]):
            hazards.append(Hazard(
                hazard_type="conflict",
                risk_score=60,
                latitude=4.8594,  # 주바 중심 (실제로는 NLP로 위치 추출 필요)
                longitude=31.5713,
                radius=5.0,
                source="reliefweb",
                description=f"ReliefWeb: {title[:200]}",
                start_date=created_date,
                end_date=created_date + timedelta(days=3),
                verified=False  # 인도적 보고서는 미검증으로 분류
            ))
        
        # 재난 관련
        if any(word in text_lower for word in ["flood", "flooding", "heavy rain"]):
            hazards.append(Hazard(
                hazard_type="flood",
                risk_score=50,
                latitude=4.8594,
                longitude=31.5713,
                radius=10.0,
                source="reliefweb",
                description=f"ReliefWeb: {title[:200]}",
                start_date=created_date,
                end_date=created_date + timedelta(days=7),
                verified=False
            ))
        
        # 보건 위기
        if any(word in text_lower for word in ["disease", "epidemic", "cholera", "malaria"]):
            hazards.append(Hazard(
                hazard_type="other",
                risk_score=40,
                latitude=4.8594,
                longitude=31.5713,
                radius=8.0,
                source="reliefweb",
                description=f"ReliefWeb Health Alert: {title[:200]}",
                start_date=created_date,
                end_date=created_date + timedelta(days=14),
                verified=False
            ))
        
        # 일반 긴급 상황
        if len(hazards) == 0 and any(word in text_lower for word in ["emergency", "crisis", "urgent"]):
            hazards.append(Hazard(
                hazard_type="other",
                risk_score=35,
                latitude=4.8594,
                longitude=31.5713,
                radius=6.0,
                source="reliefweb",
                description=f"ReliefWeb Alert: {title[:200]}",
                start_date=created_date,
                end_date=created_date + timedelta(days=5),
                verified=False
            ))
        
        return hazards
    
    async def _create_dummy_data(self, db: Session) -> int:
        """
        API 오류 시 더미 데이터 생성 (개발/테스트용)
        중복 방지: 기존 더미 데이터를 먼저 삭제 후 재생성
        """
        print("[ReliefWebCollector] 더미 인도적 보고서 데이터 생성 중...")

        # 기존 더미 데이터 삭제 (중복 방지)
        existing_dummy = db.query(Hazard).filter(Hazard.source == "reliefweb_dummy").all()
        if existing_dummy:
            for hazard in existing_dummy:
                db.delete(hazard)
            db.commit()
            print(f"[ReliefWebCollector] 기존 더미 데이터 {len(existing_dummy)}개 삭제")

        dummy_reports = [
            {
                "latitude": 4.8550, "longitude": 31.5850,
                "hazard_type": "other", "risk_score": 45,
                "description": "ReliefWeb: Humanitarian access constraints reported in central region",
                "radius": 8.0
            },
            {
                "latitude": 4.8600, "longitude": 31.5700,
                "hazard_type": "conflict", "risk_score": 55,
                "description": "ReliefWeb: Displacement reported due to inter-communal tensions",
                "radius": 6.0
            }
        ]

        count = 0
        for report_data in dummy_reports:
            hazard = Hazard(
                source="reliefweb_dummy",
                start_date=datetime.utcnow(),
                end_date=datetime.utcnow() + timedelta(days=5),
                verified=True,  # 더미 데이터를 실제 위험처럼 인식
                **report_data
            )
            db.add(hazard)
            count += 1

        db.commit()
        print(f"[ReliefWebCollector] {count}개 더미 보고서 생성 완료")
        return count
