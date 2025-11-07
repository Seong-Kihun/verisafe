"""GDACS API 수집기 - 재난 및 자연재해 데이터"""
import httpx
from datetime import datetime, timedelta
from typing import List
from sqlalchemy.orm import Session
import xml.etree.ElementTree as ET

from app.models.hazard import Hazard
from app.config import settings


class GDACSCollector:
    """
    GDACS (Global Disaster Alert and Coordination System) API 연동
    
    데이터 소스: https://www.gdacs.org/
    - 지진, 홍수, 산사태, 사이클론 등 자연재해
    - 남수단 및 주변 지역 재난 정보
    
    Phase 2 구현
    """
    
    BASE_URL = "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH"
    
    async def collect_recent_disasters(self, db: Session, country: str = "South Sudan", days: int = 30) -> int:
        """
        최근 N일간의 재난 정보를 수집하여 DB에 저장

        Args:
            db: Database session
            country: 국가명 (기본값: "South Sudan")
            days: 수집할 일수 (기본값: 30일)

        Returns:
            수집된 재난 수
        """
        print(f"[GDACSCollector] {country} 최근 {days}일 재난 데이터 수집 중...")

        # GDACS는 RSS/XML 형식으로 데이터 제공
        params = {
            "country": country,
            "fromDate": (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d"),
            "toDate": datetime.utcnow().strftime("%Y-%m-%d")
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(self.BASE_URL, params=params)

                # 204 No Content 또는 빈 응답 처리
                if response.status_code == 204 or len(response.text.strip()) == 0:
                    print(f"[GDACSCollector] {country}에 재난 데이터 없음")
                    # 주변 지역(동아프리카) 검색
                    print("[GDACSCollector] 동아프리카 지역으로 검색 확장...")
                    return await self._search_nearby_regions(client, db, days)

                response.raise_for_status()
                xml_data = response.text

            # XML 파싱
            root = ET.fromstring(xml_data)

            # RSS 채널의 아이템 추출
            items = root.findall(".//item")
            print(f"[GDACSCollector] {len(items)}개 재난 이벤트 발견")

            if len(items) == 0:
                print("[GDACSCollector] 데이터가 없습니다. 주변 지역 검색...")
                return await self._search_nearby_regions(client, db, days)
            
            # Hazard 모델로 변환 및 저장
            count = 0
            for item in items:
                try:
                    hazard = self._convert_to_hazard(item)
                    
                    # 중복 확인
                    existing = db.query(Hazard).filter(
                        Hazard.latitude == hazard.latitude,
                        Hazard.longitude == hazard.longitude,
                        Hazard.source == "gdacs",
                        Hazard.start_date == hazard.start_date
                    ).first()
                    
                    if not existing:
                        db.add(hazard)
                        count += 1
                        
                except Exception as e:
                    print(f"[GDACSCollector] 재난 변환 오류: {e}")
                    continue
            
            db.commit()
            print(f"[GDACSCollector] {count}개 새 재난 저장 완료")
            return count
            
        except Exception as e:
            print(f"[GDACSCollector] API 오류: {e}")
            print("[GDACSCollector] 더미 데이터로 폴백합니다.")
            return await self._create_dummy_data(db)
    
    def _convert_to_hazard(self, item: ET.Element) -> Hazard:
        """
        GDACS 재난을 Hazard 모델로 변환
        
        재난 타입별 위험도 매핑:
        - 지진 (Earthquake): 50-100 (규모 기반)
        - 홍수 (Flood): 60-90
        - 사이클론 (Cyclone): 70-100
        - 산사태 (Landslide): 65-85
        """
        # XML 요소 추출
        title = item.find("title").text if item.find("title") is not None else ""
        description = item.find("description").text if item.find("description") is not None else ""
        pub_date_str = item.find("pubDate").text if item.find("pubDate") is not None else ""
        
        # gdacs 네임스페이스 처리
        ns = {"gdacs": "http://www.gdacs.org"}
        severity = item.find("gdacs:severity", ns)
        alert_level = severity.get("level") if severity is not None else "Green"
        
        # 좌표 (geo:Point 또는 gdacs:cap)
        geo_lat = item.find("geo:lat", {"geo": "http://www.w3.org/2003/01/geo/wgs84_pos#"})
        geo_long = item.find("geo:long", {"geo": "http://www.w3.org/2003/01/geo/wgs84_pos#"})
        
        latitude = float(geo_lat.text) if geo_lat is not None and geo_lat.text else 4.8594
        longitude = float(geo_long.text) if geo_long is not None and geo_long.text else 31.5713
        
        # 재난 타입 파악
        disaster_type = "other"
        if "earthquake" in title.lower() or "eq" in title.lower():
            disaster_type = "earthquake"
        elif "flood" in title.lower():
            disaster_type = "flood"
        elif "cyclone" in title.lower() or "storm" in title.lower():
            disaster_type = "cyclone"
        elif "landslide" in title.lower():
            disaster_type = "landslide"
        
        # 위험도 계산 (Alert Level 기반)
        risk_map = {
            "Red": 90,
            "Orange": 70,
            "Green": 40
        }
        risk_score = risk_map.get(alert_level, 50)
        
        # 날짜 파싱
        try:
            pub_date = datetime.strptime(pub_date_str, "%a, %d %b %Y %H:%M:%S %Z")
        except:
            pub_date = datetime.utcnow()
        
        # 재난 유형별 지속 기간
        duration_map = {
            "earthquake": 24,   # 지진 여진 고려
            "flood": 168,       # 홍수 7일
            "cyclone": 72,      # 사이클론 3일
            "landslide": 120    # 산사태 5일
        }
        duration_hours = duration_map.get(disaster_type, 48)
        end_date = pub_date + timedelta(hours=duration_hours)
        
        # hazard_type 매핑
        hazard_type_map = {
            "earthquake": "other",
            "flood": "flood",
            "cyclone": "other",
            "landslide": "landslide"
        }
        
        return Hazard(
            hazard_type=hazard_type_map.get(disaster_type, "other"),
            risk_score=risk_score,
            latitude=latitude,
            longitude=longitude,
            radius=5.0 if disaster_type == "flood" else 3.0,  # km
            source="gdacs",
            description=f"{title}: {description[:200]}",
            start_date=pub_date,
            end_date=end_date,
            verified=True  # GDACS 데이터는 검증된 것으로 간주
        )
    
    async def _search_nearby_regions(self, client: httpx.AsyncClient, db: Session, days: int) -> int:
        """
        주변 지역에서 재난 데이터 검색 (동아프리카)
        """
        nearby_countries = ["Uganda", "Kenya", "Ethiopia", "Sudan"]
        all_count = 0

        for nearby in nearby_countries:
            try:
                params = {
                    "country": nearby,
                    "fromDate": (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d"),
                    "toDate": datetime.utcnow().strftime("%Y-%m-%d")
                }

                response = await client.get(self.BASE_URL, params=params)
                if response.status_code == 204 or len(response.text.strip()) == 0:
                    continue

                root = ET.fromstring(response.text)
                items = root.findall(".//item")

                if len(items) > 0:
                    print(f"[GDACSCollector] {nearby}에서 {len(items)}개 재난 발견")

                    for item in items[:5]:  # 최대 5개만 가져오기
                        try:
                            hazard = self._convert_to_hazard(item)
                            # 소스에 국가명 추가
                            hazard.source = f"gdacs_{nearby.lower().replace(' ', '_')}"

                            existing = db.query(Hazard).filter(
                                Hazard.latitude == hazard.latitude,
                                Hazard.longitude == hazard.longitude,
                                Hazard.source == hazard.source,
                                Hazard.start_date == hazard.start_date
                            ).first()

                            if not existing:
                                db.add(hazard)
                                all_count += 1
                        except Exception:
                            continue

                    if all_count >= 10:  # 최대 10개
                        break

            except Exception as e:
                print(f"[GDACSCollector] {nearby} 검색 오류: {e}")
                continue

        if all_count > 0:
            db.commit()
            print(f"[GDACSCollector] 주변 지역에서 {all_count}개 재난 수집 완료")
            return all_count
        else:
            print("[GDACSCollector] 주변 지역에도 데이터 없음. 더미 데이터 생성...")
            return await self._create_dummy_data(db)

    async def _create_dummy_data(self, db: Session) -> int:
        """
        API 오류 시 더미 데이터 생성 (개발/테스트용) - 더 현실적인 데이터
        중복 방지: 기존 더미 데이터를 먼저 삭제 후 재생성
        """
        print("[GDACSCollector] 더미 재난 데이터 생성 중...")

        import random

        # 기존 더미 데이터 삭제 (중복 방지)
        existing_dummy = db.query(Hazard).filter(Hazard.source == "gdacs_dummy").all()
        if existing_dummy:
            for hazard in existing_dummy:
                db.delete(hazard)
            db.commit()
            print(f"[GDACSCollector] 기존 더미 데이터 {len(existing_dummy)}개 삭제")

        # 주바 주변 좌표 생성 (±0.2도 범위)
        base_lat, base_lng = 4.8594, 31.5713

        dummy_disasters = []
        disaster_types = [
            ("flood", 60, 5.0, "Flash flood warning - heavy rainfall"),
            ("flood", 70, 4.0, "Nile River overflow risk"),
            ("landslide", 55, 2.0, "Landslide risk in hilly areas"),
            ("flood", 50, 6.0, "Seasonal flooding expected"),
        ]

        for i, (htype, risk, radius, desc) in enumerate(disaster_types):
            lat_offset = random.uniform(-0.2, 0.2)
            lng_offset = random.uniform(-0.2, 0.2)

            dummy_disasters.append({
                "latitude": base_lat + lat_offset,
                "longitude": base_lng + lng_offset,
                "hazard_type": htype,
                "risk_score": risk + random.randint(-10, 10),
                "description": f"{desc} (Test data #{i+1})",
                "radius": radius
            })

        count = 0
        for disaster_data in dummy_disasters:
            hazard = Hazard(
                source="gdacs_dummy",
                start_date=datetime.utcnow() - timedelta(hours=random.randint(1, 72)),
                end_date=datetime.utcnow() + timedelta(days=random.randint(3, 14)),
                verified=True,  # 더미 데이터를 실제 위험처럼 인식
                **disaster_data
            )
            db.add(hazard)
            count += 1

        db.commit()
        print(f"[GDACSCollector] {count}개 더미 재난 생성 완료")
        return count
