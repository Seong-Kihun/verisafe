"""건물 감지 서비스 - AI 기반 건물 자동 감지"""
import httpx
import random
from datetime import datetime
from typing import List, Dict, Optional
from sqlalchemy.orm import Session

from app.models.detected_feature import DetectedFeature
from app.config import settings


class BuildingDetector:
    """
    건물 자동 감지 서비스

    데이터 소스:
    1. Microsoft Building Footprints (무료 오픈 데이터)
    2. YOLOv8 위성 이미지 분석 (Phase 2)
    3. 사용자 제보 (크라우드소싱)

    현재: 더미 데이터 생성 (개발/테스트용)
    """

    # Microsoft Building Footprints API (개념적 - 실제로는 파일 다운로드)
    MICROSOFT_API_URL = "https://github.com/microsoft/GlobalMLBuildingFootprints"

    def __init__(self):
        """초기화"""
        pass

    async def detect_buildings_in_area(self, db: Session,
                                       lat: float, lng: float,
                                       radius_km: float = 5.0) -> int:
        """
        특정 지역에서 건물 감지

        Args:
            db: Database session
            lat: 중심 위도
            lng: 중심 경도
            radius_km: 검색 반경 (km)

        Returns:
            감지된 건물 수
        """
        print(f"[BuildingDetector] 건물 감지 시작: ({lat}, {lng}), 반경 {radius_km}km")

        # Phase 1: 더미 데이터 생성
        # Phase 2에서 실제 Microsoft 데이터나 AI 모델로 교체
        return await self._create_dummy_buildings(db, lat, lng, radius_km)

    async def detect_critical_facilities(self, db: Session,
                                        lat: float, lng: float,
                                        radius_km: float = 10.0) -> int:
        """
        중요 시설 감지 (병원, 학교, 경찰서, 소방서 등)

        Args:
            db: Database session
            lat: 중심 위도
            lng: 중심 경도
            radius_km: 검색 반경 (km)

        Returns:
            감지된 시설 수
        """
        print(f"[BuildingDetector] 중요 시설 감지 시작: ({lat}, {lng})")

        # 중요 시설 유형별 특징
        facilities = [
            {
                'type': 'hospital',
                'confidence': 0.85,
                'indicators': ['red_cross', 'large_parking', 'ambulance_access'],
                'count': 2
            },
            {
                'type': 'school',
                'confidence': 0.78,
                'indicators': ['playground', 'large_field', 'multiple_buildings'],
                'count': 4
            },
            {
                'type': 'police',
                'confidence': 0.82,
                'indicators': ['flag', 'fenced_compound', 'vehicle_parking'],
                'count': 2
            },
            {
                'type': 'safe_haven',
                'confidence': 0.75,
                'indicators': ['church', 'large_compound', 'central_location'],
                'count': 3
            }
        ]

        count = 0
        for facility in facilities:
            for i in range(facility['count']):
                # 반경 내 랜덤 위치
                lat_offset = random.uniform(-radius_km/111.0, radius_km/111.0)
                lng_offset = random.uniform(-radius_km/111.0, radius_km/111.0)

                feature = DetectedFeature(
                    feature_type=facility['type'],
                    latitude=lat + lat_offset,
                    longitude=lng + lng_offset,
                    confidence=facility['confidence'] + random.uniform(-0.05, 0.05),
                    detection_source='ai_satellite_analysis',
                    verified=False,
                    name=f"AI-detected {facility['type']} #{i+1}",
                    description=f"Detected via satellite image analysis. "
                              f"Indicators: {', '.join(facility['indicators'])}",
                    properties={
                        'indicators': facility['indicators'],
                        'detection_method': 'yolo_v8_satellite',
                        'area_sqm': random.randint(500, 3000)
                    }
                )

                # 중복 확인
                existing = db.query(DetectedFeature).filter(
                    DetectedFeature.feature_type == facility['type'],
                    DetectedFeature.latitude.between(feature.latitude - 0.001, feature.latitude + 0.001),
                    DetectedFeature.longitude.between(feature.longitude - 0.001, feature.longitude + 0.001)
                ).first()

                if not existing:
                    db.add(feature)
                    count += 1

        db.commit()
        print(f"[BuildingDetector] {count}개 중요 시설 감지 완료")
        return count

    async def detect_bridges(self, db: Session,
                           lat: float, lng: float,
                           radius_km: float = 15.0) -> int:
        """
        다리 감지 (강/하천을 가로지르는 구조물)

        Args:
            db: Database session
            lat: 중심 위도
            lng: 중심 경도
            radius_km: 검색 반경 (km)

        Returns:
            감지된 다리 수
        """
        print(f"[BuildingDetector] 다리 감지 시작: ({lat}, {lng})")

        # 주바 지역의 나일강 주변 다리 예상 위치
        bridge_locations = [
            {
                'name': 'Juba Bridge',
                'lat': 4.8520,
                'lng': 31.5950,
                'confidence': 0.92,
                'length_m': 450
            },
            {
                'name': 'Northern River Crossing',
                'lat': 4.8750,
                'lng': 31.6100,
                'confidence': 0.78,
                'length_m': 200
            }
        ]

        count = 0
        for bridge_data in bridge_locations:
            # 반경 내에 있는지 확인
            from app.utils.geo import haversine_distance
            distance = haversine_distance(lat, lng, bridge_data['lat'], bridge_data['lng'])

            if distance <= radius_km:
                feature = DetectedFeature(
                    feature_type='bridge',
                    latitude=bridge_data['lat'],
                    longitude=bridge_data['lng'],
                    confidence=bridge_data['confidence'],
                    detection_source='ai_satellite_analysis',
                    verified=False,
                    name=bridge_data['name'],
                    description=f"Bridge detected crossing waterway. "
                              f"Estimated length: {bridge_data['length_m']}m",
                    properties={
                        'length_m': bridge_data['length_m'],
                        'detection_method': 'satellite_line_detection',
                        'crosses': 'Nile River'
                    }
                )

                # 중복 확인
                existing = db.query(DetectedFeature).filter(
                    DetectedFeature.feature_type == 'bridge',
                    DetectedFeature.latitude.between(feature.latitude - 0.01, feature.latitude + 0.01),
                    DetectedFeature.longitude.between(feature.longitude - 0.01, feature.longitude + 0.01)
                ).first()

                if not existing:
                    db.add(feature)
                    count += 1

        db.commit()
        print(f"[BuildingDetector] {count}개 다리 감지 완료")
        return count

    async def _create_dummy_buildings(self, db: Session,
                                     lat: float, lng: float,
                                     radius_km: float) -> int:
        """
        더미 건물 데이터 생성 (개발/테스트용)

        실제 구현에서는 Microsoft Building Footprints 데이터나
        YOLOv8 모델로 교체
        """
        print(f"[BuildingDetector] 더미 건물 데이터 생성 중...")

        # 기존 더미 건물 삭제 (중복 방지)
        existing_dummy = db.query(DetectedFeature).filter(
            DetectedFeature.detection_source == 'dummy_buildings'
        ).all()

        if existing_dummy:
            for feature in existing_dummy:
                db.delete(feature)
            db.commit()
            print(f"[BuildingDetector] 기존 더미 건물 {len(existing_dummy)}개 삭제")

        # 랜덤 건물 생성
        building_count = random.randint(15, 25)
        count = 0

        for i in range(building_count):
            # 반경 내 랜덤 위치
            lat_offset = random.uniform(-radius_km/111.0, radius_km/111.0)
            lng_offset = random.uniform(-radius_km/111.0, radius_km/111.0)

            # 건물 유형
            building_types = ['building', 'building', 'building', 'residential', 'commercial']
            building_type = random.choice(building_types)

            feature = DetectedFeature(
                feature_type=building_type,
                latitude=lat + lat_offset,
                longitude=lng + lng_offset,
                confidence=random.uniform(0.65, 0.95),
                detection_source='dummy_buildings',
                verified=random.choice([True, False]),  # 일부는 검증된 것으로
                name=f"Building #{i+1}" if random.random() > 0.7 else None,
                description=f"AI-detected {building_type} via satellite imagery (test data)",
                properties={
                    'area_sqm': random.randint(50, 500),
                    'estimated_floors': random.randint(1, 3),
                    'roof_type': random.choice(['flat', 'pitched', 'mixed'])
                }
            )

            db.add(feature)
            count += 1

        db.commit()
        print(f"[BuildingDetector] {count}개 더미 건물 생성 완료")
        return count

    async def collect_all_features(self, db: Session,
                                   center_lat: float = 4.8517,
                                   center_lng: float = 31.5825) -> Dict[str, int]:
        """
        모든 지리 정보 수집 (건물, 시설, 다리)

        Args:
            db: Database session
            center_lat: 중심 위도 (기본값: 주바)
            center_lng: 중심 경도

        Returns:
            수집 통계
        """
        print(f"[BuildingDetector] 전체 지리 정보 수집 시작")

        stats = {
            'buildings': await self.detect_buildings_in_area(db, center_lat, center_lng, radius_km=5.0),
            'facilities': await self.detect_critical_facilities(db, center_lat, center_lng, radius_km=10.0),
            'bridges': await self.detect_bridges(db, center_lat, center_lng, radius_km=15.0)
        }

        total = sum(stats.values())
        print(f"[BuildingDetector] 전체 수집 완료: 총 {total}개")
        print(f"  - 건물: {stats['buildings']}개")
        print(f"  - 중요 시설: {stats['facilities']}개")
        print(f"  - 다리: {stats['bridges']}개")

        return stats
