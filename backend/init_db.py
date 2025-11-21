"""데이터베이스 초기화 스크립트"""
from app.database import Base, engine, SessionLocal
from app.models import User, Road, Hazard, HazardScoringRule, Report, Landmark
from app.models.hazard import Hazard as HazardModel
from app.models.landmark import Landmark as LandmarkModel
from app.models.safe_haven import SafeHaven
from datetime import datetime, timedelta
import uuid


def init_db():
    """모든 테이블 생성"""
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("[OK] Tables created")


def seed_data():
    """초기 데이터 삽입"""
    db = SessionLocal()
    
    try:
        # 1. hazard_scoring_rules 데이터
        print("Inserting hazard scoring rules...")
        rules_data = [
            {
                'hazard_type': 'armed_conflict',
                'base_risk_score': 95,
                'min_risk_score': 90,
                'max_risk_score': 100,
                'default_duration_hours': 72,
                'default_radius_km': 10.0,
                'icon': 'gun',
                'color': '#EF4444',
                'description': '무력충돌 (총격, 폭격 등)'
            },
            {
                'hazard_type': 'protest_riot',
                'base_risk_score': 80,
                'min_risk_score': 70,
                'max_risk_score': 85,
                'default_duration_hours': 72,
                'default_radius_km': 5.0,
                'icon': 'crowd',
                'color': '#F59E0B',
                'description': '시위/폭동'
            },
            {
                'hazard_type': 'checkpoint',
                'base_risk_score': 70,
                'min_risk_score': 60,
                'max_risk_score': 80,
                'default_duration_hours': 24,
                'default_radius_km': 2.0,
                'icon': 'warning',
                'color': '#FF6B6B',
                'description': '불법 검문소'
            },
            {
                'hazard_type': 'road_damage',
                'base_risk_score': 80,
                'min_risk_score': 70,
                'max_risk_score': 90,
                'default_duration_hours': 168,
                'default_radius_km': 0.1,
                'icon': 'construction',
                'color': '#F97316',
                'description': '도로 유실/파손'
            },
            {
                'hazard_type': 'natural_disaster',
                'base_risk_score': 85,
                'min_risk_score': 70,
                'max_risk_score': 90,
                'default_duration_hours': 168,
                'default_radius_km': 5.0,
                'icon': 'explosion',
                'color': '#DC2626',
                'description': '자연재해'
            },
            {
                'hazard_type': 'safe_haven',
                'base_risk_score': 0,
                'min_risk_score': 0,
                'max_risk_score': 0,
                'default_duration_hours': 24,
                'default_radius_km': 0.1,
                'icon': 'building',
                'color': '#10B981',
                'description': '안전 거점 (병원, 대사관 등)'
            },
            {
                'hazard_type': 'other',
                'base_risk_score': 50,
                'min_risk_score': 40,
                'max_risk_score': 60,
                'default_duration_hours': 48,
                'default_radius_km': 3.0,
                'icon': 'other',
                'color': '#6B7280',
                'description': '기타'
            },
        ]
        
        inserted_count = 0
        for rule_data in rules_data:
            # 이미 존재하는지 확인
            existing = db.query(HazardScoringRule).filter(
                HazardScoringRule.hazard_type == rule_data['hazard_type']
            ).first()
            
            if not existing:
                rule = HazardScoringRule(**rule_data)
                db.add(rule)
                inserted_count += 1
        
        db.commit()
        print(f"[OK] Inserted {inserted_count} new hazard scoring rules (skipped {len(rules_data) - inserted_count} existing)")
        
        # 2. 테스트 사용자 (개발용: 간단한 해시)
        print("Inserting test users...")
        import hashlib
        
        test_users_data = [
            {
                'username': 'admin',
                'email': 'admin@verisafe.com',
                'password_hash': hashlib.sha256('admin123'.encode()).hexdigest(),
                'role': 'admin',
                'verified': True
            },
            {
                'username': 'testuser',
                'email': 'user@verisafe.com',
                'password_hash': hashlib.sha256('user123'.encode()).hexdigest(),
                'role': 'user',
                'verified': True
            }
        ]
        
        inserted_count = 0
        for user_data in test_users_data:
            # 이미 존재하는지 확인
            existing = db.query(User).filter(
                User.username == user_data['username']
            ).first()
            
            if not existing:
                user = User(id=uuid.uuid4(), **user_data)
                db.add(user)
                inserted_count += 1
        
        db.commit()
        print(f"[OK] Inserted {inserted_count} new test users (skipped {len(test_users_data) - inserted_count} existing)")
        
        # 3. 더미 위험 데이터 (주바 중심) - 다양한 위험 유형과 점수로 테스트용
        print("Inserting dummy hazard data...")
        now = datetime.now()
        
        # Juba 중심 좌표: 4.8594, 31.5713
        # 주변 지역에 다양한 위험 정보 생성
        dummy_hazards = [
            # 무력충돌 (높은 위험도) - South Sudan
            HazardModel(
                hazard_type='armed_conflict',
                risk_score=95,
                latitude=4.8620,
                longitude=31.5750,
                radius=10.0,
                country='SS',
                source='system',
                description='총격전 발생 - 매우 위험한 지역',
                start_date=now - timedelta(hours=2),
                end_date=now + timedelta(hours=70),
                verified=True
            ),
            HazardModel(
                hazard_type='armed_conflict',
                risk_score=90,
                latitude=4.8550,
                longitude=31.5800,
                radius=8.0,
                country='SS',
                source='external_api',
                description='폭격 위험 지역',
                start_date=now - timedelta(hours=12),
                end_date=now + timedelta(hours=60),
                verified=True
            ),
            
            # 시위/폭동 (중간-높은 위험도) - South Sudan
            HazardModel(
                hazard_type='protest_riot',
                risk_score=85,
                latitude=4.8650,
                longitude=31.5850,
                radius=5.0,
                country='SS',
                source='user_report',
                description='대규모 시위 진행 중 - 폭력 가능성',
                start_date=now - timedelta(hours=3),
                end_date=now + timedelta(hours=69),
                verified=True
            ),
            HazardModel(
                hazard_type='protest_riot',
                risk_score=75,
                latitude=4.8480,
                longitude=31.5650,
                radius=4.0,
                country='SS',
                source='system',
                description='학생 시위 예상 지역',
                start_date=now - timedelta(hours=1),
                end_date=now + timedelta(hours=71),
                verified=True
            ),
            HazardModel(
                hazard_type='protest_riot',
                risk_score=70,
                latitude=4.8520,
                longitude=31.5900,
                radius=3.5,
                country='SS',
                source='user_report',
                description='소규모 집회 - 주의 필요',
                start_date=now - timedelta(hours=5),
                end_date=now + timedelta(hours=67),
                verified=True
            ),
            
            # 검문소 (중간 위험도) - South Sudan
            HazardModel(
                hazard_type='checkpoint',
                risk_score=75,
                latitude=4.8500,
                longitude=31.5800,
                radius=2.0,
                country='SS',
                source='system',
                description='불법 검문소 - 매우 위험',
                start_date=now - timedelta(hours=6),
                end_date=now + timedelta(hours=18),
                verified=True
            ),
            HazardModel(
                hazard_type='checkpoint',
                risk_score=70,
                latitude=4.8600,
                longitude=31.5750,
                radius=2.0,
                country='SS',
                source='system',
                description='불법 검문소 설치됨',
                start_date=now - timedelta(hours=10),
                end_date=now + timedelta(hours=14),
                verified=True
            ),
            HazardModel(
                hazard_type='checkpoint',
                risk_score=65,
                latitude=4.8450,
                longitude=31.5720,
                radius=1.5,
                country='SS',
                source='user_report',
                description='검문소 - 통행 지연 가능',
                start_date=now - timedelta(hours=8),
                end_date=now + timedelta(hours=16),
                verified=True
            ),
            HazardModel(
                hazard_type='checkpoint',
                risk_score=60,
                latitude=4.8570,
                longitude=31.5680,
                radius=1.8,
                country='SS',
                source='system',
                description='일반 검문소',
                start_date=now - timedelta(hours=4),
                end_date=now + timedelta(hours=20),
                verified=True
            ),
            
            # 도로 손상 (중간 위험도) - South Sudan
            HazardModel(
                hazard_type='road_damage',
                risk_score=85,
                latitude=4.8550,
                longitude=31.5700,
                radius=0.1,
                country='SS',
                source='user_report',
                description='큰 구덩이 - 차량 통행 불가',
                start_date=now - timedelta(hours=1),
                end_date=now + timedelta(hours=167),
                verified=True
            ),
            HazardModel(
                hazard_type='road_damage',
                risk_score=80,
                latitude=4.8630,
                longitude=31.5780,
                radius=0.15,
                country='SS',
                source='system',
                description='도로 유실 - 우회 필요',
                start_date=now - timedelta(hours=24),
                end_date=now + timedelta(hours=144),
                verified=True
            ),
            HazardModel(
                hazard_type='road_damage',
                risk_score=75,
                latitude=4.8470,
                longitude=31.5820,
                radius=0.08,
                country='SS',
                source='user_report',
                description='도로 파손 - 천천히 통행',
                start_date=now - timedelta(hours=12),
                end_date=now + timedelta(hours=156),
                verified=True
            ),
            HazardModel(
                hazard_type='road_damage',
                risk_score=70,
                latitude=4.8580,
                longitude=31.5650,
                radius=0.05,
                country='SS',
                source='system',
                description='작은 구덩이',
                start_date=now - timedelta(hours=6),
                end_date=now + timedelta(hours=162),
                verified=True
            ),
            
            # 무력충돌 추가 (South Sudan - 다양한 위치)
            HazardModel(
                hazard_type='armed_conflict',
                risk_score=92,
                latitude=4.8700,
                longitude=31.5600,
                radius=12.0,
                country='SS',
                source='external_api',
                description='무력충돌 지역 - 교전 발생',
                start_date=now - timedelta(hours=4),
                end_date=now + timedelta(hours=68),
                verified=True
            ),
            HazardModel(
                hazard_type='armed_conflict',
                risk_score=88,
                latitude=4.8400,
                longitude=31.5900,
                radius=9.0,
                country='SS',
                source='system',
                description='교전 가능 지역 - 접근 금지',
                start_date=now - timedelta(hours=8),
                end_date=now + timedelta(hours=64),
                verified=True
            ),
            HazardModel(
                hazard_type='armed_conflict',
                risk_score=94,
                latitude=4.8660,
                longitude=31.5720,
                radius=11.0,
                country='SS',
                source='user_report',
                description='격렬한 총격전 - 극도로 위험',
                start_date=now - timedelta(hours=1),
                end_date=now + timedelta(hours=71),
                verified=True
            ),
            HazardModel(
                hazard_type='armed_conflict',
                risk_score=91,
                latitude=4.8510,
                longitude=31.5740,
                radius=10.0,
                country='SS',
                source='external_api',
                description='무장 단체 출몰 - 매우 위험',
                start_date=now - timedelta(hours=6),
                end_date=now + timedelta(hours=66),
                verified=True
            ),
            HazardModel(
                hazard_type='armed_conflict',
                risk_score=93,
                latitude=4.8560,
                longitude=31.5830,
                radius=8.5,
                country='SS',
                source='system',
                description='공격 위험 - 즉시 대피 권고',
                start_date=now - timedelta(hours=3),
                end_date=now + timedelta(hours=69),
                verified=True
            ),
            HazardModel(
                hazard_type='armed_conflict',
                risk_score=89,
                latitude=4.8490,
                longitude=31.5770,
                radius=7.0,
                country='SS',
                source='user_report',
                description='폭발 및 총성 보고됨',
                start_date=now - timedelta(hours=5),
                end_date=now + timedelta(hours=67),
                verified=True
            ),

            # Kenya 데이터 추가 (Nairobi 중심: -1.2864, 36.8172)
            HazardModel(
                hazard_type='protest_riot',
                risk_score=75,
                latitude=-1.2850,
                longitude=36.8200,
                radius=3.0,
                country='KE',
                source='external_api',
                description='나이로비 중심가 시위 진행 중',
                start_date=now - timedelta(hours=6),
                end_date=now + timedelta(hours=66),
                verified=True
            ),
            HazardModel(
                hazard_type='checkpoint',
                risk_score=65,
                latitude=-1.2900,
                longitude=36.8150,
                radius=1.5,
                country='KE',
                source='system',
                description='경찰 검문소 - 교통 혼잡',
                start_date=now - timedelta(hours=3),
                end_date=now + timedelta(hours=21),
                verified=True
            ),

            # Ethiopia 데이터 추가 (Addis Ababa 중심: 9.0320, 38.7469)
            HazardModel(
                hazard_type='armed_conflict',
                risk_score=90,
                latitude=9.0300,
                longitude=38.7500,
                radius=8.0,
                country='ET',
                source='external_api',
                description='무력충돌 - 아디스아바바 외곽',
                start_date=now - timedelta(hours=10),
                end_date=now + timedelta(hours=62),
                verified=True
            ),
            HazardModel(
                hazard_type='road_damage',
                risk_score=80,
                latitude=9.0350,
                longitude=38.7400,
                radius=0.2,
                country='ET',
                source='user_report',
                description='도로 유실 - 통행 불가',
                start_date=now - timedelta(hours=12),
                end_date=now + timedelta(hours=156),
                verified=True
            ),
        ]
        
        # 기존 위험 정보 삭제 (중복 방지)
        db.query(HazardModel).delete()
        db.commit()
        
        db.add_all(dummy_hazards)
        db.commit()
        print(f"[OK] Inserted {len(dummy_hazards)} test hazards (various types and risk scores)")
        
        # 4. 더미 랜드마크 (주바 주요 장소)
        print("Inserting dummy landmarks...")
        
        dummy_landmarks = [
            LandmarkModel(
                name='Juba International Airport',
                category='airport',
                latitude=4.8670,
                longitude=31.5880,
                description='주바 국제공항'
            ),
            LandmarkModel(
                name='Juba City Hall',
                category='government',
                latitude=4.8500,
                longitude=31.6000,
                description='주바 시청'
            ),
            LandmarkModel(
                name='Juba Teaching Hospital',
                category='hospital',
                latitude=4.8470,
                longitude=31.5800,
                description='주바 대학병원'
            ),
            LandmarkModel(
                name='Crown Hotel',
                category='hotel',
                latitude=4.8520,
                longitude=31.5900,
                description='크라운 호텔'
            ),
            LandmarkModel(
                name='UN House',
                category='government',
                latitude=4.8550,
                longitude=31.5950,
                description='UN 주바 사무소'
            ),
        ]
        
        db.add_all(dummy_landmarks)
        db.commit()
        print(f"[OK] Inserted {len(dummy_landmarks)} dummy landmarks")

        # 5. 더미 안전 대피처 (주바 지역)
        print("Inserting dummy safe havens...")

        dummy_safe_havens = [
            # 대사관
            SafeHaven(
                name='US Embassy - Juba',
                category='embassy',
                latitude=4.8530,
                longitude=31.5920,
                address='Kololo Road, Juba',
                phone='+211-912-105-188',
                hours='08:00-17:00 (Mon-Fri)',
                verified=True,
                notes='미국 대사관 - 긴급 상황 시 미국 시민 보호'
            ),
            SafeHaven(
                name='Korean Embassy - South Sudan',
                category='embassy',
                latitude=4.8545,
                longitude=31.5935,
                address='Thong Ping Area, Juba',
                phone='+211-920-000-000',
                hours='09:00-18:00 (Mon-Fri)',
                verified=True,
                notes='대한민국 대사관 - 한국 국민 보호'
            ),
            SafeHaven(
                name='UK Embassy - Juba',
                category='embassy',
                latitude=4.8510,
                longitude=31.5945,
                address='EU Compound, Juba',
                phone='+211-912-185-000',
                hours='08:00-16:00 (Mon-Fri)',
                verified=True,
                notes='영국 대사관'
            ),

            # 병원
            SafeHaven(
                name='Juba Teaching Hospital',
                category='hospital',
                latitude=4.8470,
                longitude=31.5800,
                address='Hospital Road, Juba',
                phone='+211-955-000-000',
                hours='24시간 운영',
                capacity=500,
                verified=True,
                notes='주바 대학병원 - 24시간 응급실 운영'
            ),
            SafeHaven(
                name='Al-Sabah Children Hospital',
                category='hospital',
                latitude=4.8520,
                longitude=31.5875,
                address='Munuki Area, Juba',
                phone='+211-955-111-000',
                hours='24시간 운영',
                capacity=200,
                verified=True,
                notes='어린이 병원 - 소아과 전문'
            ),

            # UN 시설
            SafeHaven(
                name='UN House - Juba',
                category='un',
                latitude=4.8550,
                longitude=31.5950,
                address='UN Compound, Juba',
                phone='+211-912-000-000',
                hours='24시간 운영',
                capacity=1000,
                verified=True,
                notes='UN 주바 사무소 - 민간인 보호 가능'
            ),
            SafeHaven(
                name='UNMISS Tomping Base',
                category='un',
                latitude=4.8600,
                longitude=31.5700,
                address='Tomping Area, Juba',
                phone='+211-912-000-100',
                hours='24시간 운영',
                capacity=5000,
                verified=True,
                notes='유엔 남수단 임무단 기지 - 대규모 민간인 보호 시설'
            ),

            # 경찰서
            SafeHaven(
                name='Juba Central Police Station',
                category='police',
                latitude=4.8505,
                longitude=31.5815,
                address='Central Juba',
                phone='+211-977-000-000',
                hours='24시간 운영',
                verified=True,
                notes='주바 중앙 경찰서'
            ),

            # 안전 호텔
            SafeHaven(
                name='Juba Grand Hotel',
                category='hotel',
                latitude=4.8525,
                longitude=31.5895,
                address='Munuki, Juba',
                phone='+211-956-000-000',
                hours='24시간 운영',
                capacity=150,
                verified=True,
                notes='안전한 숙박 시설 - 보안 강화'
            ),
            SafeHaven(
                name='Crown Hotel Juba',
                category='hotel',
                latitude=4.8520,
                longitude=31.5900,
                address='Hai Referendum, Juba',
                phone='+211-955-222-000',
                hours='24시간 운영',
                capacity=100,
                verified=True,
                notes='시내 중심 호텔 - 보안 시설 완비'
            ),

            # 대피소
            SafeHaven(
                name='Red Cross Emergency Shelter',
                category='shelter',
                latitude=4.8480,
                longitude=31.5850,
                address='Kator Area, Juba',
                phone='+211-955-333-000',
                hours='24시간 운영',
                capacity=300,
                verified=True,
                notes='적십자 긴급 대피소 - 무료 제공'
            ),
        ]

        db.add_all(dummy_safe_havens)
        db.commit()
        print(f"[OK] Inserted {len(dummy_safe_havens)} dummy safe havens")

        print("\n[SUCCESS] Database initialized successfully!")
        
    except Exception as e:
        print(f"[ERROR] {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
    seed_data()
