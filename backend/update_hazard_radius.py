"""위험 정보 반경을 더 현실적인 값으로 업데이트"""
import sys
sys.path.append('.')

from app.database import SessionLocal
from app.models.hazard import HazardScoringRule, Hazard
from sqlalchemy import update

# 현실적인 반경 값 (km)
REALISTIC_RADIUS = {
    'armed_conflict': 1.5,      # 10km -> 1.5km (무력 충돌은 국지적)
    'protest_riot': 0.8,        # 5km -> 0.8km (시위/폭동은 특정 지역)
    'protest': 0.8,             # 2km -> 0.8km
    'checkpoint': 0.3,          # 0.5km -> 0.3km (검문소는 매우 국지적)
    'road_damage': 0.5,         # 1km -> 0.5km (도로 손상)
    'natural_disaster': 3.0,    # 30km -> 3km (자연재해)
    'flood': 2.0,               # 20km -> 2km (홍수)
    'landslide': 1.0,           # 5km -> 1km (산사태)
    'other': 1.0,               # 3km -> 1km
    'safe_haven': 0.1,          # 0.1km (대피처)
}

def update_radius():
    """반경 업데이트"""
    db = SessionLocal()

    try:
        print("=" * 60)
        print("위험 정보 반경 업데이트")
        print("=" * 60)

        # 1. HazardScoringRule 업데이트
        print("\n[1] HazardScoringRule 업데이트 중...")
        for hazard_type, radius in REALISTIC_RADIUS.items():
            result = db.execute(
                update(HazardScoringRule)
                .where(HazardScoringRule.hazard_type == hazard_type)
                .values(default_radius_km=radius)
            )
            if result.rowcount > 0:
                print(f"  [OK] {hazard_type}: {radius}km")

        db.commit()

        # 2. Hazard 테이블 업데이트 (기존 위험 정보도 업데이트)
        print("\n[2] Hazard 테이블 업데이트 중...")
        for hazard_type, radius in REALISTIC_RADIUS.items():
            result = db.execute(
                update(Hazard)
                .where(Hazard.hazard_type == hazard_type)
                .values(radius=radius)
            )
            if result.rowcount > 0:
                print(f"  [OK] {hazard_type}: {result.rowcount}개 업데이트 -> {radius}km")

        db.commit()

        # 3. 검증
        print("\n[3] 업데이트 검증...")
        rules = db.query(HazardScoringRule).all()
        print("\nHazardScoringRule:")
        for rule in rules:
            print(f"  - {rule.hazard_type}: {rule.default_radius_km}km")

        hazards = db.query(Hazard).limit(10).all()
        print(f"\nHazard (샘플 {len(hazards)}개):")
        for hazard in hazards:
            print(f"  - {hazard.hazard_type}: {hazard.radius}km")

        print("\n[SUCCESS] 반경 업데이트 완료!")

    except Exception as e:
        print(f"\n[ERROR] 업데이트 실패: {e}")
        db.rollback()
        raise
    finally:
        db.close()

if __name__ == '__main__':
    update_radius()
