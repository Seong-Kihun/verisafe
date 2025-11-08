"""
HazardScoringRule 테이블에 기본 규칙 데이터 생성
각 위험 유형별로 위험도, 반경, 지속시간 등의 기본값 설정
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.database import SessionLocal
from app.models.hazard import HazardScoringRule
from app.utils.logger import get_logger

logger = get_logger(__name__)

# 위험 유형별 스코어링 규칙
SCORING_RULES = [
    {
        'hazard_type': 'armed_conflict',
        'base_risk_score': 85,
        'min_risk_score': 70,
        'max_risk_score': 95,
        'default_duration_hours': 168,  # 7일
        'default_radius_km': 10.0,
        'icon': '🔫',
        'color': '#D32F2F',
        'description': '무장 충돌 지역'
    },
    {
        'hazard_type': 'conflict',
        'base_risk_score': 70,
        'min_risk_score': 60,
        'max_risk_score': 85,
        'default_duration_hours': 72,  # 3일
        'default_radius_km': 7.0,
        'icon': '⚔️',
        'color': '#E64A19',
        'description': '분쟁 지역'
    },
    {
        'hazard_type': 'protest_riot',
        'base_risk_score': 65,
        'min_risk_score': 50,
        'max_risk_score': 75,
        'default_duration_hours': 24,  # 1일
        'default_radius_km': 3.0,
        'icon': '👥',
        'color': '#F57C00',
        'description': '대규모 시위/폭동'
    },
    {
        'hazard_type': 'protest',
        'base_risk_score': 45,
        'min_risk_score': 30,
        'max_risk_score': 60,
        'default_duration_hours': 12,  # 12시간
        'default_radius_km': 2.0,
        'icon': '📢',
        'color': '#FFA726',
        'description': '평화 시위'
    },
    {
        'hazard_type': 'checkpoint',
        'base_risk_score': 55,
        'min_risk_score': 40,
        'max_risk_score': 70,
        'default_duration_hours': 336,  # 14일
        'default_radius_km': 0.5,
        'icon': '⚠️',
        'color': '#FB8C00',
        'description': '검문소 (불법 또는 위험)'
    },
    {
        'hazard_type': 'road_damage',
        'base_risk_score': 35,
        'min_risk_score': 20,
        'max_risk_score': 50,
        'default_duration_hours': 720,  # 30일
        'default_radius_km': 1.0,
        'icon': '🚧',
        'color': '#FF9800',
        'description': '도로 파손/통행 불가'
    },
    {
        'hazard_type': 'natural_disaster',
        'base_risk_score': 75,
        'min_risk_score': 60,
        'max_risk_score': 90,
        'default_duration_hours': 168,  # 7일
        'default_radius_km': 30.0,
        'icon': '💥',
        'color': '#D84315',
        'description': '자연재해'
    },
    {
        'hazard_type': 'flood',
        'base_risk_score': 65,
        'min_risk_score': 50,
        'max_risk_score': 80,
        'default_duration_hours': 120,  # 5일
        'default_radius_km': 20.0,
        'icon': '🌊',
        'color': '#0277BD',
        'description': '홍수'
    },
    {
        'hazard_type': 'landslide',
        'base_risk_score': 70,
        'min_risk_score': 55,
        'max_risk_score': 85,
        'default_duration_hours': 240,  # 10일
        'default_radius_km': 5.0,
        'icon': '⛰️',
        'color': '#5D4037',
        'description': '산사태'
    },
    {
        'hazard_type': 'other',
        'base_risk_score': 40,
        'min_risk_score': 20,
        'max_risk_score': 60,
        'default_duration_hours': 48,  # 2일
        'default_radius_km': 3.0,
        'icon': '⚠️',
        'color': '#757575',
        'description': '기타 위험'
    }
]


def main():
    """메인 함수: HazardScoringRule 데이터 생성"""
    db = SessionLocal()

    try:
        logger.info("HazardScoringRule 데이터 생성 시작...")

        # 기존 규칙 확인
        existing_count = db.query(HazardScoringRule).count()
        if existing_count > 0:
            logger.warning(f"기존 규칙 {existing_count}개 발견 - 삭제 후 재생성")
            db.query(HazardScoringRule).delete()
            db.commit()

        # 새 규칙 생성
        created_count = 0
        for rule_data in SCORING_RULES:
            rule = HazardScoringRule(**rule_data)
            db.add(rule)
            created_count += 1
            logger.info(f"  ✅ {rule_data['hazard_type']} 규칙 생성 (base={rule_data['base_risk_score']}, radius={rule_data['default_radius_km']}km)")

        db.commit()
        logger.info(f"\n총 {created_count}개의 HazardScoringRule 생성 완료!")

        # 확인
        logger.info("\n=== 생성된 규칙 목록 ===")
        rules = db.query(HazardScoringRule).all()
        for rule in rules:
            logger.info(f"{rule.hazard_type}: base={rule.base_risk_score}, range={rule.min_risk_score}-{rule.max_risk_score}, radius={rule.default_radius_km}km")

    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    main()
