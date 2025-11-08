"""
각 국가별 샘플 위험 정보 데이터 생성
12개 국가에 각각 10-15개의 샘플 위험 정보 추가
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from datetime import datetime, timedelta
import random
from app.database import SessionLocal
from app.models.hazard import Hazard, HazardScoringRule
from app.utils.logger import get_logger

logger = get_logger(__name__)

# 국가별 정보 (코드, 중심 좌표, 도시명)
COUNTRIES = [
    {'code': 'SS', 'name': '남수단', 'lat': 4.8594, 'lon': 31.5713, 'city': 'Juba'},
    {'code': 'KE', 'name': '케냐', 'lat': -1.2864, 'lon': 36.8172, 'city': 'Nairobi'},
    {'code': 'UG', 'name': '우간다', 'lat': 0.3476, 'lon': 32.5825, 'city': 'Kampala'},
    {'code': 'ET', 'name': '에티오피아', 'lat': 9.0320, 'lon': 38.7469, 'city': 'Addis Ababa'},
    {'code': 'SO', 'name': '소말리아', 'lat': 2.0469, 'lon': 45.3182, 'city': 'Mogadishu'},
    {'code': 'CD', 'name': '콩고민주공화국', 'lat': -4.3276, 'lon': 15.3136, 'city': 'Kinshasa'},
    {'code': 'CF', 'name': '중앙아프리카공화국', 'lat': 4.3947, 'lon': 18.5582, 'city': 'Bangui'},
    {'code': 'SD', 'name': '수단', 'lat': 15.5007, 'lon': 32.5599, 'city': 'Khartoum'},
    {'code': 'YE', 'name': '예멘', 'lat': 15.5527, 'lon': 48.5164, 'city': 'Sana\'a'},
    {'code': 'SY', 'name': '시리아', 'lat': 33.5138, 'lon': 36.2765, 'city': 'Damascus'},
    {'code': 'IQ', 'name': '이라크', 'lat': 33.3152, 'lon': 44.3661, 'city': 'Baghdad'},
    {'code': 'AF', 'name': '아프가니스탄', 'lat': 34.5553, 'lon': 69.2075, 'city': 'Kabul'},
]

# 위험 유형별 설명 (다양성을 위해)
HAZARD_DESCRIPTIONS = {
    'armed_conflict': ['무장 충돌 발생', '교전 지역', '군사 작전 진행 중'],
    'conflict': ['충돌 발생', '긴장 고조', '분쟁 지역'],
    'protest_riot': ['대규모 시위', '폭동 발생', '소요 사태'],
    'protest': ['평화 시위', '집회 진행 중', '시민 집결'],
    'checkpoint': ['불법 검문소', '무장 검문', '통행 제한'],
    'road_damage': ['도로 파손', '교량 붕괴', '통행 불가'],
    'natural_disaster': ['홍수 발생', '산사태', '자연재해 지역'],
    'flood': ['홍수 범람', '침수 지역', '하천 범람'],
    'landslide': ['산사태 발생', '토사 붕괴', '낙석 위험'],
    'other': ['기타 위험', '주의 요망', '안전 주의'],
}


def generate_hazards_for_country(country_info, scoring_rules, count=12):
    """특정 국가에 대한 샘플 위험 정보 생성 (HazardScoringRule 기반)"""
    hazards = []

    for i in range(count):
        # 랜덤 위험 유형 선택 (scoring_rules에서)
        rule = random.choice(scoring_rules)

        # 국가 중심점 주변에 랜덤 좌표 생성 (±0.5도 범위)
        lat_offset = random.uniform(-0.5, 0.5)
        lon_offset = random.uniform(-0.5, 0.5)

        # 위험도 점수 (scoring rule의 min-max 범위 내에서 랜덤)
        risk_score = random.randint(rule.min_risk_score, rule.max_risk_score)

        # 반경 (기본값 기준 ±50% 범위)
        radius_min = rule.default_radius_km * 0.5
        radius_max = rule.default_radius_km * 1.5
        radius = round(random.uniform(radius_min, radius_max), 2)

        # 설명
        descriptions = HAZARD_DESCRIPTIONS.get(rule.hazard_type, ['위험 발생'])
        description = random.choice(descriptions)

        # 시작일 (최근 1-30일 내)
        days_ago = random.randint(1, 30)
        start_date = datetime.utcnow() - timedelta(days=days_ago)

        # 종료일 (일부는 영구적, 일부는 scoring rule의 duration 기준)
        end_date = None
        if random.random() > 0.3:  # 70%는 종료일 있음
            # scoring rule의 default_duration_hours 사용 (±50% 범위)
            duration_hours = int(rule.default_duration_hours * random.uniform(0.5, 1.5))
            end_date = start_date + timedelta(hours=duration_hours)

        # 출처
        sources = ['external_api', 'user_report', 'system', 'acled', 'gdelt']
        source = random.choice(sources)

        # 검증 여부 (80%는 검증됨)
        verified = random.random() > 0.2

        hazard = Hazard(
            hazard_type=rule.hazard_type,
            risk_score=risk_score,
            latitude=country_info['lat'] + lat_offset,
            longitude=country_info['lon'] + lon_offset,
            radius=radius,
            country=country_info['code'],
            description=f"{description} - {country_info['city']} 인근",
            source=source,
            start_date=start_date,
            end_date=end_date,
            verified=verified
        )

        hazards.append(hazard)

    return hazards


def main():
    """메인 함수: 모든 국가에 대해 샘플 데이터 생성"""
    db = SessionLocal()

    try:
        logger.info("샘플 위험 정보 데이터 생성 시작...")

        # HazardScoringRule 로드
        scoring_rules = db.query(HazardScoringRule).all()
        if not scoring_rules:
            logger.error("❌ HazardScoringRule 데이터가 없습니다!")
            logger.error("먼저 create_hazard_scoring_rules.py를 실행하세요.")
            return

        logger.info(f"✅ {len(scoring_rules)}개의 HazardScoringRule 로드 완료")

        total_created = 0

        for country in COUNTRIES:
            logger.info(f"\n{country['name']} ({country['code']}) 데이터 생성 중...")

            # 기존 데이터 확인
            existing_count = db.query(Hazard).filter(
                Hazard.country == country['code']
            ).count()

            if existing_count > 0:
                logger.info(f"  기존 데이터 {existing_count}개 발견 - 건너뛰기")
                continue

            # 샘플 데이터 생성 (국가별 10-15개)
            count = random.randint(10, 15)
            hazards = generate_hazards_for_country(country, scoring_rules, count)

            # 데이터베이스에 저장
            for hazard in hazards:
                db.add(hazard)

            db.commit()
            total_created += len(hazards)

            logger.info(f"  ✅ {len(hazards)}개 생성 완료")

        logger.info(f"\n총 {total_created}개의 샘플 위험 정보 생성 완료!")

        # 통계 출력
        logger.info("\n=== 국가별 데이터 현황 ===")
        for country in COUNTRIES:
            count = db.query(Hazard).filter(
                Hazard.country == country['code']
            ).count()
            logger.info(f"{country['name']}: {count}개")

    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    main()
