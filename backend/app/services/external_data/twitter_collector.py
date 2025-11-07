"""Twitter/X API 수집기 - 실시간 소셜 미디어 모니터링"""
import httpx
from datetime import datetime, timedelta
from typing import List, Optional
from sqlalchemy.orm import Session

from app.models.hazard import Hazard
from app.config import settings


class TwitterCollector:
    """
    Twitter/X API v2 연동 - 실시간 위험 정보 수집

    데이터 소스: https://developer.twitter.com/
    - 위험 관련 키워드 모니터링
    - 지리적 위치 기반 트윗 수집
    - 소셜 미디어 기반 조기 경보

    Phase 2 구현
    """

    BASE_URL = "https://api.twitter.com/2/tweets/search/recent"

    # 위험 감지 키워드 (남수단 맥락)
    HAZARD_KEYWORDS = {
        "conflict": ["conflict", "fighting", "violence", "attack", "armed", "shooting", "gunfire", "clash"],
        "protest": ["protest", "demonstration", "rally", "march", "strike"],
        "disaster": ["flood", "flooding", "drought", "famine", "disease", "epidemic", "cholera"],
        "checkpoint": ["checkpoint", "roadblock", "blockade"],
        "emergency": ["emergency", "urgent", "crisis", "danger", "warning"]
    }

    def __init__(self, bearer_token: Optional[str] = None):
        """
        Args:
            bearer_token: Twitter API Bearer Token (settings.twitter_bearer_token 사용 또는 명시적 제공)
        """
        self.bearer_token = bearer_token or settings.twitter_bearer_token

    async def collect_recent_tweets(self, db: Session, location: str = "South Sudan", hours: int = 24) -> int:
        """
        최근 N시간의 트윗을 수집하여 DB에 저장

        Args:
            db: Database session
            location: 위치 키워드 (기본값: "South Sudan")
            hours: 수집할 시간 범위 (기본값: 24시간)

        Returns:
            수집된 위험 정보 수
        """
        if not self.bearer_token:
            print("[TwitterCollector] 경고: API 토큰이 설정되지 않았습니다. 더미 데이터를 생성합니다.")
            return await self._create_dummy_data(db)

        print(f"[TwitterCollector] {location} 최근 {hours}시간 트윗 수집 중...")

        # 검색 쿼리 생성
        all_keywords = []
        for category_keywords in self.HAZARD_KEYWORDS.values():
            all_keywords.extend(category_keywords)

        # 트위터 검색 쿼리: (keyword1 OR keyword2 OR ...) AND location -is:retweet
        keywords_query = " OR ".join(all_keywords)
        query = f"({keywords_query}) {location} -is:retweet lang:en"

        # 시간 범위
        start_time = (datetime.utcnow() - timedelta(hours=hours)).isoformat() + "Z"

        params = {
            "query": query,
            "start_time": start_time,
            "max_results": 100,  # 최대 100개 (Twitter API 제한)
            "tweet.fields": "created_at,geo,public_metrics,entities",
            "expansions": "geo.place_id",
            "place.fields": "full_name,geo,place_type"
        }

        headers = {
            "Authorization": f"Bearer {self.bearer_token}"
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(self.BASE_URL, params=params, headers=headers)
                response.raise_for_status()
                data = response.json()

            if "data" not in data or not data["data"]:
                print("[TwitterCollector] 검색된 트윗이 없습니다.")
                return 0

            tweets = data["data"]
            places = {p["id"]: p for p in data.get("includes", {}).get("places", [])}

            print(f"[TwitterCollector] {len(tweets)}개 트윗 발견")

            # Hazard 모델로 변환 및 저장
            count = 0
            for tweet in tweets:
                try:
                    hazard = self._convert_to_hazard(tweet, places)

                    if not hazard:
                        continue

                    # 중복 확인 (동일 소스 ID)
                    source_id = f"twitter_{tweet['id']}"
                    existing = db.query(Hazard).filter(
                        Hazard.source == source_id
                    ).first()

                    if not existing:
                        db.add(hazard)
                        count += 1

                except Exception as e:
                    print(f"[TwitterCollector] 트윗 변환 오류: {e}")
                    continue

            db.commit()
            print(f"[TwitterCollector] {count}개 새 위험 정보 저장 완료")
            return count

        except httpx.HTTPError as e:
            print(f"[TwitterCollector] API 오류: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"[TwitterCollector] 응답 내용: {e.response.text}")
            return 0

    def _convert_to_hazard(self, tweet: dict, places: dict) -> Optional[Hazard]:
        """
        트윗을 Hazard 모델로 변환

        트윗 분석:
        1. 키워드 매칭으로 위험 유형 판단
        2. 리트윗/좋아요 수로 신뢰도 계산
        3. 위치 정보 추출 (있는 경우)
        """
        text = tweet.get("text", "").lower()

        # 위험 유형 감지
        hazard_type = "other"
        matched_keywords = []

        for h_type, keywords in self.HAZARD_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    hazard_type = h_type
                    matched_keywords.append(keyword)
                    break
            if hazard_type != "other":
                break

        # hazard_type 정규화
        hazard_type_map = {
            "conflict": "conflict",
            "protest": "protest",
            "disaster": "natural_disaster",
            "checkpoint": "checkpoint",
            "emergency": "other"
        }
        hazard_type = hazard_type_map.get(hazard_type, "other")

        # 위치 정보 추출
        latitude, longitude = self._extract_location(tweet, places)

        if not latitude or not longitude:
            # 위치 정보 없으면 남수단 중심 (Juba)
            latitude = 4.8517
            longitude = 31.5825

        # 신뢰도 점수 계산 (공개 지표 기반)
        metrics = tweet.get("public_metrics", {})
        retweet_count = metrics.get("retweet_count", 0)
        like_count = metrics.get("like_count", 0)

        # 기본 위험도 (소셜 미디어는 낮게 시작)
        base_risk = {
            "conflict": 60,
            "protest": 40,
            "natural_disaster": 50,
            "checkpoint": 35,
            "other": 30
        }.get(hazard_type, 30)

        # 인기도로 위험도 조정 (최대 +20점)
        engagement = retweet_count + (like_count * 0.5)
        risk_adjustment = min(20, engagement // 10)
        risk_score = min(100, base_risk + risk_adjustment)

        # 낮은 위험도는 필터링 (스팸 방지)
        if risk_score < 30:
            return None

        # 날짜 파싱
        created_at_str = tweet.get("created_at", "")
        try:
            created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
        except:
            created_at = datetime.utcnow()

        # 지속 기간: 소셜 미디어 정보는 짧게 (12-24시간)
        duration_hours = 24 if hazard_type == "conflict" else 12
        end_date = created_at + timedelta(hours=duration_hours)

        # 설명문 생성
        description = f"Twitter: {text[:200]}"
        if matched_keywords:
            description += f" [Keywords: {', '.join(matched_keywords[:3])}]"

        return Hazard(
            hazard_type=hazard_type,
            risk_score=risk_score,
            latitude=latitude,
            longitude=longitude,
            radius=1.0,  # km (소셜 미디어는 반경 작게)
            source=f"twitter_{tweet['id']}",
            description=description,
            start_date=created_at,
            end_date=end_date,
            verified=False  # 소셜 미디어는 미검증
        )

    def _extract_location(self, tweet: dict, places: dict) -> tuple:
        """
        트윗에서 위치 정보 추출

        Returns:
            (latitude, longitude) 또는 (None, None)
        """
        # 1. geo 필드에서 직접 좌표
        geo = tweet.get("geo", {})
        if geo and "coordinates" in geo:
            coords = geo["coordinates"]
            if "coordinates" in coords:
                # Point: [lng, lat]
                return coords["coordinates"][1], coords["coordinates"][0]

        # 2. place_id로 장소 조회
        if "place_id" in geo and geo["place_id"] in places:
            place = places[geo["place_id"]]
            place_geo = place.get("geo", {})

            if "bbox" in place_geo:
                # Bounding box의 중심점 계산
                bbox = place_geo["bbox"]
                lat = (bbox[1] + bbox[3]) / 2
                lng = (bbox[0] + bbox[2]) / 2
                return lat, lng

        return None, None

    async def _create_dummy_data(self, db: Session) -> int:
        """
        API 토큰이 없을 때 더미 데이터 생성 (개발/테스트용)
        중복 방지: 기존 더미 데이터를 먼저 삭제 후 재생성
        """
        print("[TwitterCollector] 더미 트위터 데이터 생성 중...")

        # 기존 더미 데이터 삭제 (중복 방지)
        existing_dummy = db.query(Hazard).filter(Hazard.source == "twitter_dummy").all()
        if existing_dummy:
            for hazard in existing_dummy:
                db.delete(hazard)
            db.commit()
            print(f"[TwitterCollector] 기존 더미 데이터 {len(existing_dummy)}개 삭제")

        dummy_tweets = [
            {
                "latitude": 4.8550, "longitude": 31.5850,
                "hazard_type": "conflict", "risk_score": 55,
                "description": "Twitter: Reports of gunfire heard near central district #SouthSudan",
                "radius": 1.0
            },
            {
                "latitude": 4.8480, "longitude": 31.5920,
                "hazard_type": "protest", "risk_score": 42,
                "description": "Twitter: Peaceful protest gathering at main square #Juba",
                "radius": 0.8
            },
            {
                "latitude": 4.8600, "longitude": 31.5780,
                "hazard_type": "natural_disaster", "risk_score": 48,
                "description": "Twitter: Heavy flooding reported in northern suburbs",
                "radius": 1.2
            }
        ]

        count = 0
        for tweet_data in dummy_tweets:
            hazard = Hazard(
                source="twitter_dummy",
                start_date=datetime.utcnow(),
                end_date=datetime.utcnow() + timedelta(hours=12),
                verified=True,  # 더미 데이터를 실제 위험처럼 인식
                **tweet_data
            )
            db.add(hazard)
            count += 1

        db.commit()
        print(f"[TwitterCollector] {count}개 더미 트윗 생성 완료")
        return count
