"""NewsAPI 수집기 - 뉴스 기반 위험 정보 수집"""
import httpx
from datetime import datetime, timedelta
from typing import List, Optional
from sqlalchemy.orm import Session

from app.models.hazard import Hazard
from app.config import settings


class NewsCollector:
    """
    NewsAPI 연동 - 뉴스 기사 기반 위험 정보

    데이터 소스: https://newsapi.org/
    - 남수단 관련 뉴스 수집
    - 키워드 분석으로 위험 유형 감지
    - 신뢰할 수 있는 언론사 소스

    Phase 2 구현
    """

    BASE_URL = "https://newsapi.org/v2/everything"

    # 위험 감지 키워드 및 가중치
    HAZARD_KEYWORDS = {
        "conflict": {
            "keywords": ["war", "conflict", "fighting", "violence", "battle", "armed", "military", "attack", "killed", "casualties"],
            "base_risk": 70
        },
        "protest": {
            "keywords": ["protest", "demonstration", "rally", "unrest", "riot"],
            "base_risk": 45
        },
        "disaster": {
            "keywords": ["flood", "drought", "famine", "epidemic", "disease", "cholera", "disaster", "emergency"],
            "base_risk": 60
        },
        "humanitarian": {
            "keywords": ["refugee", "displaced", "humanitarian crisis", "aid", "relief"],
            "base_risk": 50
        },
        "political": {
            "keywords": ["coup", "government", "political crisis", "election violence"],
            "base_risk": 55
        }
    }

    # 신뢰할 수 있는 언론사 (추가 가중치)
    TRUSTED_SOURCES = [
        "bbc", "reuters", "ap", "aljazeera", "guardian", "nytimes", "cnn",
        "afp", "dw", "voa", "rfi", "irinnews"
    ]

    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: NewsAPI 키 (settings.news_api_key 사용 또는 명시적 제공)
        """
        self.api_key = api_key or settings.news_api_key

    async def collect_recent_news(self, db: Session, query: str = "South Sudan", days: int = 3) -> int:
        """
        최근 N일간의 뉴스를 수집하여 DB에 저장

        Args:
            db: Database session
            query: 검색 쿼리 (기본값: "South Sudan")
            days: 수집할 일수 (기본값: 3일)

        Returns:
            수집된 위험 정보 수
        """
        if not self.api_key:
            print("[NewsCollector] 경고: API 키가 설정되지 않았습니다. 더미 데이터를 생성합니다.")
            return await self._create_dummy_data(db)

        print(f"[NewsCollector] '{query}' 최근 {days}일 뉴스 수집 중...")

        # 날짜 범위 계산
        from_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

        params = {
            "apiKey": self.api_key,
            "q": query,
            "from": from_date,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 100  # 최대 100개
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(self.BASE_URL, params=params)
                response.raise_for_status()
                data = response.json()

            if data.get("status") != "ok":
                print(f"[NewsCollector] API 응답 오류: {data.get('message', 'Unknown error')}")
                return 0

            articles = data.get("articles", [])
            print(f"[NewsCollector] {len(articles)}개 기사 발견")

            # Hazard 모델로 변환 및 저장
            count = 0
            for article in articles:
                try:
                    hazard = self._convert_to_hazard(article)

                    if not hazard:
                        continue

                    # 중복 확인 (동일 소스 URL)
                    url = article.get("url", "")
                    source_id = f"news_{hash(url)}"

                    existing = db.query(Hazard).filter(
                        Hazard.source == source_id
                    ).first()

                    if not existing:
                        db.add(hazard)
                        count += 1

                except Exception as e:
                    print(f"[NewsCollector] 기사 변환 오류: {e}")
                    continue

            db.commit()
            print(f"[NewsCollector] {count}개 새 위험 정보 저장 완료")
            return count

        except httpx.HTTPError as e:
            print(f"[NewsCollector] API 오류: {e}")
            return 0

    def _convert_to_hazard(self, article: dict) -> Optional[Hazard]:
        """
        뉴스 기사를 Hazard 모델로 변환

        분석 방법:
        1. 제목 + 설명 + 내용에서 키워드 매칭
        2. 매칭된 키워드 수와 신뢰도로 위험도 계산
        3. 언론사 신뢰도 반영
        """
        title = article.get("title", "").lower()
        description = article.get("description", "").lower()
        content = article.get("content", "").lower()

        full_text = f"{title} {description} {content}"

        # 위험 유형 감지 및 점수 계산
        best_match = None
        max_score = 0

        for h_type, config in self.HAZARD_KEYWORDS.items():
            keywords = config["keywords"]
            matched_count = sum(1 for keyword in keywords if keyword in full_text)

            if matched_count > 0:
                score = matched_count * 10 + config["base_risk"]
                if score > max_score:
                    max_score = score
                    best_match = h_type

        if not best_match:
            # 위험 키워드가 없으면 저장하지 않음
            return None

        # hazard_type 정규화
        hazard_type_map = {
            "conflict": "conflict",
            "protest": "protest",
            "disaster": "natural_disaster",
            "humanitarian": "other",
            "political": "other"
        }
        hazard_type = hazard_type_map.get(best_match, "other")

        # 신뢰도 점수 계산
        source_name = article.get("source", {}).get("name", "").lower()
        is_trusted = any(trusted in source_name for trusted in self.TRUSTED_SOURCES)

        # 기본 위험도
        risk_score = self.HAZARD_KEYWORDS[best_match]["base_risk"]

        # 키워드 매칭 강도로 조정
        keyword_boost = min(30, (max_score - risk_score))
        risk_score = min(100, risk_score + keyword_boost)

        # 신뢰할 수 있는 언론사는 +10점
        if is_trusted:
            risk_score = min(100, risk_score + 10)

        # 낮은 위험도는 필터링
        if risk_score < 40:
            return None

        # 날짜 파싱
        published_at_str = article.get("publishedAt", "")
        try:
            published_at = datetime.fromisoformat(published_at_str.replace("Z", "+00:00"))
        except:
            published_at = datetime.utcnow()

        # 지속 기간: 뉴스 정보는 중간 정도 (48-72시간)
        duration_hours = 72 if hazard_type == "conflict" else 48
        end_date = published_at + timedelta(hours=duration_hours)

        # 위치: 뉴스는 일반적으로 구체적 위치 없음 → 남수단 중심 (Juba)
        # TODO: NLP로 기사 내 위치 추출 (Phase 3)
        latitude = 4.8517
        longitude = 31.5825

        # 설명문 생성
        description_text = f"News [{article.get('source', {}).get('name', 'Unknown')}]: {article.get('title', '')[:150]}"
        if article.get("description"):
            description_text += f" - {article.get('description')[:100]}"

        # URL 저장
        url = article.get("url", "")

        return Hazard(
            hazard_type=hazard_type,
            risk_score=risk_score,
            latitude=latitude,
            longitude=longitude,
            radius=5.0,  # km (뉴스는 넓은 지역 커버)
            source=f"news_{hash(url)}",
            description=description_text,
            start_date=published_at,
            end_date=end_date,
            verified=is_trusted  # 신뢰할 수 있는 언론사만 검증됨
        )

    async def _create_dummy_data(self, db: Session) -> int:
        """
        API 키가 없을 때 더미 데이터 생성 (개발/테스트용)
        중복 방지: 기존 더미 데이터를 먼저 삭제 후 재생성
        """
        print("[NewsCollector] 더미 뉴스 데이터 생성 중...")

        # 기존 더미 데이터 삭제 (중복 방지)
        existing_dummy = db.query(Hazard).filter(Hazard.source == "news_dummy").all()
        if existing_dummy:
            for hazard in existing_dummy:
                db.delete(hazard)
            db.commit()
            print(f"[NewsCollector] 기존 더미 데이터 {len(existing_dummy)}개 삭제")

        dummy_news = [
            {
                "latitude": 4.8517, "longitude": 31.5825,
                "hazard_type": "conflict", "risk_score": 75,
                "description": "News [BBC]: Armed clashes reported in Juba suburbs amid rising tensions",
                "radius": 5.0
            },
            {
                "latitude": 4.8517, "longitude": 31.5825,
                "hazard_type": "natural_disaster", "risk_score": 65,
                "description": "News [Reuters]: Heavy flooding affects thousands in Upper Nile region",
                "radius": 8.0
            },
            {
                "latitude": 4.8517, "longitude": 31.5825,
                "hazard_type": "other", "risk_score": 52,
                "description": "News [Al Jazeera]: Humanitarian crisis deepens as aid access restricted",
                "radius": 6.0
            }
        ]

        count = 0
        for news_data in dummy_news:
            hazard = Hazard(
                source="news_dummy",
                start_date=datetime.utcnow(),
                end_date=datetime.utcnow() + timedelta(hours=48),
                verified=True,
                **news_data
            )
            db.add(hazard)
            count += 1

        db.commit()
        print(f"[NewsCollector] {count}개 더미 뉴스 생성 완료")
        return count
