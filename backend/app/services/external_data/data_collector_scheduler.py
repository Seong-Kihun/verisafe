"""외부 데이터 수집 스케줄러"""
import asyncio
from datetime import datetime
from sqlalchemy.orm import Session

from app.services.external_data.acled_collector import ACLEDCollector
from app.services.external_data.gdacs_collector import GDACSCollector
from app.services.external_data.reliefweb_collector import ReliefWebCollector
from app.services.external_data.twitter_collector import TwitterCollector
from app.services.external_data.news_collector import NewsCollector
from app.services.external_data.sentinel_collector import SentinelCollector
from app.services.ai_detection.building_detector import BuildingDetector
from app.services.graph_manager import GraphManager


class DataCollectorScheduler:
    """
    외부 데이터 수집 스케줄러
    
    Phase 2 구현:
    - 매일 정해진 시간에 외부 API에서 데이터 수집
    - 수집된 데이터를 Hazard 테이블에 저장
    - GraphManager에 위험도 업데이트 요청
    """
    
    def __init__(self, db_session_factory, graph_manager: GraphManager):
        """
        Args:
            db_session_factory: Database session factory (callable)
            graph_manager: GraphManager 인스턴스
        """
        self.db_session_factory = db_session_factory
        self.graph_manager = graph_manager

        # Phase 1 collectors
        self.acled = ACLEDCollector()
        self.gdacs = GDACSCollector()
        self.reliefweb = ReliefWebCollector()

        # Phase 2 collectors
        self.twitter = TwitterCollector()
        self.news = NewsCollector()
        self.sentinel = SentinelCollector()

        # AI detection
        self.building_detector = BuildingDetector()

        self.is_running = False
    
    async def collect_all_data(self, db: Session) -> dict:
        """
        모든 외부 데이터 소스에서 데이터 수집
        
        Returns:
            수집 통계 딕셔너리
        """
        print(f"[DataCollectorScheduler] 데이터 수집 시작: {datetime.utcnow()}")
        
        stats = {
            "timestamp": datetime.utcnow().isoformat(),
            "acled": 0,
            "gdacs": 0,
            "reliefweb": 0,
            "twitter": 0,
            "news": 0,
            "sentinel": 0,
            "ai_detection": 0,
            "total": 0,
            "errors": []
        }
        
        # 1. ACLED (분쟁 데이터)
        try:
            acled_count = await self.acled.collect_recent_events(db, country="South Sudan", days=7)
            stats["acled"] = acled_count
            stats["total"] += acled_count
        except Exception as e:
            error_msg = f"ACLED 수집 오류: {str(e)}"
            print(f"[DataCollectorScheduler] {error_msg}")
            stats["errors"].append(error_msg)
        
        # 2. GDACS (재난 데이터)
        try:
            gdacs_count = await self.gdacs.collect_recent_disasters(db, country="South Sudan", days=30)
            stats["gdacs"] = gdacs_count
            stats["total"] += gdacs_count
        except Exception as e:
            error_msg = f"GDACS 수집 오류: {str(e)}"
            print(f"[DataCollectorScheduler] {error_msg}")
            stats["errors"].append(error_msg)
        
        # 3. ReliefWeb (인도적 보고서)
        try:
            reliefweb_count = await self.reliefweb.collect_recent_reports(db, country="South Sudan", days=7)
            stats["reliefweb"] = reliefweb_count
            stats["total"] += reliefweb_count
        except Exception as e:
            error_msg = f"ReliefWeb 수집 오류: {str(e)}"
            print(f"[DataCollectorScheduler] {error_msg}")
            stats["errors"].append(error_msg)

        # 4. Twitter/X (소셜 미디어)
        try:
            twitter_count = await self.twitter.collect_recent_tweets(db, location="South Sudan", hours=24)
            stats["twitter"] = twitter_count
            stats["total"] += twitter_count
        except Exception as e:
            error_msg = f"Twitter 수집 오류: {str(e)}"
            print(f"[DataCollectorScheduler] {error_msg}")
            stats["errors"].append(error_msg)

        # 5. NewsAPI (뉴스 기사)
        try:
            news_count = await self.news.collect_recent_news(db, query="South Sudan", days=3)
            stats["news"] = news_count
            stats["total"] += news_count
        except Exception as e:
            error_msg = f"News 수집 오류: {str(e)}"
            print(f"[DataCollectorScheduler] {error_msg}")
            stats["errors"].append(error_msg)

        # 6. Sentinel Hub (위성 이미지)
        try:
            sentinel_count = await self.sentinel.collect_satellite_data(db, days=7)
            stats["sentinel"] = sentinel_count
            stats["total"] += sentinel_count
        except Exception as e:
            error_msg = f"Sentinel 수집 오류: {str(e)}"
            print(f"[DataCollectorScheduler] {error_msg}")
            stats["errors"].append(error_msg)

        # 7. AI Building Detection (건물, 중요 시설, 다리 감지)
        try:
            detection_stats = await self.building_detector.collect_all_features(db)
            detection_total = sum(detection_stats.values())
            stats["ai_detection"] = detection_total
            stats["total"] += detection_total
            print(f"[DataCollectorScheduler] AI 감지 완료: {detection_total}개 (건물: {detection_stats['buildings']}, 시설: {detection_stats['facilities']}, 다리: {detection_stats['bridges']})")
        except Exception as e:
            error_msg = f"AI Detection 오류: {str(e)}"
            print(f"[DataCollectorScheduler] {error_msg}")
            stats["errors"].append(error_msg)

        print(f"[DataCollectorScheduler] 수집 완료: 총 {stats['total']}개 항목")

        # 8. GraphManager에 위험도 업데이트 요청
        try:
            await self.graph_manager.load_hazards_to_graph(db)
            print("[DataCollectorScheduler] GraphManager 위험도 업데이트 완료")
        except Exception as e:
            error_msg = f"GraphManager 업데이트 오류: {str(e)}"
            print(f"[DataCollectorScheduler] {error_msg}")
            stats["errors"].append(error_msg)
        
        return stats
    
    async def start_scheduler(self, interval_hours: int = 24):
        """
        스케줄러 시작 (백그라운드 실행)
        
        Args:
            interval_hours: 수집 주기 (시간 단위, 기본 24시간)
        """
        self.is_running = True
        print(f"[DataCollectorScheduler] 스케줄러 시작 (주기: {interval_hours}시간)")
        
        while self.is_running:
            try:
                # DB 세션 생성
                db = self.db_session_factory()
                
                # 데이터 수집 실행
                stats = await self.collect_all_data(db)
                
                # 세션 종료
                db.close()
                
                # 다음 실행까지 대기
                await asyncio.sleep(interval_hours * 3600)
                
            except Exception as e:
                print(f"[DataCollectorScheduler] 스케줄러 오류: {e}")
                await asyncio.sleep(3600)  # 오류 시 1시간 후 재시도
    
    def stop_scheduler(self):
        """스케줄러 중지"""
        self.is_running = False
        print("[DataCollectorScheduler] 스케줄러 중지됨")
    
    async def run_once(self, db: Session) -> dict:
        """
        한 번만 실행 (테스트 또는 수동 트리거용)
        
        Returns:
            수집 통계
        """
        return await self.collect_all_data(db)
