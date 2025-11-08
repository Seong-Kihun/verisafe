"""VeriSafe API 메인 애플리케이션 (PostgreSQL + Redis)"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    debug=settings.debug
)

# Static 파일 서빙 설정 (데모 페이지용)
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/demo", StaticFiles(directory=static_dir, html=True), name="demo")
    logger.info(f"Static 파일 서빙 활성화: /demo")
else:
    logger.warning(f"Static 디렉토리 없음: {static_dir}")


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 실행"""
    import asyncio
    from app.database import SessionLocal, init_db
    from app.services.graph_manager import GraphManager
    from app.services.hazard_scorer import HazardScorer
    from app.services.redis_manager import redis_manager

    logger.info("서버 시작 - 초기화 시작")

    # 0. Secret Key 검증 (프로덕션만 강제, 개발은 경고만)
    if settings.debug:
        logger.info("개발 환경 모드 - 기본 설정으로 시작")
    settings.validate_production_secrets()

    # 1. 데이터베이스 초기화 (테이블 생성)
    logger.info("데이터베이스 테이블 초기화 중")
    init_db()

    # 2. Redis 초기화
    logger.info("Redis 연결 초기화 중")
    redis_manager.initialize()

    # 3. 백그라운드 초기화
    async def initialize_background():
        """백그라운드에서 초기화 수행"""
        try:
            # GraphManager 초기화 (한 번만 실행)
            db = SessionLocal()
            try:
                graph_manager = GraphManager()
                await graph_manager.initialize(db)
            finally:
                db.close()  # 초기화 완료 후 세션 정리

            # HazardScorer 초기화 (session_factory 전달)
            hazard_scorer = HazardScorer(graph_manager, session_factory=SessionLocal)

            # 초기 위험도 계산
            await hazard_scorer.update_all_risk_scores()

            # 백그라운드 스케줄러 시작 (내부에서 새 세션 생성)
            asyncio.create_task(hazard_scorer.start_scheduler())

            logger.info("GraphManager + HazardScorer 초기화 완료")

            # 외부 데이터 수집 스케줄러 추가
            from app.services.external_data.data_collector_scheduler import DataCollectorScheduler

            logger.info("외부 데이터 수집 스케줄러 초기화 중")
            data_scheduler = DataCollectorScheduler(SessionLocal, graph_manager)

            # 서버 시작 시 즉시 1회 수집 (비동기로 실행)
            logger.info("초기 외부 데이터 수집 시작")
            asyncio.create_task(data_scheduler.run_once(db))

            # 24시간마다 자동 수집 시작 (백그라운드)
            asyncio.create_task(data_scheduler.start_scheduler(interval_hours=24))
            logger.info("외부 데이터 스케줄러 시작됨 (24시간 주기)")

        except Exception as e:
            logger.error(f"초기화 오류: {e}", exc_info=True)

    # 백그라운드 태스크로 실행
    asyncio.create_task(initialize_background())


@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 실행"""
    from app.services.redis_manager import redis_manager

    logger.info("서버 종료 - 리소스 정리")

    # Redis 연결 종료
    client = redis_manager.get_client()
    if client:
        client.close()
        logger.info("Redis 연결 종료")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """API 상태 확인"""
    from app.services.redis_manager import redis_manager

    redis_stats = redis_manager.get_stats()

    return {
        "name": settings.app_name,
        "version": settings.version,
        "status": "running",
        "database": settings.database_type,
        "redis": redis_stats.get("status", "unknown")
    }


@app.get("/health")
async def health_check():
    """헬스 체크"""
    from app.services.redis_manager import redis_manager

    redis_status = redis_manager.get_stats().get("status")

    return {
        "status": "healthy",
        "database": settings.database_type,
        "redis": redis_status
    }


# 라우터 등록
from app.routes import auth, map, report, route, external_data, admin, data_dashboard, ai_predictions, ai_training, safe_havens, emergency, safety_checkin, detected_feature, mapper, reviewer
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(map.router, prefix="/api/map", tags=["map"])
app.include_router(report.router, prefix="/api/reports", tags=["reports"])
app.include_router(route.router, prefix="/api/route", tags=["route"])
app.include_router(external_data.router, prefix="/api/external-data", tags=["external_data"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])
app.include_router(data_dashboard.router, prefix="/api/data", tags=["data_dashboard"])
app.include_router(ai_predictions.router, prefix="/api/ai", tags=["ai_predictions"])
app.include_router(ai_training.router, prefix="/api/ai/training", tags=["ai_training"])
app.include_router(safe_havens.router, prefix="/api/safe-havens", tags=["safe_havens"])
app.include_router(emergency.router, prefix="/api/emergency", tags=["emergency"])
app.include_router(safety_checkin.router, prefix="/api/safety-checkin", tags=["safety_checkin"])
app.include_router(detected_feature.router, prefix="/api/map", tags=["detected_features"])
app.include_router(mapper.router, prefix="/api/mapper", tags=["mapper"])
app.include_router(reviewer.router, prefix="/api/review", tags=["reviewer"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
