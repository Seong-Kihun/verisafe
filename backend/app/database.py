"""데이터베이스 연결 설정"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import settings

# SQLAlchemy 엔진 생성
engine = create_engine(
    settings.database_url,
    echo=settings.debug,  # SQL 쿼리 로깅 (개발 환경에서만)
    pool_size=10,         # 커넥션 풀 크기
    max_overflow=20,      # 최대 추가 커넥션
    pool_pre_ping=True,   # 연결 유효성 체크
    pool_recycle=3600     # 1시간마다 커넥션 재생성
)

# 세션 팩토리
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base 클래스
Base = declarative_base()


def get_db():
    """데이터베이스 세션 의존성"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """데이터베이스 초기화 (테이블 생성)"""
    from app.models import user, road, hazard, landmark, report, safe_haven, sos_event, safety_checkin
    Base.metadata.create_all(bind=engine)
    print("[Database] 테이블 생성 완료")
