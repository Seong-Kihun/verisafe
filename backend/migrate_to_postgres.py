"""SQLite → PostgreSQL 데이터 마이그레이션 스크립트"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.config import settings
from app.models.user import User
from app.models.hazard import Hazard
from app.models.report import Report
from app.models.landmark import Landmark

# SQLite 연결
sqlite_engine = create_engine('sqlite:///./verisafe.db', echo=True)
SqliteSession = sessionmaker(bind=sqlite_engine)

# PostgreSQL 연결
postgres_engine = create_engine(settings.database_url, echo=True)
PostgresSession = sessionmaker(bind=postgres_engine)


def migrate_users():
    """사용자 데이터 마이그레이션"""
    sqlite_db = SqliteSession()
    postgres_db = PostgresSession()

    try:
        users = sqlite_db.query(User).all()
        print(f"[Migrate] {len(users)}명의 사용자 마이그레이션 중...")

        for user in users:
            # PostgreSQL에 삽입
            postgres_db.merge(user)

        postgres_db.commit()
        print(f"[Migrate] 사용자 마이그레이션 완료")
    except Exception as e:
        print(f"[Migrate] 사용자 마이그레이션 오류: {e}")
        postgres_db.rollback()
    finally:
        sqlite_db.close()
        postgres_db.close()


def migrate_hazards():
    """위험 정보 마이그레이션"""
    sqlite_db = SqliteSession()
    postgres_db = PostgresSession()

    try:
        hazards = sqlite_db.query(Hazard).all()
        print(f"[Migrate] {len(hazards)}개의 위험 정보 마이그레이션 중...")

        for hazard in hazards:
            postgres_db.merge(hazard)

        postgres_db.commit()
        print(f"[Migrate] 위험 정보 마이그레이션 완료")
    except Exception as e:
        print(f"[Migrate] 위험 정보 마이그레이션 오류: {e}")
        postgres_db.rollback()
    finally:
        sqlite_db.close()
        postgres_db.close()


def migrate_reports():
    """제보 데이터 마이그레이션"""
    sqlite_db = SqliteSession()
    postgres_db = PostgresSession()

    try:
        reports = sqlite_db.query(Report).all()
        print(f"[Migrate] {len(reports)}개의 제보 마이그레이션 중...")

        for report in reports:
            postgres_db.merge(report)

        postgres_db.commit()
        print(f"[Migrate] 제보 마이그레이션 완료")
    except Exception as e:
        print(f"[Migrate] 제보 마이그레이션 오류: {e}")
        postgres_db.rollback()
    finally:
        sqlite_db.close()
        postgres_db.close()


def migrate_landmarks():
    """랜드마크 마이그레이션"""
    sqlite_db = SqliteSession()
    postgres_db = PostgresSession()

    try:
        landmarks = sqlite_db.query(Landmark).all()
        print(f"[Migrate] {len(landmarks)}개의 랜드마크 마이그레이션 중...")

        for landmark in landmarks:
            postgres_db.merge(landmark)

        postgres_db.commit()
        print(f"[Migrate] 랜드마크 마이그레이션 완료")
    except Exception as e:
        print(f"[Migrate] 랜드마크 마이그레이션 오류: {e}")
        postgres_db.rollback()
    finally:
        sqlite_db.close()
        postgres_db.close()


if __name__ == "__main__":
    print("=" * 60)
    print("SQLite → PostgreSQL 데이터 마이그레이션 시작")
    print("=" * 60)

    migrate_users()
    migrate_hazards()
    migrate_reports()
    migrate_landmarks()

    print("=" * 60)
    print("마이그레이션 완료!")
    print("=" * 60)
