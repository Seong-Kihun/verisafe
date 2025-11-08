"""
데이터베이스 초기화 스크립트
테이블을 생성하고 기본 데이터를 설정합니다.
"""
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(__file__))

from app.database import init_db

if __name__ == "__main__":
    print("데이터베이스 초기화 시작...")
    init_db()
    print("✅ 데이터베이스 초기화 완료!")
