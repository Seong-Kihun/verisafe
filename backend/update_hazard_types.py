"""
위험 타입 중복 제거 마이그레이션 스크립트

중복 타입 통합:
- conflict → armed_conflict
- protest_riot → protest
"""
import sqlite3
from pathlib import Path
import sys
import io

# UTF-8 출력 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 데이터베이스 경로
DB_PATH = Path(__file__).parent / "verisafe.db"

def update_hazard_types():
    """데이터베이스의 중복 위험 타입 업데이트"""

    if not DB_PATH.exists():
        print(f"[ERROR] 데이터베이스를 찾을 수 없습니다: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # 1. hazards 테이블 업데이트
        print("\n[INFO] hazards 테이블 업데이트 중...")

        # conflict → armed_conflict
        cursor.execute("SELECT COUNT(*) FROM hazards WHERE hazard_type = 'conflict'")
        conflict_count = cursor.fetchone()[0]
        if conflict_count > 0:
            cursor.execute("""
                UPDATE hazards
                SET hazard_type = 'armed_conflict'
                WHERE hazard_type = 'conflict'
            """)
            print(f"  [OK] 'conflict' -> 'armed_conflict' ({conflict_count}개)")
        else:
            print(f"  [INFO] 'conflict' 타입 데이터 없음")

        # protest_riot → protest
        cursor.execute("SELECT COUNT(*) FROM hazards WHERE hazard_type = 'protest_riot'")
        protest_riot_count = cursor.fetchone()[0]
        if protest_riot_count > 0:
            cursor.execute("""
                UPDATE hazards
                SET hazard_type = 'protest'
                WHERE hazard_type = 'protest_riot'
            """)
            print(f"  [OK] 'protest_riot' -> 'protest' ({protest_riot_count}개)")
        else:
            print(f"  [INFO] 'protest_riot' 타입 데이터 없음")

        # 2. hazard_scoring_rules 테이블 업데이트
        print("\n[INFO] hazard_scoring_rules 테이블 업데이트 중...")

        # conflict 처리: armed_conflict가 이미 있으면 conflict 삭제, 없으면 conflict를 armed_conflict로 변경
        cursor.execute("SELECT COUNT(*) FROM hazard_scoring_rules WHERE hazard_type = 'armed_conflict'")
        has_armed_conflict = cursor.fetchone()[0] > 0

        cursor.execute("SELECT COUNT(*) FROM hazard_scoring_rules WHERE hazard_type = 'conflict'")
        has_conflict = cursor.fetchone()[0] > 0

        if has_conflict:
            if has_armed_conflict:
                # armed_conflict가 이미 있으면 conflict 삭제
                cursor.execute("DELETE FROM hazard_scoring_rules WHERE hazard_type = 'conflict'")
                print(f"  [OK] 'conflict' 규칙 삭제 (armed_conflict 존재)")
            else:
                # armed_conflict가 없으면 conflict를 armed_conflict로 변경
                cursor.execute("""
                    UPDATE hazard_scoring_rules
                    SET hazard_type = 'armed_conflict'
                    WHERE hazard_type = 'conflict'
                """)
                print(f"  [OK] 'conflict' -> 'armed_conflict'")
        else:
            print(f"  [INFO] 'conflict' 타입 규칙 없음")

        # protest_riot 처리: protest가 이미 있으면 protest_riot 삭제, 없으면 protest_riot을 protest로 변경
        cursor.execute("SELECT COUNT(*) FROM hazard_scoring_rules WHERE hazard_type = 'protest'")
        has_protest = cursor.fetchone()[0] > 0

        cursor.execute("SELECT COUNT(*) FROM hazard_scoring_rules WHERE hazard_type = 'protest_riot'")
        has_protest_riot = cursor.fetchone()[0] > 0

        if has_protest_riot:
            if has_protest:
                # protest가 이미 있으면 protest_riot 삭제
                cursor.execute("DELETE FROM hazard_scoring_rules WHERE hazard_type = 'protest_riot'")
                print(f"  [OK] 'protest_riot' 규칙 삭제 (protest 존재)")
            else:
                # protest가 없으면 protest_riot을 protest로 변경
                cursor.execute("""
                    UPDATE hazard_scoring_rules
                    SET hazard_type = 'protest'
                    WHERE hazard_type = 'protest_riot'
                """)
                print(f"  [OK] 'protest_riot' -> 'protest'")
        else:
            print(f"  [INFO] 'protest_riot' 타입 규칙 없음")

        # 3. reports 테이블 업데이트 (있는 경우)
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='reports'
        """)
        if cursor.fetchone():
            print("\n[INFO] reports 테이블 업데이트 중...")

            # conflict → armed_conflict
            cursor.execute("SELECT COUNT(*) FROM reports WHERE hazard_type = 'conflict'")
            report_conflict_count = cursor.fetchone()[0]
            if report_conflict_count > 0:
                cursor.execute("""
                    UPDATE reports
                    SET hazard_type = 'armed_conflict'
                    WHERE hazard_type = 'conflict'
                """)
                print(f"  [OK] 'conflict' -> 'armed_conflict' ({report_conflict_count}개)")
            else:
                print(f"  [INFO] 'conflict' 타입 리포트 없음")

            # protest_riot → protest
            cursor.execute("SELECT COUNT(*) FROM reports WHERE hazard_type = 'protest_riot'")
            report_protest_count = cursor.fetchone()[0]
            if report_protest_count > 0:
                cursor.execute("""
                    UPDATE reports
                    SET hazard_type = 'protest'
                    WHERE hazard_type = 'protest_riot'
                """)
                print(f"  [OK] 'protest_riot' -> 'protest' ({report_protest_count}개)")
            else:
                print(f"  [INFO] 'protest_riot' 타입 리포트 없음")

        # 변경사항 커밋
        conn.commit()

        # 최종 상태 확인
        print("\n[INFO] 업데이트 후 위험 타입 분포:")
        cursor.execute("""
            SELECT hazard_type, COUNT(*) as count
            FROM hazards
            GROUP BY hazard_type
            ORDER BY count DESC
        """)
        for hazard_type, count in cursor.fetchall():
            print(f"  - {hazard_type}: {count}개")

        print("\n[SUCCESS] 위험 타입 업데이트 완료!")

    except Exception as e:
        print(f"\n[ERROR] 오류 발생: {e}")
        conn.rollback()
        raise

    finally:
        conn.close()

if __name__ == "__main__":
    print("="*60)
    print("위험 타입 중복 제거 마이그레이션")
    print("="*60)
    update_hazard_types()
