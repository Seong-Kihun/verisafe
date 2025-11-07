"""시스템 헬스체크 스크립트 - 모든 구성 요소 점검"""
import sys
import os


def check_python_version():
    """Python 버전 확인"""
    print("\n[1/8] Python Version Check")
    print("=" * 60)
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and version.minor >= 10:
        print("[OK] Python version is compatible (3.10+)")
        return True
    else:
        print("[ERROR] Python 3.10+ required")
        return False


def check_dependencies():
    """필수 패키지 설치 확인"""
    print("\n[2/8] Dependencies Check")
    print("=" * 60)

    # 항상 필요한 패키지
    required_packages = [
        'fastapi',
        'uvicorn',
        'sqlalchemy',
        'redis',
        'jose',  # python-jose
        'passlib',
        'httpx',
        'networkx',
        'pydantic',
    ]

    # 조건부 패키지
    from app.config import settings
    conditional_packages = []

    if settings.database_type == "postgresql":
        conditional_packages.append(('psycopg2', 'PostgreSQL driver (database_type=postgresql)'))
    else:
        print("[INFO] Using SQLite - psycopg2 not required")

    all_ok = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"[OK] {package}")
        except ImportError:
            print(f"[ERROR] {package} not installed")
            all_ok = False

    # 조건부 패키지 확인
    for package, reason in conditional_packages:
        try:
            __import__(package)
            print(f"[OK] {package} ({reason})")
        except ImportError:
            print(f"[ERROR] {package} not installed - {reason}")
            all_ok = False

    return all_ok


def check_database():
    """데이터베이스 연결 확인"""
    print("\n[3/8] Database Connection Check")
    print("=" * 60)

    try:
        from app.database import engine, get_db
        from app.models import Hazard, Landmark

        # 연결 테스트
        with engine.connect() as conn:
            print(f"[OK] Database connected: {engine.url}")

        # 테이블 확인
        db = next(get_db())
        hazard_count = db.query(Hazard).count()
        landmark_count = db.query(Landmark).count()

        print(f"[OK] Hazards table: {hazard_count} records")
        print(f"[OK] Landmarks table: {landmark_count} records")

        return True

    except Exception as e:
        print(f"[ERROR] Database error: {e}")
        return False


def check_redis():
    """Redis 연결 확인"""
    print("\n[4/8] Redis Connection Check")
    print("=" * 60)

    try:
        from app.services.redis_manager import redis_manager

        # Redis 초기화 시도
        redis_manager.initialize()

        # 클라이언트가 None이면 연결 실패
        if redis_manager.get_client() is None:
            print("[WARN] Redis not running or not accessible")
            print("       App will work without caching")
            return True  # Redis는 선택사항

        # 연결 테스트
        redis_manager.set("health_check", "ok", ttl=10)
        result = redis_manager.get("health_check")

        if result == "ok":
            print("[OK] Redis connected and working")
            return True
        else:
            print(f"[WARN] Redis test failed: expected 'ok', got {repr(result)}")
            return True  # Redis는 선택사항

    except Exception as e:
        print(f"[WARN] Redis not available: {e}")
        print("       App will work without caching")
        return True  # Redis는 선택사항


def check_graph_manager():
    """GraphManager 초기화 확인"""
    print("\n[5/8] GraphManager Check")
    print("=" * 60)

    try:
        from app.services.graph_manager import GraphManager

        graph_manager = GraphManager()
        graph = graph_manager.get_graph()

        node_count = graph.number_of_nodes()
        edge_count = graph.number_of_edges()

        print(f"[OK] Graph loaded: {node_count} nodes, {edge_count} edges")

        if node_count == 0:
            print("[WARN] Graph has no nodes - may need initialization")

        return True

    except Exception as e:
        print(f"[ERROR] GraphManager error: {e}")
        return False


def check_env_variables():
    """환경 변수 확인"""
    print("\n[6/8] Environment Variables Check")
    print("=" * 60)

    try:
        from app.config import settings

        # 필수 설정 확인
        print(f"[INFO] Database type: {settings.database_type}")
        print(f"[INFO] Redis host: {settings.redis_host}")
        print(f"[INFO] Debug mode: {settings.debug}")

        # API 키 확인
        if settings.acled_api_key:
            print("[OK] ACLED API key configured")
        else:
            print("[WARN] ACLED API key not configured (using dummy data)")

        # 보안 경고
        settings.validate_production_secrets()

        return True

    except Exception as e:
        print(f"[ERROR] Config error: {e}")
        return False


def check_external_apis():
    """외부 API 연결 확인"""
    print("\n[7/8] External APIs Check")
    print("=" * 60)

    try:
        import httpx
        import asyncio

        async def test_apis():
            async with httpx.AsyncClient(timeout=10.0) as client:
                # GDACS 테스트
                try:
                    response = await client.get(
                        "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH",
                        params={'fromDate': '2024-01-01', 'toDate': '2024-01-02'}
                    )
                    print(f"[OK] GDACS API: Status {response.status_code}")
                except Exception as e:
                    print(f"[WARN] GDACS API: {e}")

        asyncio.run(test_apis())
        return True

    except Exception as e:
        print(f"[ERROR] External API check failed: {e}")
        return False


def check_file_structure():
    """파일 구조 확인"""
    print("\n[8/8] File Structure Check")
    print("=" * 60)

    required_dirs = [
        'app',
        'app/models',
        'app/routes',
        'app/services',
        'app/schemas',
    ]

    required_files = [
        'app/main.py',
        'app/database.py',
        'app/config.py',
    ]

    all_ok = True

    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"[OK] {dir_path}/")
        else:
            print(f"[ERROR] {dir_path}/ not found")
            all_ok = False

    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"[OK] {file_path}")
        else:
            print(f"[ERROR] {file_path} not found")
            all_ok = False

    return all_ok


def main():
    """메인 헬스체크"""
    print("\n" + "=" * 60)
    print("VeriSafe Backend Health Check")
    print("=" * 60)

    results = []

    results.append(("Python Version", check_python_version()))
    results.append(("Dependencies", check_dependencies()))
    results.append(("Database", check_database()))
    results.append(("Redis", check_redis()))
    results.append(("GraphManager", check_graph_manager()))
    results.append(("Environment", check_env_variables()))
    results.append(("External APIs", check_external_apis()))
    results.append(("File Structure", check_file_structure()))

    # 결과 요약
    print("\n" + "=" * 60)
    print("Health Check Summary")
    print("=" * 60)

    for name, status in results:
        status_text = "[OK]" if status else "[FAIL]"
        print(f"{status_text:8} {name}")

    passed = sum(1 for _, status in results if status)
    total = len(results)

    print("=" * 60)
    print(f"Result: {passed}/{total} checks passed")

    if passed == total:
        print("\n[SUCCESS] All systems operational!")
        return 0
    elif passed >= total * 0.7:
        print("\n[WARN] System is operational with warnings")
        return 0
    else:
        print("\n[ERROR] Critical issues detected")
        return 1


if __name__ == "__main__":
    sys.exit(main())
