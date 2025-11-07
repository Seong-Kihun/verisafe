"""Phase 1 기능 테스트"""
import requests
import json

BASE_URL = "http://localhost:8000"


def test_health():
    """헬스 체크"""
    print("\n[Test] 헬스 체크...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    assert response.status_code == 200
    assert response.json()['status'] == 'healthy'
    print("✅ 헬스 체크 통과")


def test_redis_connection():
    """Redis 연결 테스트"""
    print("\n[Test] Redis 연결...")
    response = requests.get(f"{BASE_URL}/api/route/cache/stats")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    assert response.status_code == 200
    assert response.json()['status'] == 'connected'
    print("✅ Redis 연결 확인")


def test_route_calculation_cache():
    """경로 계산 캐싱 테스트"""
    print("\n[Test] 경로 계산 캐싱...")

    # 첫 번째 요청 (캐시 미스)
    payload = {
        "start": {"lat": 4.8670, "lng": 31.5880},
        "end": {"lat": 4.8500, "lng": 31.6000},
        "preference": "safe",
        "transportation_mode": "car",
        "max_routes": 3
    }

    print("\n1. 첫 번째 요청 (캐시 미스 예상)...")
    response1 = requests.post(f"{BASE_URL}/api/route/calculate", json=payload)
    print(f"Status: {response1.status_code}")
    result1 = response1.json()
    print(f"Cache Hit: {result1.get('cache_hit')}")
    print(f"Calculation Time: {result1.get('calculation_time_ms')}ms")
    assert result1.get('cache_hit') == False

    # 두 번째 요청 (캐시 히트)
    print("\n2. 두 번째 요청 (캐시 히트 예상)...")
    response2 = requests.post(f"{BASE_URL}/api/route/calculate", json=payload)
    result2 = response2.json()
    print(f"Cache Hit: {result2.get('cache_hit')}")
    print(f"Calculation Time: {result2.get('calculation_time_ms')}ms")
    assert result2.get('cache_hit') == True
    assert result2.get('calculation_time_ms') < 100  # 캐시는 100ms 이하

    print("✅ 캐싱 동작 확인")


def test_postgis_query():
    """PostGIS 공간 쿼리 테스트"""
    print("\n[Test] PostGIS 공간 쿼리...")

    # 위험 정보 조회 (특정 반경 내)
    response = requests.get(f"{BASE_URL}/api/map/hazards", params={
        "lat": 4.8594,
        "lng": 31.5713,
        "radius": 5.0
    })

    print(f"Status: {response.status_code}")
    hazards = response.json().get('hazards', [])
    print(f"찾은 위험 정보: {len(hazards)}개")

    for hazard in hazards[:3]:
        print(f"  - {hazard.get('hazard_type')}: 위험도 {hazard.get('risk_score')}")

    print("✅ PostGIS 공간 쿼리 확인")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1 기능 테스트 시작")
    print("=" * 60)

    try:
        test_health()
        test_redis_connection()
        test_route_calculation_cache()
        # test_postgis_query()  # 위험 정보가 있을 때 활성화

        print("\n" + "=" * 60)
        print("✅ 모든 테스트 통과!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
