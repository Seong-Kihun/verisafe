"""
간단한 API 테스트 스크립트
백엔드 서버가 실행 중일 때 이 스크립트를 실행하여 API를 테스트합니다.

사용법:
    python backend/test_api.py
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_search_api():
    """검색 API 테스트"""
    print("\n=== 1. 검색 API 테스트 ===")
    url = f"{BASE_URL}/api/map/search/autocomplete"
    params = {"q": "juba"}
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ 검색 성공: {len(data)}개 결과")
        print(f"   첫 번째 결과: {data[0] if data else '없음'}")
        
        # importance 정렬 확인
        if len(data) > 1:
            importances = [item.get('importance', 0) for item in data]
            is_sorted = all(importances[i] >= importances[i+1] for i in range(len(importances)-1))
            print(f"   Importance 정렬: {'✅ 정렬됨' if is_sorted else '❌ 정렬 안됨'}")
        
        return True
    except Exception as e:
        print(f"❌ 검색 실패: {e}")
        return False

def test_route_calculate():
    """경로 계산 API 테스트"""
    print("\n=== 2. 경로 계산 API 테스트 ===")
    url = f"{BASE_URL}/api/route/calculate"
    
    payload = {
        "start": {"lat": 4.8594, "lng": 31.5713},
        "end": {"lat": 4.8459, "lng": 31.5959},
        "preference": "safe",
        "transportation_mode": "car"
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        
        routes = data.get('routes', [])
        print(f"✅ 경로 계산 성공: {len(routes)}개 경로")
        
        for i, route in enumerate(routes, 1):
            print(f"   경로 {i}:")
            print(f"     - ID: {route.get('id')}")
            print(f"     - 타입: {route.get('type')}")
            print(f"     - 거리: {route.get('distance')}km ({route.get('distance_meters')}m)")
            print(f"     - 소요 시간: {route.get('duration')}분 ({route.get('duration_seconds')}초)")
            print(f"     - 위험 점수: {route.get('risk_score')}/10")
            print(f"     - 이동 수단: {route.get('transportation_mode')}")
            print(f"     - Waypoints: {len(route.get('waypoints', []))}개")
            print(f"     - Polyline: {len(route.get('polyline', []))}개 좌표")
        
        return True, routes[0].get('id') if routes else None
    except Exception as e:
        print(f"❌ 경로 계산 실패: {e}")
        print(f"   응답: {response.text if 'response' in locals() else 'N/A'}")
        return False, None

def test_route_hazards(route_id, polyline):
    """경로 위험 정보 API 테스트"""
    print("\n=== 3. 경로 위험 정보 API 테스트 ===")
    
    if not route_id or not polyline:
        print("⚠️  경로 계산이 실패하여 위험 정보 테스트를 건너뜁니다.")
        return False
    
    url = f"{BASE_URL}/api/route/{route_id}/hazards"
    params = {"polyline": json.dumps(polyline)}
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ 위험 정보 조회 성공")
        print(f"   총 위험 정보: {data.get('summary', {}).get('total_hazards', 0)}개")
        print(f"   가장 많은 위험 유형: {data.get('summary', {}).get('highest_risk_type', '없음')}")
        print(f"   위험 유형별: {data.get('summary', {}).get('hazards_by_type_count', {})}")
        
        hazards = data.get('hazards', [])
        if hazards:
            print(f"   위험 정보 상세 (최대 3개):")
            for i, hazard in enumerate(hazards[:3], 1):
                print(f"     {i}. {hazard.get('hazard_type')} (위험도: {hazard.get('risk_score')}, 거리: {hazard.get('distance_from_route', 0):.1f}m)")
        
        return True
    except Exception as e:
        print(f"❌ 위험 정보 조회 실패: {e}")
        print(f"   응답: {response.text if 'response' in locals() else 'N/A'}")
        return False

def test_different_transportation_modes():
    """다양한 이동 수단 테스트"""
    print("\n=== 4. 이동 수단별 테스트 ===")
    
    modes = ['car', 'walking', 'bicycle']
    url = f"{BASE_URL}/api/route/calculate"
    
    payload_base = {
        "start": {"lat": 4.8594, "lng": 31.5713},
        "end": {"lat": 4.8459, "lng": 31.5959},
        "preference": "safe"
    }
    
    for mode in modes:
        payload = {**payload_base, "transportation_mode": mode}
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            routes = data.get('routes', [])
            
            if routes:
                route = routes[0]
                print(f"   {mode}: {route.get('duration')}분, {route.get('distance')}km")
        except Exception as e:
            print(f"   {mode}: ❌ 실패 - {e}")

if __name__ == "__main__":
    print("=" * 50)
    print("VeriSafe API 테스트")
    print("=" * 50)
    print("\n⚠️  백엔드 서버가 실행 중인지 확인하세요!")
    print("   (python backend/main.py 또는 uvicorn app.main:app --reload)")
    
    input("\n엔터를 눌러 테스트를 시작하세요...")
    
    # 테스트 실행
    test_search_api()
    success, route_id = test_route_calculate()
    
    if success:
        # 경로가 있으면 위험 정보 테스트
        # 실제 polyline 데이터는 경로 계산 응답에서 가져와야 함
        test_polyline = [[4.8594, 31.5713], [4.8459, 31.5959]]  # 예시 데이터
        test_route_hazards(route_id or "test_route", test_polyline)
        
        test_different_transportation_modes()
    
    print("\n" + "=" * 50)
    print("테스트 완료!")
    print("=" * 50)

