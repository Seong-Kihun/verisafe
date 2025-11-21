"""경로 위험도 계산 테스트"""
import sys
sys.path.append('.')

from app.services.graph_manager import GraphManager
from app.services.route_calculator import RouteCalculator
from app.services.hazard_scorer import HazardScorer
from app.database import SessionLocal
from app.utils.logger import get_logger

logger = get_logger(__name__)

def test_route_risk():
    """경로 위험도 계산 테스트"""

    # Juba 지역의 테스트 좌표
    # 출발지: Juba 중심부
    start = (4.8594, 31.5713)
    # 목적지: 약간 떨어진 위치
    end = (4.8400, 31.5900)

    print("=" * 80)
    print("경로 위험도 계산 테스트")
    print("=" * 80)
    print(f"출발지: {start}")
    print(f"목적지: {end}")
    print()

    # 1. GraphManager 및 HazardScorer 초기화
    graph_manager = GraphManager()
    graph = graph_manager.get_graph()

    print(f"그래프 정보:")
    print(f"  - 노드 수: {graph.number_of_nodes()}")
    print(f"  - 엣지 수: {graph.number_of_edges()}")

    # 샘플 엣지의 위험도 확인
    sample_edges = list(graph.edges(data=True))[:5]
    print(f"\n샘플 엣지의 위험도:")
    for u, v, data in sample_edges:
        print(f"  - ({u} -> {v}): risk_score={data.get('risk_score', 0)}, length={data.get('length', 0):.4f}km")

    # 2. 위험도 업데이트
    print("\n위험도 업데이트 중...")
    db = SessionLocal()
    try:
        hazard_scorer = HazardScorer(db)
        hazard_scorer.update_graph_risk_scores(graph_manager, exclude_hazard_types=[])
        print("위험도 업데이트 완료")

        # 업데이트 후 샘플 엣지 확인
        print(f"\n업데이트 후 샘플 엣지의 위험도:")
        for u, v, data in sample_edges:
            print(f"  - ({u} -> {v}): risk_score={data.get('risk_score', 0)}, length={data.get('length', 0):.4f}km")
    finally:
        db.close()

    # 3. 경로 계산
    print("\n경로 계산 중...")
    route_calculator = RouteCalculator(graph_manager)
    result = route_calculator.calculate_route(
        start=start,
        end=end,
        preference='balanced',
        transportation_mode='car',
        max_routes=3
    )

    if 'error' in result:
        print(f"오류: {result['error']}")
        return

    # 4. 결과 출력
    routes = result.get('routes', [])
    print(f"\n계산된 경로: {len(routes)}개")
    print()

    for i, route in enumerate(routes, 1):
        print(f"경로 {i} ({route.get('type', 'unknown')}):")
        print(f"  - 거리: {route.get('distance', 0):.2f}km")
        print(f"  - 시간: {route.get('duration', 0)}분")
        print(f"  - 위험도: {route.get('risk_score', 0)}/10")
        print(f"  - 좌표 개수: {len(route.get('polyline', []))}")
        print()

if __name__ == '__main__':
    test_route_risk()
