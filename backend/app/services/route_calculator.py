"""RouteCalculator: K-Shortest Paths 알고리즘 기반 경로 계산"""
import networkx as nx  # type: ignore[reportMissingModuleSource]
from typing import Tuple, List, Callable
import math
import hashlib
import json
from itertools import islice

# NetworkX의 shortest paths 알고리즘들
astar_path = nx.algorithms.shortest_paths.astar.astar_path
shortest_simple_paths = nx.algorithms.simple_paths.shortest_simple_paths

from app.services.graph_manager import GraphManager
from app.services.redis_manager import redis_manager
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RouteCalculator:
    """A* 알고리즘 기반 경로 계산 (Redis 캐싱 지원)"""

    def __init__(self, graph_manager: GraphManager):
        self.graph_manager = graph_manager

    def _generate_cache_key(self, start: Tuple[float, float], end: Tuple[float, float],
                           preference: str, transportation_mode: str, max_routes: int) -> str:
        """
        경로 계산 캐시 키 생성

        Args:
            start, end, preference, transportation_mode, max_routes

        Returns:
            Redis 캐시 키 (예: "route:abc123...")
        """
        # 좌표를 소수점 5자리로 반올림 (약 1.1m 정확도)
        start_rounded = (round(start[0], 5), round(start[1], 5))
        end_rounded = (round(end[0], 5), round(end[1], 5))

        # 캐시 키 데이터 구성
        cache_data = {
            "start": start_rounded,
            "end": end_rounded,
            "preference": preference,
            "mode": transportation_mode,
            "max_routes": max_routes
        }

        # JSON 직렬화 후 해시
        cache_str = json.dumps(cache_data, sort_keys=True)
        cache_hash = hashlib.md5(cache_str.encode()).hexdigest()

        return f"route:{cache_hash}"
    
    def calculate_route(self, start: Tuple[float, float], end: Tuple[float, float], preference='safe', transportation_mode='car', max_routes=6) -> dict:
        """
        경로 계산 메인 함수 (K-Shortest Paths 알고리즘 사용)

        Args:
            start: (lat, lng)
            end: (lat, lng)
            preference: 'safe' (안전 우선) or 'fast' (빠르기 우선)
            transportation_mode: 'car', 'walking', 'bicycle'
            max_routes: 최대 반환할 경로 수 (기본 6개)

        Returns:
            Dictionary with routes (최대 max_routes개의 다양한 경로)
        """
        # 1. Redis 캐시 확인
        cache_key = self._generate_cache_key(start, end, preference, transportation_mode, max_routes)
        cached_result = redis_manager.get(cache_key)

        if cached_result:
            logger.info(f"캐시 히트: {cache_key[:20]}...")
            cached_result['cached'] = True
            return cached_result

        logger.info(f"캐시 미스 - K-Shortest Paths 경로 계산 시작")

        graph = self.graph_manager.get_graph()

        # 1. 최근접 노드 찾기
        start_node = self.find_nearest_node(graph, start)
        end_node = self.find_nearest_node(graph, end)

        logger.debug(f"출발지: 노드 {start_node}, 목적지: 노드 {end_node}")

        # 2. 이동 수단별 속도 설정 (km/h)
        speed_map = {
            'car': 30,
            'walking': 5,
            'bicycle': 15
        }
        speed = speed_map.get(transportation_mode, 30)

        # 3. 다양한 가중치 함수 정의
        weight_functions = self._create_weight_functions(preference)

        # 4. 여러 가중치 함수로 K-Shortest Paths 계산
        routes = []
        seen_paths = set()  # 중복 경로 방지
        route_idx = 1
        safe_route_added = False  # safe 타입 경로 추가 여부
        fast_route_added = False  # fast 타입 경로 추가 여부

        try:
            # Strategy 1: 각 가중치 함수마다 A* 알고리즘으로 경로 찾기
            for weight_name, weight_func in weight_functions:
                if len(routes) >= max_routes:
                    break

                try:
                    # A* 알고리즘으로 단일 최적 경로 찾기 (가중치 함수 변경으로 다양한 경로 생성)
                    path = astar_path(
                        graph,
                        start_node,
                        end_node,
                        weight=weight_func,
                        heuristic=self.heuristic_function
                    )

                    path_key = tuple(path)

                    # 'safe'와 'fast' 타입은 중복 검사 없이 강제로 추가 (사용자가 선택할 수 있도록)
                    is_primary_route = weight_name in ['safe', 'fast']
                    should_add = False

                    if is_primary_route:
                        # safe/fast 타입은 아직 추가되지 않았으면 무조건 추가
                        if weight_name == 'safe' and not safe_route_added:
                            should_add = True
                            safe_route_added = True
                        elif weight_name == 'fast' and not fast_route_added:
                            should_add = True
                            fast_route_added = True
                    else:
                        # 다른 타입은 중복 경로 제거 및 유사도 체크
                        if path_key not in seen_paths and self._is_diverse_route(path, seen_paths, graph):
                            should_add = True

                    if should_add:
                        seen_paths.add(path_key)
                        formatted = self.format_route(path, graph, speed, transportation_mode)

                        # 경로 타입 및 라벨 결정
                        if weight_name == 'safe':
                            formatted['type'] = 'safe'
                            formatted['id'] = f'safe_route_{route_idx}'
                            formatted['label'] = '가장 안전한 경로'
                        elif weight_name == 'fast':
                            formatted['type'] = 'fast'
                            formatted['id'] = f'fast_route_{route_idx}'
                            formatted['label'] = '가장 빠른 경로'
                        elif weight_name.startswith('balanced'):
                            formatted['type'] = 'balanced'
                            formatted['id'] = f'balanced_route_{route_idx}'
                            formatted['label'] = '균형잡힌 경로'
                        elif weight_name.startswith('distance_priority'):
                            formatted['type'] = 'fast'
                            formatted['id'] = f'distance_route_{route_idx}'
                            formatted['label'] = '빠른 경로'
                        elif weight_name.startswith('very_safe') or weight_name.startswith('ultra_safe'):
                            formatted['type'] = 'safe'
                            formatted['id'] = f'extra_safe_route_{route_idx}'
                            formatted['label'] = '매우 안전한 경로'
                        elif weight_name.startswith('inverse'):
                            formatted['type'] = 'alternative'
                            formatted['id'] = f'alt_route_{route_idx}'
                            formatted['label'] = '대체 경로'
                        else:
                            formatted['type'] = 'alternative'
                            formatted['id'] = f'route_{route_idx}'
                            formatted['label'] = '대체 경로'

                        routes.append(formatted)
                        route_idx += 1
                        logger.debug(f"가중치 '{weight_name}'로 경로 추가: {len(path)} 노드")

                except (nx.NetworkXNoPath, nx.NodeNotFound, Exception) as e:
                    logger.debug(f"가중치 '{weight_name}' 경로 계산 실패: {e}")
                    continue

            # Strategy 2: If we still don't have enough routes, use simple_paths to find more alternatives
            if len(routes) < max_routes:
                logger.info(f"경로가 {len(routes)}개뿐입니다. shortest_simple_paths로 추가 경로 탐색")

                try:
                    # Use length-only weight for finding truly different paths
                    def simple_weight(u, v, d):
                        if hasattr(d, 'values') and not ('length' in d or 'risk_score' in d):
                            edge_values = list(d.values())
                            if edge_values:
                                edge_data = min(edge_values, key=lambda e: e.get('length', float('inf')))
                            else:
                                edge_data = {'length': 1, 'risk_score': 0}
                        else:
                            edge_data = d
                        return edge_data.get('length', 1)

                    # Find up to (max_routes - len(routes)) additional simple paths
                    additional_needed = max_routes - len(routes) + 3  # Get a few extra to filter
                    simple_paths_iter = shortest_simple_paths(graph, start_node, end_node, weight=simple_weight)

                    paths_checked = 0
                    for path in islice(simple_paths_iter, additional_needed):
                        paths_checked += 1
                        if len(routes) >= max_routes:
                            break

                        path_key = tuple(path)
                        if path_key in seen_paths:
                            logger.debug(f"경로 {paths_checked}: 이미 추가된 경로 (중복)")
                            continue

                        if not self._is_diverse_route(path, seen_paths, graph, similarity_threshold=0.85):
                            logger.debug(f"경로 {paths_checked}: 유사도가 너무 높음 (제외)")
                            continue

                        seen_paths.add(path_key)
                        formatted = self.format_route(path, graph, speed, transportation_mode)
                        formatted['type'] = 'alternative'
                        formatted['id'] = f'alt_path_{route_idx}'
                        formatted['label'] = '대체 경로'
                        routes.append(formatted)
                        route_idx += 1
                        logger.info(f"shortest_simple_paths로 경로 추가: {len(path)} 노드 (checked {paths_checked} paths)")

                except (nx.NetworkXNoPath, nx.NetworkXError, Exception) as e:
                    logger.debug(f"shortest_simple_paths 실패: {e}")
                    pass

            # 경로가 없으면 최소한 기본 경로라도 찾기
            if len(routes) == 0:
                logger.warning("다양한 경로를 찾을 수 없어 기본 최단 경로를 반환합니다")

                def basic_weight(u, v, d):
                    # Handle MultiDiGraph edge data
                    if hasattr(d, 'values') and not ('length' in d or 'risk_score' in d):
                        edge_values = list(d.values())
                        if edge_values:
                            edge_data = min(edge_values, key=lambda e: e.get('length', 0))
                        else:
                            edge_data = {'length': 0, 'risk_score': 0}
                    else:
                        edge_data = d
                    return edge_data.get('length', 0) * 1000

                basic_route = astar_path(
                    graph,
                    start_node,
                    end_node,
                    weight=basic_weight,
                    heuristic=self.heuristic_function
                )
                formatted = self.format_route(basic_route, graph, speed, transportation_mode)
                formatted['type'] = 'fast'
                formatted['id'] = 'route_1'
                routes.append(formatted)

            # 최소 2개 경로 보장: Penalty-based 대체 경로 생성
            if len(routes) < 2:
                logger.info(f"경로가 {len(routes)}개뿐입니다. Penalty-based 방식으로 대체 경로 생성 시도")

                # 첫 번째 경로의 엣지 추출
                if len(routes) > 0:
                    first_route_nodes = routes[0].get('waypoints', [])
                    if len(first_route_nodes) > 0:
                        # 첫 번째 경로를 노드 ID로 복원 (역변환)
                        first_path_edges = set()
                        for i in range(len(first_route_nodes) - 1):
                            # 근접 노드 찾기
                            node1 = self.find_nearest_node(graph, first_route_nodes[i])
                            node2 = self.find_nearest_node(graph, first_route_nodes[i + 1])
                            if graph.has_edge(node1, node2):
                                first_path_edges.add((node1, node2))

                        # Penalty-based 가중치 함수 (첫 번째 경로의 엣지에 큰 페널티)
                        penalty = 10000  # 큰 페널티 값

                        def penalty_weight(u, v, d):
                            # Handle MultiDiGraph edge data
                            if hasattr(d, 'values') and not ('length' in d or 'risk_score' in d):
                                edge_values = list(d.values())
                                if edge_values:
                                    edge_data = min(edge_values, key=lambda e: e.get('length', 0))
                                else:
                                    edge_data = {'length': 0, 'risk_score': 0}
                            else:
                                edge_data = d

                            base_weight = edge_data.get('length', 0) * 1000 + edge_data.get('risk_score', 0) * 50
                            if (u, v) in first_path_edges:
                                return base_weight + penalty
                            return base_weight

                        try:
                            alt_path = astar_path(
                                graph,
                                start_node,
                                end_node,
                                weight=penalty_weight,
                                heuristic=self.heuristic_function
                            )

                            # 대체 경로가 첫 번째 경로와 다른지 확인
                            if tuple(alt_path) not in seen_paths:
                                formatted = self.format_route(alt_path, graph, speed, transportation_mode)
                                formatted['type'] = 'alternative'
                                formatted['id'] = f'route_{len(routes) + 1}'
                                routes.append(formatted)
                                logger.info(f"Penalty-based 대체 경로 생성 성공: {len(alt_path)} 노드")
                        except Exception as e:
                            logger.debug(f"Penalty-based 대체 경로 생성 실패: {e}")

            # 경로를 위험 점수 기준으로 정렬 (낮은 순서)
            routes.sort(key=lambda r: r.get('risk_score', 10))

            logger.info(f"총 {len(routes)}개의 다양한 경로 생성 완료")

            result = {"routes": routes[:max_routes], "cached": False}

            # 5. Redis에 결과 캐싱 (TTL: 10분 = 600초)
            redis_manager.set(cache_key, result, ttl=600)
            logger.info(f"결과 캐싱 완료: {cache_key[:20]}...")

            return result

        except nx.NetworkXNoPath:
            return {"error": "경로를 찾을 수 없습니다", "routes": []}

    def _create_weight_functions(self, preference: str) -> List[Tuple[str, Callable]]:
        """
        다양한 가중치 함수 생성

        Args:
            preference: 'safe' or 'fast'

        Returns:
            List of (name, weight_function) tuples
        """

        def _get_edge_data(d):
            """Helper function to extract edge data from MultiDiGraph or DiGraph"""
            # For MultiDiGraph, d is a dict-like object of {key: edge_data}
            # For DiGraph, d is the edge_data dict itself
            if hasattr(d, 'values') and not ('length' in d or 'risk_score' in d):
                # MultiDiGraph: select edge with minimum weight (length + risk)
                # This ensures we pick the best edge among parallel edges
                edge_values = list(d.values())
                if edge_values:
                    return min(edge_values, key=lambda edge: edge.get('length', 0) + edge.get('risk_score', 0) * 0.01)
                return {'length': 0, 'risk_score': 0}  # Fallback
            return d

        weight_functions = []

        # 1. 안전 우선 경로 (거리 + 위험도*100)
        if preference == 'safe':
            weight_functions.append((
                'safe',
                lambda u, v, d: _get_edge_data(d).get('length', 0) * 1000 + _get_edge_data(d).get('risk_score', 0) * 100
            ))

        # 2. 빠른 경로 (거리만)
        weight_functions.append((
            'fast',
            lambda u, v, d: _get_edge_data(d).get('length', 0) * 1000
        ))

        # 3. 균형 잡힌 경로들 (다양한 위험도 가중치)
        weight_functions.extend([
            ('balanced_light', lambda u, v, d: _get_edge_data(d).get('length', 0) * 1000 + _get_edge_data(d).get('risk_score', 0) * 30),
            ('balanced_medium', lambda u, v, d: _get_edge_data(d).get('length', 0) * 1000 + _get_edge_data(d).get('risk_score', 0) * 70),
            ('balanced_heavy', lambda u, v, d: _get_edge_data(d).get('length', 0) * 1000 + _get_edge_data(d).get('risk_score', 0) * 150),
        ])

        # 4. 거리 우선 경로들 (더 짧은 거리를 선호, 위험도는 덜 중요)
        weight_functions.extend([
            ('distance_priority_1', lambda u, v, d: _get_edge_data(d).get('length', 0) * 1000 + _get_edge_data(d).get('risk_score', 0) * 10),
            ('distance_priority_2', lambda u, v, d: _get_edge_data(d).get('length', 0) * 1000 + _get_edge_data(d).get('risk_score', 0) * 5),
        ])

        # 5. 안전 우선 경로들 (위험도를 매우 중요시)
        weight_functions.extend([
            ('very_safe', lambda u, v, d: _get_edge_data(d).get('length', 0) * 1000 + _get_edge_data(d).get('risk_score', 0) * 200),
            ('ultra_safe', lambda u, v, d: _get_edge_data(d).get('length', 0) * 1000 + _get_edge_data(d).get('risk_score', 0) * 500),
        ])

        # 6. 반대 가중치 (위험도가 낮은 곳을 피하는 경로 - 특이 경로 생성)
        # 이것은 일부러 다른 경로를 생성하기 위한 트릭
        weight_functions.extend([
            ('inverse_risk_1', lambda u, v, d: _get_edge_data(d).get('length', 0) * 1000 + max(0, (100 - _get_edge_data(d).get('risk_score', 0))) * 20),
            ('inverse_risk_2', lambda u, v, d: _get_edge_data(d).get('length', 0) * 1000 + max(0, (100 - _get_edge_data(d).get('risk_score', 0))) * 50),
        ])

        return weight_functions

    def _is_diverse_route(self, new_path: List, seen_paths: set, graph: nx.DiGraph, similarity_threshold: float = 0.7) -> bool:
        """
        새 경로가 기존 경로들과 충분히 다른지 확인 (경로 유사도 체크)

        Args:
            new_path: 새로운 경로 (노드 리스트)
            seen_paths: 이미 추가된 경로들 (path_key set)
            graph: NetworkX 그래프
            similarity_threshold: 유사도 임계값 (0.7 = 70% 이상 같으면 제외, 30% 이상 달라야 함)
                                Default changed from 0.5 to 0.7 to allow more diverse routes

        Returns:
            True if 충분히 다르면, False if 너무 유사하면
        """
        new_path_set = set(new_path)

        for seen_path_key in seen_paths:
            seen_path_set = set(seen_path_key)

            # Jaccard 유사도 계산: |A ∩ B| / |A ∪ B|
            intersection = len(new_path_set & seen_path_set)
            union = len(new_path_set | seen_path_set)

            if union == 0:
                continue

            similarity = intersection / union

            # 유사도가 임계값 이상이면 너무 비슷함
            if similarity >= similarity_threshold:
                logger.debug(f"경로 유사도 {similarity:.2f} >= {similarity_threshold}, 제외")
                return False

        return True

    def find_nearest_node(self, graph: nx.DiGraph, point: Tuple[float, float], use_postgis: bool = False, db=None):
        """
        좌표에서 가장 가까운 그래프 노드 찾기

        ✅ 최적화: PostGIS 공간 인덱스 선택 가능
        - use_postgis=True + db: O(log N) - GIST 인덱스 활용
        - use_postgis=False: O(N) - 메모리 순회

        Args:
            graph: NetworkX 그래프
            point: (lat, lng) 좌표
            use_postgis: PostGIS 최적화 사용 여부
            db: Database session (use_postgis=True일 때 필수)

        Returns:
            nearest_node_id
        """
        # PostGIS 최적화 경로
        if use_postgis and db is not None:
            try:
                from sqlalchemy import text
                lat, lng = point

                # KNN (<->) 연산자: 공간 인덱스 자동 활용
                result = db.execute(text("""
                    SELECT osm_id
                    FROM roads
                    WHERE geometry IS NOT NULL
                    ORDER BY geometry <-> ST_SetSRID(ST_MakePoint(:lng, :lat), 4326)::geography
                    LIMIT 1
                """), {"lng": lng, "lat": lat})

                row = result.first()
                if row and row[0]:
                    # OSM ID로 NetworkX 노드 매칭
                    for node, data in graph.nodes(data=True):
                        if data.get('osmid') == row[0]:
                            return node
                    logger.debug(f"PostGIS OSM ID {row[0]} 매칭 실패, 폴백")

            except Exception as e:
                logger.debug(f"PostGIS 오류: {e}, 메모리 순회로 폴백")

        # 메모리 순회 (폴백 또는 기본 동작)
        nearest_node = None
        min_distance = float('inf')

        for node, data in graph.nodes(data=True):
            node_point = (data['y'], data['x'])  # OSMnx는 (lat, lon) 순서
            distance = self.calculate_distance_km(point, node_point)

            if distance < min_distance:
                min_distance = distance
                nearest_node = node

        return nearest_node
    
    def heuristic_function(self, u, v):
        """
        A* 휴리스틱 함수 (유클리드 거리)
        
        Args:
            u: 노드 ID
            v: 노드 ID
        
        Returns:
            estimated_distance: 두 노드 간 추정 거리(km)
        """
        graph = self.graph_manager.get_graph()
        u_data = graph.nodes[u]
        v_data = graph.nodes[v]
        
        u_point = (u_data['y'], u_data['x'])
        v_point = (v_data['y'], v_data['x'])
        
        return self.calculate_distance_km(u_point, v_point)
    
    def format_route(self, route_nodes: list, graph: nx.DiGraph, speed: float, transportation_mode: str):
        """
        노드 리스트를 경로 정보로 변환
        
        Args:
            route_nodes: 노드 ID 리스트
            graph: NetworkX 그래프
            speed: 이동 속도 (km/h)
            transportation_mode: 이동 수단
        
        Returns:
            Dictionary with polyline, distance, duration, risk_score, waypoints
        """
        total_distance = 0
        total_risk = 0
        waypoints = []  # 주요 경로점

        logger.debug(f"format_route 시작: {len(route_nodes)}개 노드")
        
        for i in range(len(route_nodes) - 1):
            u = route_nodes[i]
            v = route_nodes[i + 1]

            try:
                # Handle MultiDiGraph - get the edge with minimum length (or first edge if single)
                edge_dict = graph[u][v]

                # For MultiDiGraph, edge_dict is an AtlasView with keys {0: edge_data, 1: edge_data, ...}
                # For DiGraph, edge_dict is the edge_data dict itself
                if hasattr(edge_dict, 'values') and not ('length' in edge_dict or 'risk_score' in edge_dict):
                    # MultiDiGraph: select edge with minimum length
                    edge_values = list(edge_dict.values())
                    if edge_values:
                        edge_data = min(edge_values, key=lambda d: d.get('length', float('inf')))
                    else:
                        logger.debug(f"경고: 엣지 ({u}, {v})에 데이터가 없음")
                        continue
                else:
                    # DiGraph: use directly
                    edge_data = edge_dict
            except KeyError:
                logger.debug(f"경고: 엣지 ({u}, {v})를 찾을 수 없음")
                continue

            # 거리 누적 (graph_manager에서 이미 km 단위로 변환됨)
            distance_km = edge_data.get('length', 0)
            if distance_km == 0:
                # length가 없으면 Haversine 공식으로 계산
                u_data = graph.nodes[u]
                v_data = graph.nodes[v]
                u_point = (u_data['y'], u_data['x'])
                v_point = (v_data['y'], v_data['x'])
                distance_km = self.calculate_distance_km(u_point, v_point)
                logger.debug(f"엣지 {i}: length 없음, Haversine 계산: {distance_km:.4f}km")

            total_distance += distance_km

            # 위험도 누적 (거리 가중)
            risk = edge_data.get('risk_score', 0)
            if i < 5:  # 처음 5개 엣지만 로그 출력
                logger.debug(f"엣지 {i}: ({u} -> {v}), 거리={distance_km:.4f}km, 위험도={risk}")
            total_risk += risk * distance_km

        logger.debug(f"총 거리: {total_distance:.4f}km, 총 위험도(가중합): {total_risk:.4f}, 속도: {speed}km/h")
        
        # 위험도 평균 계산 (거리 가중) - 0-10 스케일로 정규화
        # 참고: hazard_scorer에서 risk_score는 0-100 스케일로 저장됨
        if total_distance > 0:
            avg_risk = total_risk / total_distance
            logger.debug(f"평균 위험도 (0-100 스케일): {avg_risk:.4f}")
            # 위험 점수를 0-10 스케일로 변환 (정밀도 유지)
            # 수정 전: int(round(avg_risk / 10)) → 정밀도 손실 (3.5 → 4)
            # 수정 후: round(avg_risk / 10) → 정밀도 유지 (3.5 → 4, 하지만 반올림 규칙 적용)
            normalized_risk = min(10, max(0, round(avg_risk / 10)))
            logger.debug(f"정규화된 위험도 (0-10 스케일): {normalized_risk}")
        else:
            normalized_risk = 0
            logger.debug(f"경고: 총 거리가 0이므로 위험도 계산 불가")

        # 소요 시간 계산 (이동 수단별 속도 사용)
        duration_minutes = int((total_distance / speed) * 60) if speed > 0 else 0
        duration_seconds = int((total_distance / speed) * 3600) if speed > 0 else 0

        logger.debug(f"계산된 값: 거리={total_distance:.4f}km, 시간={duration_minutes}분 ({duration_seconds}초), 위험도={normalized_risk}")
        
        # Polyline 생성 (경로 좌표 리스트 - 모든 노드)
        polyline = [
            [graph.nodes[node]['y'], graph.nodes[node]['x']]
            for node in route_nodes
        ]
        
        # Waypoints 생성 (주요 경로점만 - 시작, 끝, 중간 주요 지점)
        # 간단히 시작, 중간, 끝점만 선택 (나중에 개선 가능)
        if len(route_nodes) > 2:
            waypoints = [
                [graph.nodes[route_nodes[0]]['y'], graph.nodes[route_nodes[0]]['x']],  # 시작
                [graph.nodes[route_nodes[len(route_nodes)//2]]['y'], graph.nodes[route_nodes[len(route_nodes)//2]]['x']],  # 중간
                [graph.nodes[route_nodes[-1]]['y'], graph.nodes[route_nodes[-1]]['x']]  # 끝
            ]
        else:
            waypoints = polyline
        
        return {
            "polyline": polyline,
            "waypoints": waypoints,
            "distance": round(total_distance, 2),  # km
            "distance_meters": round(total_distance * 1000, 0),  # m
            "duration": duration_minutes,           # minutes
            "duration_seconds": duration_seconds,   # seconds
            "risk_score": normalized_risk,          # 0-10 스케일
            "transportation_mode": transportation_mode
        }
    
    @staticmethod
    def calculate_distance_km(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """
        두 좌표 간 거리 계산 (Haversine 공식)

        Args:
            point1: (lat, lng)
            point2: (lat, lng)

        Returns:
            distance in km
        """
        from app.utils.geo import haversine_distance
        return haversine_distance(point1[0], point1[1], point2[0], point2[1])
