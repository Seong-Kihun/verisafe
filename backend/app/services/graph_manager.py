"""GraphManager: 도로 네트워크 그래프를 메모리에 유지하는 Singleton"""
from typing import Optional
import networkx as nx
from sqlalchemy.orm import Session
from app.database import get_db
from app.utils.logger import get_logger

logger = get_logger(__name__)


class GraphManager:
    """도로 네트워크 그래프를 메모리에 유지하는 Singleton"""
    
    _instance = None
    _graph: Optional[nx.DiGraph] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def initialize(self, db: Session):
        """서버 시작 시 한 번만 실행"""
        if self._graph is None:
            logger.info("주바 도로 네트워크 그래프 로딩 중...")
            self._graph = await self._load_from_database(db)

            # 모든 엣지에 위험도 속성 초기화
            for u, v, data in self._graph.edges(data=True):
                data['risk_score'] = 0

            logger.info(f"완료: {len(self._graph.nodes)}개 노드, {len(self._graph.edges)}개 엣지")
    
    async def _load_from_database(self, db: Session):
        """
        DB에서 도로 데이터를 로드하여 NetworkX 그래프 생성
        
        OSMnx로 실제 주바 도로 데이터 로드 (실패 시 더미 데이터)
        """
        try:
            import osmnx as ox
            import asyncio

            logger.info("OSMnx로 주바 도로 데이터 다운로드 중...")
            # 타임아웃 설정 (30초) - OSMnx 호출이 너무 오래 걸리면 더미 데이터로 폴백
            try:
                # 동기 함수를 비동기로 실행 (타임아웃 적용)
                loop = asyncio.get_event_loop()
                G = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: ox.graph_from_place("Juba, South Sudan", network_type='drive')),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                logger.warning("OSMnx 타임아웃 (30초) - 더미 그래프로 폴백")
                raise Exception("OSMnx timeout")
            
            # 길이를 km 단위로 변환
            edges_with_length = 0
            edges_without_length = 0
            for u, v, data in G.edges(data=True):
                if 'length' in data and data['length'] is not None:
                    data['length'] = data['length'] / 1000  # m → km
                    edges_with_length += 1
                else:
                    # length가 없으면 Haversine 공식으로 계산
                    import math
                    R = 6371  # 지구 반지름 (km)
                    u_data = G.nodes[u]
                    v_data = G.nodes[v]
                    lat1, lon1 = math.radians(u_data['y']), math.radians(u_data['x'])
                    lat2, lon2 = math.radians(v_data['y']), math.radians(v_data['x'])
                    dlat = lat2 - lat1
                    dlon = lon2 - lon1
                    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
                    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                    distance_km = R * c
                    data['length'] = distance_km
                    edges_without_length += 1

            logger.info(f"OSM 데이터 로드 성공: {len(G.nodes)}개 노드, {len(G.edges)}개 엣지")
            logger.info(f"엣지 length 속성: {edges_with_length}개 있음, {edges_without_length}개 계산됨")
            return G

        except Exception as e:
            logger.error(f"OSM 로드 실패: {e}")
            logger.info("더미 그래프로 폴백...")

            # 실전 더미 그래프 생성: 위험 정보와 겹치는 실제 좌표 사용
            # 주바 중심부의 실제 도로 네트워크를 시뮬레이션
            G = nx.DiGraph()

            # 좌표계: (lat, lon)
            # 주바 중심: 4.8594°N, 31.5713°E
            # 더미 데이터 위험 정보 범위: lat 4.848~4.865, lng 31.565~31.59

            # 노드 추가: 위험 지역을 포함하도록 더 많은 노드 생성
            nodes = []
            node_id = 1

            # 그리드 형태로 노드 생성 (10x10 = 100 nodes)
            lat_start, lat_end = 4.845, 4.870
            lng_start, lng_end = 31.560, 31.595
            lat_step = (lat_end - lat_start) / 9  # 10개 구간
            lng_step = (lng_end - lng_start) / 9

            node_positions = {}  # {node_id: (lat, lng)}

            for i in range(10):
                for j in range(10):
                    lat = lat_start + i * lat_step
                    lng = lng_start + j * lng_step
                    nodes.append((node_id, {'y': lat, 'x': lng, 'name': f'Node_{node_id}'}))
                    node_positions[node_id] = (lat, lng)
                    node_id += 1

            G.add_nodes_from(nodes)

            # 엣지 추가: 그리드 형태의 도로 네트워크 (가로/세로 연결)
            edges = []
            for i in range(10):
                for j in range(10):
                    current_id = i * 10 + j + 1

                    # 오른쪽 노드와 연결
                    if j < 9:
                        right_id = current_id + 1
                        distance = self._calculate_haversine_km(
                            node_positions[current_id][0], node_positions[current_id][1],
                            node_positions[right_id][0], node_positions[right_id][1]
                        )
                        edges.append((current_id, right_id, {'length': distance, 'name': f'Road_{current_id}_{right_id}'}))
                        edges.append((right_id, current_id, {'length': distance, 'name': f'Road_{right_id}_{current_id}'}))

                    # 아래 노드와 연결
                    if i < 9:
                        bottom_id = current_id + 10
                        distance = self._calculate_haversine_km(
                            node_positions[current_id][0], node_positions[current_id][1],
                            node_positions[bottom_id][0], node_positions[bottom_id][1]
                        )
                        edges.append((current_id, bottom_id, {'length': distance, 'name': f'Road_{current_id}_{bottom_id}'}))
                        edges.append((bottom_id, current_id, {'length': distance, 'name': f'Road_{bottom_id}_{current_id}'}))

            G.add_edges_from(edges)

            logger.info(f"더미 그래프 생성 완료: {len(G.nodes)}개 노드, {len(G.edges)}개 엣지")
            return G

    def _calculate_haversine_km(self, lat1, lng1, lat2, lng2):
        """Haversine 거리 계산 (km)"""
        import math
        R = 6371  # 지구 반지름 (km)
        lat1_rad, lng1_rad = math.radians(lat1), math.radians(lng1)
        lat2_rad, lng2_rad = math.radians(lat2), math.radians(lng2)
        dlat = lat2_rad - lat1_rad
        dlng = lng2_rad - lng1_rad
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c
    
    def get_graph(self) -> nx.DiGraph:
        """메모리의 그래프 반환 (초기화되지 않은 경우 더미 그래프 반환)"""
        if self._graph is None:
            logger.warning("경고: 그래프가 아직 초기화되지 않았습니다. 더미 그래프를 반환합니다.")
            # 더미 그래프 즉시 생성 (초기화 완료 전까지 사용)
            return self._create_dummy_graph()
        return self._graph
    
    def _create_dummy_graph(self) -> nx.DiGraph:
        """더미 그래프 생성 (초기화 전 임시 사용)"""
        G = nx.DiGraph()
        nodes = [
            (1, {'y': 4.8670, 'x': 31.5880, 'name': 'Juba Airport'}),
            (2, {'y': 4.8550, 'x': 31.5900, 'name': 'Midpoint 1'}),
            (3, {'y': 4.8500, 'x': 31.6000, 'name': 'City Hall'}),
            (4, {'y': 4.8470, 'x': 31.5800, 'name': 'Hospital'}),
        ]
        G.add_nodes_from(nodes)
        edges = [
            (1, 2, {'length': 1.5, 'name': 'Airport Road', 'risk_score': 0}),
            (2, 3, {'length': 0.8, 'name': 'City Hall Road', 'risk_score': 0}),
            (1, 4, {'length': 2.0, 'name': 'Hospital Road', 'risk_score': 0}),
        ]
        G.add_edges_from(edges)
        return G
    
    def update_edge_risk(self, node_u, node_v, risk_score: int):
        """특정 엣지의 위험도 업데이트"""
        if self._graph and self._graph.has_edge(node_u, node_v):
            self._graph[node_u][node_v]['risk_score'] = risk_score
        else:
            logger.debug(f"경고: 엣지 ({node_u}, {node_v})를 찾을 수 없습니다.")

    async def store_roads_in_db(self, db: Session):
        """
        도로 네트워크를 PostgreSQL + PostGIS에 저장

        Phase 1: 초기 구현
        - NetworkX 그래프 → PostGIS LINESTRING 변환
        - 공간 인덱스 활용 가능
        """
        from app.models.road import Road
        from sqlalchemy import text

        if self._graph is None:
            logger.warning("그래프가 로드되지 않아 저장할 수 없습니다.")
            return

        logger.info(f"도로 {len(self._graph.edges)}개를 DB에 저장 중...")

        # 기존 도로 데이터 삭제 (재초기화)
        db.query(Road).delete()

        count = 0
        for u, v, data in self._graph.edges(data=True):
            try:
                u_data = self._graph.nodes[u]
                v_data = self._graph.nodes[v]

                # LINESTRING 생성 (WKT 형식)
                linestring_wkt = f"LINESTRING({u_data['x']} {u_data['y']}, {v_data['x']} {v_data['y']})"

                road = Road(
                    osm_id=data.get('osmid'),
                    name=data.get('name'),
                    road_type=data.get('highway', 'unknown'),
                    length_km=data.get('length', 0)
                )

                # geometry 설정 (raw SQL)
                db.add(road)
                db.flush()  # ID 생성

                db.execute(
                    text("UPDATE roads SET geometry = ST_GeomFromText(:wkt, 4326) WHERE id = :road_id"),
                    {"wkt": linestring_wkt, "road_id": str(road.id)}
                )

                count += 1
                if count % 500 == 0:
                    logger.info(f"{count}개 저장 완료...")
                    db.commit()

            except Exception as e:
                logger.error(f"도로 저장 오류: {e}")
                continue

        db.commit()
        logger.info(f"총 {count}개 도로 저장 완료")

    async def load_hazards_to_graph(self, db: Session):
        """
        위험 정보를 로드하여 그래프 엣지에 위험도 적용

        SQLite/PostgreSQL 모두 지원:
        - SQLite: Python haversine 거리 계산
        - PostgreSQL: PostGIS 공간 쿼리 (향후 최적화)
        """
        from app.models.hazard import Hazard
        from datetime import datetime
        import math

        if self._graph is None:
            logger.warning("그래프가 로드되지 않았습니다.")
            return

        logger.info("위험 정보 로딩 중...")

        # 활성 위험 요소만 가져오기
        hazards = db.query(Hazard).filter(
            (Hazard.end_date == None) | (Hazard.end_date > datetime.utcnow())
        ).all()

        logger.info(f"{len(hazards)}개 활성 위험 요소 발견")

        # 모든 엣지의 위험도 초기화
        for u, v, data in self._graph.edges(data=True):
            data['risk_score'] = 0

        # Haversine 거리 계산 함수 (km 단위)
        def haversine_distance(lat1, lon1, lat2, lon2):
            R = 6371  # 지구 반지름 (km)
            lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
            lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad
            a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            return R * c

        # 각 위험 요소에 대해 영향받는 도로 찾기 (Python 기반 거리 계산)
        for hazard in hazards:
            try:
                hazard_lat = hazard.latitude
                hazard_lng = hazard.longitude
                hazard_radius = hazard.radius

                # 모든 엣지를 순회하며 거리 계산
                for u, v, data in self._graph.edges(data=True):
                    try:
                        # 엣지의 중점 좌표 계산
                        u_data = self._graph.nodes[u]
                        v_data = self._graph.nodes[v]
                        edge_center_lat = (u_data['y'] + v_data['y']) / 2
                        edge_center_lng = (u_data['x'] + v_data['x']) / 2

                        # 위험 요소와 엣지 중점 간의 거리 계산
                        distance_km = haversine_distance(
                            hazard_lat, hazard_lng,
                            edge_center_lat, edge_center_lng
                        )

                        # 반경 내에 있는 경우에만 위험도 적용
                        if distance_km <= hazard_radius:
                            # 거리 감쇠 계산 (가까울수록 높은 위험도)
                            decay_factor = max(0, 1 - (distance_km / hazard_radius))
                            risk_contribution = int(hazard.risk_score * decay_factor)

                            current_risk = data.get('risk_score', 0)
                            data['risk_score'] = min(100, current_risk + risk_contribution)

                    except (KeyError, TypeError) as e:
                        # 노드 좌표가 없는 경우 건너뛰기
                        continue

            except Exception as e:
                logger.error(f"위험 정보 적용 오류: {e}")
                continue

        logger.info("위험 정보 로딩 완료")
