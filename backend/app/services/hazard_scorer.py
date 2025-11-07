"""위험도 계산 서비스 (PostGIS 공간 쿼리 사용)"""
import asyncio
from datetime import datetime
from typing import List, Dict, Callable
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.services.graph_manager import GraphManager
from app.models.hazard import Hazard
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class HazardScorer:
    """도로별 위험도를 주기적으로 계산하여 그래프에 업데이트"""

    def __init__(self, graph_manager: GraphManager, session_factory: Callable = None):
        self.graph_manager = graph_manager
        self.update_interval = settings.hazard_update_interval_seconds
        self.session_factory = session_factory  # SessionLocal을 저장

    async def start_scheduler(self):
        """별도 태스크로 위험도 계산 스케줄러 시작"""
        logger.info("위험도 스케줄러 시작")
        while True:
            await self.update_all_risk_scores()
            await asyncio.sleep(self.update_interval)

    async def update_all_risk_scores(self):
        """모든 도로 엣지의 위험도를 재계산 (PostGIS 공간 쿼리 사용)"""
        logger.info(f"[{datetime.now()}] 위험도 업데이트 시작")

        if self.session_factory is None:
            logger.warning("경고: DB 세션 팩토리가 설정되지 않았습니다")
            return

        # 매번 새로운 DB 세션 생성 (동시성 문제 방지)
        db = self.session_factory()

        try:
            graph = self.graph_manager.get_graph()

            # 1. 활성 위험 정보 조회
            hazards = await self._get_active_hazards(db)
            logger.info(f"활성 위험 정보: {len(hazards)}개")

            if not hazards:
                # 위험 정보가 없으면 모든 엣지 위험도를 0으로 설정
                for u, v, data in graph.edges(data=True):
                    data['risk_score'] = 0
                logger.info(f"위험 정보 없음. 모든 엣지 위험도 0으로 설정")
                return

            # 2. 각 엣지의 위험도 계산
            updated_count = 0
            for u, v, data in graph.edges(data=True):
                risk_score = await self._calculate_edge_risk_postgis(u, v, data, hazards)
                data['risk_score'] = risk_score
                if risk_score > 0:
                    updated_count += 1

            logger.info(f"[{datetime.now()}] 위험도 업데이트 완료: {updated_count}개 엣지에 위험도 적용")

        except Exception as e:
            logger.error(f"위험도 업데이트 오류: {e}", exc_info=True)
        finally:
            # 세션 정리
            db.close()
    
    async def _calculate_edge_risk(self, u, v, data: dict, hazards: List[Dict]) -> int:
        """
        특정 엣지의 위험도 계산
        엣지 전체 선분을 고려하여 위험 범위와 겹치는 부분을 확인
        
        Args:
            u: 출발 노드 ID
            v: 도착 노드 ID
            data: NetworkX 엣지 데이터
            hazards: 활성 위험 정보 리스트
        
        Returns:
            risk_score: 0-100 사이의 위험도
        """
        from app.services.route_calculator import RouteCalculator
        
        if not hazards:
            return 0
        
        graph = self.graph_manager.get_graph()
        source_node = graph.nodes[u]
        target_node = graph.nodes[v]
        
        # 엣지의 시작점과 끝점
        edge_start = (source_node['y'], source_node['x'])  # (lat, lng)
        edge_end = (target_node['y'], target_node['x'])
        
        total_risk = 0
        max_weight = 0
        
        # 각 위험 정보에 대해 엣지 선분과의 최단 거리 계산
        for hazard in hazards:
            hazard_point = (hazard['latitude'], hazard['longitude'])
            hazard_radius_km = hazard['radius']
            
            # 선분에서 점까지의 최단 거리 계산
            # Haversine 공식 사용 (위도/경도는 구면 좌표이므로)
            def point_to_line_distance_km(point, line_start, line_end):
                """점에서 선분까지의 최단 거리 (km) - Haversine 공식 사용"""
                # 선분의 시작점과 끝점에서 점까지의 거리
                dist_to_start = RouteCalculator.calculate_distance_km(point, line_start)
                dist_to_end = RouteCalculator.calculate_distance_km(point, line_end)
                
                # 선분의 길이
                line_length = RouteCalculator.calculate_distance_km(line_start, line_end)
                
                if line_length < 1e-6:  # 선분이 점인 경우
                    return dist_to_start
                
                # 선분을 여러 구간으로 나누어 가장 가까운 점 찾기 (이진 탐색 방식)
                # 정확도를 위해 선분을 10개 구간으로 나눔
                min_dist = min(dist_to_start, dist_to_end)
                
                # 선분을 10개 구간으로 나누어 각 구간의 중점에서 거리 확인
                for i in range(11):
                    t = i / 10.0
                    # 선분 위의 점 (구면 보간)
                    # 위도/경도는 선형 보간이 아니라 구면 보간이 필요하지만,
                    # 짧은 거리에서는 선형 보간도 충분히 정확함
                    mid_lat = line_start[0] + t * (line_end[0] - line_start[0])
                    mid_lng = line_start[1] + t * (line_end[1] - line_start[1])
                    mid_point = (mid_lat, mid_lng)
                    
                    dist = RouteCalculator.calculate_distance_km(point, mid_point)
                    min_dist = min(min_dist, dist)
                
                return min_dist
            
            # 엣지 선분에서 위험 정보까지의 최단 거리
            min_distance = point_to_line_distance_km(hazard_point, edge_start, edge_end)
            
            # 위험 정보의 반경을 고려하여 실제 영향 거리 계산
            # (위험 범위가 엣지와 겹치는 경우를 확인)
            if min_distance <= hazard_radius_km:
                # 위험 범위 내에 있음
                # 거리에 따른 가중치 (가까울수록 높은 영향)
                # 거리 0km: 가중치 1.0
                # 거리 radius km: 가중치 0.5
                effective_distance = min_distance
                weight = 1.0 / (1.0 + (effective_distance / hazard_radius_km))
                
                # 가중치 합산
                total_risk += hazard['risk_score'] * weight
                max_weight += weight
        
        # 가중 평균 계산
        if max_weight > 0:
            avg_risk = total_risk / max_weight
        else:
            avg_risk = 0
        
        # 정규화 (0-100)
        return min(int(avg_risk), 100)
    
    async def _get_active_hazards(self, db: Session) -> List[Dict]:
        """
        활성화된 위험 정보 조회

        Args:
            db: 데이터베이스 세션

        Returns:
            List of dicts with {id, hazard_type, risk_score, lat, lng, radius}
        """
        try:
            # SQLite 호환 쿼리: latitude/longitude만 사용
            query = text("""
                SELECT
                    id,
                    hazard_type,
                    risk_score,
                    latitude,
                    longitude,
                    radius
                FROM hazards
                WHERE (end_date IS NULL OR end_date > datetime('now'))
                  AND verified = TRUE
                ORDER BY risk_score DESC
            """)

            result = db.execute(query)
            hazards = []

            for row in result:
                hazards.append({
                    'id': str(row.id),
                    'hazard_type': row.hazard_type,
                    'risk_score': row.risk_score,
                    'lat': row.latitude,
                    'lng': row.longitude,
                    'radius': row.radius
                })

            return hazards

        except Exception as e:
            logger.error(f"위험 정보 조회 오류: {e}")
            return []

    async def _calculate_edge_risk_postgis(self, u, v, edge_data, hazards: List[Dict]) -> int:
        """
        특정 엣지의 위험도 계산 (PostGIS 공간 쿼리 활용)

        Args:
            u, v: 노드 ID
            edge_data: NetworkX 엣지 데이터
            hazards: 활성 위험 정보 리스트

        Returns:
            risk_score: 0-100 사이의 위험도
        """
        total_risk = 0

        # 그래프에서 노드 좌표 가져오기
        graph = self.graph_manager.get_graph()
        u_data = graph.nodes[u]
        v_data = graph.nodes[v]

        # 엣지 중간점 계산
        mid_lat = (u_data['y'] + v_data['y']) / 2
        mid_lng = (u_data['x'] + v_data['x']) / 2

        # 각 위험 정보에 대해 거리 계산
        from app.utils.geo import haversine_distance

        for hazard in hazards:
            # Haversine 거리 계산
            distance_km = haversine_distance(
                mid_lat, mid_lng,
                hazard['lat'], hazard['lng']
            )

            # 영향 반경 내에 있는지 확인
            if distance_km <= hazard['radius']:
                # 거리에 따른 가중치 (가까울수록 높은 영향)
                weight = 1 / (1 + distance_km)
                total_risk += hazard['risk_score'] * weight

        # 정규화 (0-100)
        return min(int(total_risk), 100)
