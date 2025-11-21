"""경로 계산 API (캐싱 적용)"""
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from typing import List
import math
import time
import hashlib

from app.schemas.route import (
    RouteRequest, CalculateRouteResponse, RouteResponse,
    RouteHazardBriefing, RouteHazardInfo
)
from app.schemas.map import HazardResponse
from app.services.graph_manager import GraphManager
from app.services.route_calculator import RouteCalculator
from app.services.hazard_scorer import HazardScorer
from app.services.redis_manager import redis_manager
from app.database import get_db, SessionLocal
from app.models.hazard import Hazard
from app.utils.geo import haversine_distance
from app.utils.logger import get_logger
from app.middleware.rate_limiter import limiter

router = APIRouter()
logger = get_logger(__name__)

# 허용된 위험 유형 목록 (유효성 검증용)
ALLOWED_HAZARD_TYPES = {
    'armed_conflict',
    'protest_riot',
    'checkpoint',
    'road_damage',
    'natural_disaster',
    'flood',
    'landslide',
    'safe_haven',
    'other'
}

# GraphManager 및 RouteCalculator는 요청 시점에 생성
# 모듈 레벨에서 생성하면 초기화 전의 빈 그래프를 참조할 수 있음
def get_graph_manager():
    """GraphManager Singleton 인스턴스 반환"""
    return GraphManager()

def get_route_calculator():
    """RouteCalculator 인스턴스 반환 (최신 GraphManager 사용)"""
    return RouteCalculator(get_graph_manager())

def get_hazard_scorer():
    """HazardScorer 인스턴스 반환"""
    return HazardScorer(get_graph_manager(), session_factory=SessionLocal)


def generate_cache_key(request: RouteRequest) -> str:
    """
    경로 계산 요청에 대한 캐시 키 생성

    Args:
        request: RouteRequest

    Returns:
        캐시 키 (예: "route:hash_abc123")
    """
    # 요청 파라미터를 문자열로 결합 (제외된 위험 유형 포함)
    excluded_str = ','.join(sorted(request.excluded_hazard_types)) if request.excluded_hazard_types else ''
    key_str = f"{request.start.lat},{request.start.lng}|{request.end.lat},{request.end.lng}|{request.preference}|{request.transportation_mode}|{excluded_str}"

    # SHA256 해시 (처음 16자만 사용)
    key_hash = hashlib.sha256(key_str.encode()).hexdigest()[:16]

    return f"route:{key_hash}"


def calculate_route_hazards_count(db: Session, polyline: List[List[float]]) -> int:
    """
    경로 근방 위험 정보 개수 계산

    Args:
        db: 데이터베이스 세션
        polyline: 경로 좌표 배열 [[lat, lng], ...]

    Returns:
        경로 근방 100m 내 위험 정보 개수
    """
    if not polyline or len(polyline) < 2:
        return 0

    from datetime import datetime
    from sqlalchemy import text

    hazard_distance_threshold = 100  # 100m
    now = datetime.utcnow()

    try:
        # PostGIS를 사용한 공간 쿼리
        linestring_coords = ", ".join([f"{lng} {lat}" for lat, lng in polyline])
        linestring_wkt = f"LINESTRING({linestring_coords})"

        query = text("""
            SELECT COUNT(DISTINCT id)
            FROM hazards
            WHERE
                start_date <= :now
                AND (end_date >= :now OR end_date IS NULL)
                AND ST_DWithin(
                    geography(geometry),
                    geography(ST_GeomFromText(:linestring, 4326)),
                    radius * 1000 + :threshold
                )
                AND (
                    ST_Distance(
                        geography(geometry),
                        geography(ST_GeomFromText(:linestring, 4326))
                    ) - (radius * 1000)
                ) <= :threshold
        """)

        result = db.execute(query, {
            "linestring": linestring_wkt,
            "now": now,
            "threshold": hazard_distance_threshold
        })

        count = result.scalar() or 0
        return int(count)

    except Exception as e:
        # PostGIS 쿼리 실패 시 fallback: Python 기반 계산
        logger.warning(f"PostGIS hazard count query failed, using fallback: {e}")

        all_hazards = db.query(Hazard).filter(
            Hazard.start_date <= now,
            (Hazard.end_date >= now) | (Hazard.end_date.is_(None))
        ).all()

        count = 0
        for hazard in all_hazards:
            hazard_point = (hazard.latitude, hazard.longitude)
            min_distance = float('inf')

            for i in range(len(polyline) - 1):
                line_start = tuple(polyline[i])
                line_end = tuple(polyline[i + 1])

                # utils.geo의 point_to_line_distance는 km 단위 반환
                distance = haversine_distance(
                    hazard_point[0], hazard_point[1],
                    line_start[0], line_start[1]
                ) * 1000  # m으로 변환
                min_distance = min(min_distance, distance)

            hazard_radius_meters = hazard.radius * 1000
            effective_distance = max(0, min_distance - hazard_radius_meters)

            if effective_distance <= hazard_distance_threshold:
                count += 1

        return count


@router.post("/calculate", response_model=CalculateRouteResponse)
@limiter.limit("20/minute")  # 1분에 20회 경로 계산으로 제한
async def calculate_route(
    http_request: Request,
    request: RouteRequest,
    db: Session = Depends(get_db)
):
    """
    경로 계산 API (Redis 캐싱 적용, 속도 제한 20/분)

    1. 캐시 확인 → 있으면 즉시 반환
    2. 없으면 계산 → 캐시에 저장 → 반환

    Args:
        http_request: HTTP Request 객체 (속도 제한용)
        request: 경로 계산 요청 (출발지, 목적지, 선호도, 이동 수단, 제외할 위험 유형)
        db: 데이터베이스 세션

    Returns:
        CalculateRouteResponse: 계산된 경로 목록

    Raises:
        HTTPException: 400 - 잘못된 요청 (좌표 범위 오류 등)
        HTTPException: 404 - 경로를 찾을 수 없음
        HTTPException: 429 - 속도 제한 초과
        HTTPException: 500 - 서버 내부 오류
    """
    start_time = time.time()

    # 입력 유효성 검증
    if not (-90 <= request.start.lat <= 90) or not (-180 <= request.start.lng <= 180):
        raise HTTPException(status_code=400, detail="Invalid start coordinates")
    if not (-90 <= request.end.lat <= 90) or not (-180 <= request.end.lng <= 180):
        raise HTTPException(status_code=400, detail="Invalid end coordinates")

    # 1. 캐시 키 생성
    cache_key = generate_cache_key(request)

    # 2. 캐시 확인
    try:
        cached_result = redis_manager.get(cache_key)
        if cached_result:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(f"캐시 히트! 키={cache_key}, 응답 시간={elapsed_ms}ms")
            cached_result['cache_hit'] = True
            cached_result['calculation_time_ms'] = elapsed_ms
            return CalculateRouteResponse(routes=cached_result.get('routes', []))
    except Exception as cache_error:
        # 캐시 오류는 치명적이지 않으므로 로그만 남기고 계속 진행
        logger.warning(f"캐시 조회 실패 (계속 진행): {cache_error}")

    # 3. 캐시 미스 → 경로 계산
    logger.info(f"캐시 미스. 경로 계산 시작...")

    try:
        # 요청 시점에 GraphManager, RouteCalculator, HazardScorer 가져오기
        _hazard_scorer = get_hazard_scorer()
        _route_calculator = get_route_calculator()

        # 경로 계산 전에 위험도 업데이트 (최신 위험 정보 반영)
        # 제외할 위험 유형을 전달하여 필터링된 위험도 계산
        try:
            # 유효성 검증: 허용된 위험 유형만 필터링
            excluded_types = []
            if request.excluded_hazard_types:
                excluded_types = [ht for ht in request.excluded_hazard_types if ht in ALLOWED_HAZARD_TYPES]
                invalid_types = [ht for ht in request.excluded_hazard_types if ht not in ALLOWED_HAZARD_TYPES]
                if invalid_types:
                    logger.warning(f"잘못된 위험 유형 무시됨: {invalid_types}")

            await _hazard_scorer.update_all_risk_scores(excluded_hazard_types=excluded_types)
            if excluded_types:
                logger.info(f"위험도 업데이트 완료 (제외된 유형: {excluded_types}), 경로 계산 시작")
            else:
                logger.info("위험도 업데이트 완료, 경로 계산 시작")
        except Exception as e:
            logger.warning(f"경고: 위험도 업데이트 실패 (경로 계산은 계속 진행): {e}")

        # A* 알고리즘으로 경로 계산 (최대 10개 경로)
        result = _route_calculator.calculate_route(
            start=(request.start.lat, request.start.lng),
            end=(request.end.lat, request.end.lng),
            preference=request.preference,
            transportation_mode=request.transportation_mode,
            max_routes=10
        )
        
        elapsed_ms = int((time.time() - start_time) * 1000)

        # 에러 체크
        if 'error' in result:
            error_msg = result.get('error', '경로를 찾을 수 없습니다')
            logger.warning(f"경로 계산 실패: {error_msg}")

            # 위험 필터가 너무 많으면 안내 메시지 추가
            if excluded_types and len(excluded_types) > 5:
                error_msg += ' (너무 많은 위험 유형을 제외했을 수 있습니다)'

            raise HTTPException(status_code=404, detail=error_msg)

        # 결과 포맷팅 (새로운 형식: routes 배열)
        routes = []
        if 'routes' in result and len(result['routes']) > 0:
            for route_data in result['routes']:
                # 각 경로에 대해 위험 정보 개수 계산
                hazard_count = calculate_route_hazards_count(
                    db=db,
                    polyline=route_data.get('polyline', [])
                )

                routes.append(RouteResponse(
                    id=route_data.get('id', 'unknown'),
                    type=route_data.get('type', 'unknown'),
                    distance=route_data.get('distance', 0),
                    distance_meters=route_data.get('distance_meters', 0),
                    duration=route_data.get('duration', 0),
                    duration_seconds=route_data.get('duration_seconds', 0),
                    risk_score=route_data.get('risk_score', 0),
                    hazard_count=hazard_count,  # 위험 정보 개수 추가
                    transportation_mode=route_data.get('transportation_mode', 'car'),
                    waypoints=route_data.get('waypoints', []),
                    polyline=route_data.get('polyline', [])
                ))
        else:
            # 경로가 하나도 없는 경우
            logger.warning("경로를 찾을 수 없습니다")
            if excluded_types and len(excluded_types) > 5:
                raise HTTPException(
                    status_code=404,
                    detail='경로를 찾을 수 없습니다 (너무 많은 위험 유형을 제외했을 수 있습니다)'
                )
            raise HTTPException(status_code=404, detail='경로를 찾을 수 없습니다')

        # 응답 데이터 구성
        response_data = {
            "routes": [r.dict() for r in routes],
            "cache_hit": False,
            "calculation_time_ms": elapsed_ms
        }

        # 4. 캐시에 저장 (TTL: 5분)
        try:
            redis_manager.set(cache_key, response_data, ttl=300)
            logger.info(f"경로 계산 완료 및 캐시 저장. 키={cache_key}, 시간={elapsed_ms}ms, 경로 수={len(routes)}")
        except Exception as cache_error:
            # 캐시 저장 실패는 치명적이지 않으므로 로그만 남기고 계속 진행
            logger.warning(f"캐시 저장 실패 (결과는 반환됨): {cache_error}")

        # 디버그: 각 경로의 위험도 로깅
        for i, route in enumerate(routes):
            logger.info(f"  경로 {i+1} ({route.type}): 거리={route.distance}km, 위험도={route.risk_score}/10, 시간={route.duration}분")

        return CalculateRouteResponse(routes=routes)

    except Exception as e:
        logger.error(f"경로 계산 오류: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"경로 계산 실패: {str(e)}")


@router.delete("/cache/clear")
async def clear_route_cache():
    """경로 캐시 전체 삭제 (관리자용)"""
    deleted_count = redis_manager.delete_pattern("route:*")
    return {
        "message": f"{deleted_count}개의 경로 캐시가 삭제되었습니다",
        "deleted_count": deleted_count
    }


@router.get("/cache/stats")
async def get_cache_stats():
    """Redis 캐시 통계"""
    return redis_manager.get_stats()


# 중복 함수 제거: haversine_distance와 point_to_line_distance는
# app.utils.geo 모듈에서 import하여 사용


@router.get("/{route_id}/hazards", response_model=RouteHazardBriefing)
@limiter.limit("30/minute")  # 1분에 30회 조회로 제한
async def get_route_hazards(
    http_request: Request,
    route_id: str,
    polyline: str,  # JSON string of [[lat, lng], ...]
    db: Session = Depends(get_db)
):
    """
    경로 근방 위험 정보 조회
    
    Args:
        route_id: 경로 ID
        polyline: 경로 좌표 배열 (JSON string)
    
    Returns:
        RouteHazardBriefing: 경로 근방 위험 정보
    """
    import json
    
    try:
        # polyline 파싱
        route_coordinates = json.loads(polyline)
        if not route_coordinates or len(route_coordinates) < 2:
            raise HTTPException(status_code=400, detail="Invalid polyline")
        
        # 경로 근방 100m 내 위험 정보 찾기 (PostGIS 최적화)
        nearby_hazards = []
        hazard_distance_threshold = 100  # 100m

        from datetime import datetime
        from sqlalchemy import text
        now = datetime.utcnow()

        # PostGIS를 사용한 공간 쿼리 (N+1 문제 해결)
        try:
            # 경로 좌표를 LINESTRING WKT 포맷으로 변환
            # [[lat1, lng1], [lat2, lng2], ...] → "LINESTRING(lng1 lat1, lng2 lat2, ...)"
            linestring_coords = ", ".join([f"{lng} {lat}" for lat, lng in route_coordinates])
            linestring_wkt = f"LINESTRING({linestring_coords})"

            # PostGIS 공간 쿼리:
            # 1. ST_GeomFromText로 LINESTRING 생성
            # 2. ST_DWithin으로 경로 근방 위험 정보 필터링 (지리 기반, 미터 단위)
            # 3. ST_Distance로 정확한 거리 계산 (위험 정보 반경 고려)
            query = text("""
                SELECT
                    id,
                    hazard_type,
                    risk_score,
                    latitude,
                    longitude,
                    radius,
                    description,
                    ST_Distance(
                        geography(geometry),
                        geography(ST_GeomFromText(:linestring, 4326))
                    ) - (radius * 1000) as effective_distance
                FROM hazards
                WHERE
                    start_date <= :now
                    AND (end_date >= :now OR end_date IS NULL)
                    AND ST_DWithin(
                        geography(geometry),
                        geography(ST_GeomFromText(:linestring, 4326)),
                        radius * 1000 + :threshold
                    )
                ORDER BY effective_distance
            """)

            result = db.execute(query, {
                "linestring": linestring_wkt,
                "now": now,
                "threshold": hazard_distance_threshold
            })

            # 결과를 Hazard ID 리스트로 수집
            hazard_ids_with_distances = []
            for row in result:
                effective_distance = max(0, row.effective_distance)
                if effective_distance <= hazard_distance_threshold:
                    hazard_ids_with_distances.append((row.id, effective_distance))

            # Hazard 객체를 한 번에 조회 (N+1 회피)
            if hazard_ids_with_distances:
                hazard_ids = [hid for hid, _ in hazard_ids_with_distances]
                hazards = db.query(Hazard).filter(Hazard.id.in_(hazard_ids)).all()

                # ID를 키로 하는 딕셔너리 생성
                hazards_by_id = {h.id: h for h in hazards}

                # 거리와 함께 결과 구성
                for hazard_id, distance in hazard_ids_with_distances:
                    if hazard_id in hazards_by_id:
                        nearby_hazards.append({
                            'hazard': hazards_by_id[hazard_id],
                            'distance': distance
                        })

        except Exception as e:
            # PostGIS 쿼리 실패 시 fallback: Python 기반 계산
            logger.warning(f"PostGIS query failed, using fallback: {e}")
            from app.utils.geo import point_to_line_distance

            all_hazards = db.query(Hazard).filter(
                Hazard.start_date <= now,
                (Hazard.end_date >= now) | (Hazard.end_date.is_(None))
            ).all()

            for hazard in all_hazards:
                hazard_point = (hazard.latitude, hazard.longitude)
                min_distance = float('inf')

                for i in range(len(route_coordinates) - 1):
                    line_start = tuple(route_coordinates[i])
                    line_end = tuple(route_coordinates[i + 1])

                    # utils.geo의 point_to_line_distance는 km 단위 반환
                    distance = point_to_line_distance(hazard_point, line_start, line_end) * 1000  # m
                    min_distance = min(min_distance, distance)

                hazard_radius_meters = hazard.radius * 1000
                effective_distance = max(0, min_distance - hazard_radius_meters)

                if effective_distance <= hazard_distance_threshold:
                    nearby_hazards.append({
                        'hazard': hazard,
                        'distance': effective_distance
                    })
        
        # 위험 정보를 유형별로 그룹화
        hazards_by_type = {}
        hazard_responses = []
        
        for item in nearby_hazards:
            hazard = item['hazard']
            distance = item['distance']
            
            hazard_type = hazard.hazard_type
            if hazard_type not in hazards_by_type:
                hazards_by_type[hazard_type] = []
            
            hazards_by_type[hazard_type].append({
                'id': str(hazard.id),
                'risk_score': hazard.risk_score,
                'count': 1
            })
            
            # RouteHazardInfo 생성
            hazard_responses.append(RouteHazardInfo(
                hazard_id=hazard.id,
                hazard_type=hazard.hazard_type,
                risk_score=hazard.risk_score,
                latitude=hazard.latitude,
                longitude=hazard.longitude,
                distance_from_route=distance,
                description=hazard.description
            ))
        
        # 요약 정보 생성
        total_hazards = len(nearby_hazards)
        highest_risk_type = None
        highest_risk_count = 0
        
        for hazard_type, items in hazards_by_type.items():
            count = len(items)
            if count > highest_risk_count:
                highest_risk_count = count
                highest_risk_type = hazard_type
        
        summary = {
            "total_hazards": total_hazards,
            "highest_risk_type": highest_risk_type,
            "hazards_by_type_count": {k: len(v) for k, v in hazards_by_type.items()}
        }
        
        return RouteHazardBriefing(
            route_id=route_id,
            hazards=hazard_responses,
            hazards_by_type=hazards_by_type,
            summary=summary
        )

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid polyline JSON")
    except Exception as e:
        logger.error(f"get_route_hazards: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
