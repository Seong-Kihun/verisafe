"""지리/공간 계산 유틸리티"""
import math
from typing import Tuple


def haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Haversine 공식으로 두 좌표 간 거리 계산

    Args:
        lat1, lng1: 첫 번째 지점 (위도, 경도)
        lat2, lng2: 두 번째 지점 (위도, 경도)

    Returns:
        거리 (km)

    Example:
        >>> haversine_distance(4.8670, 31.5880, 4.8500, 31.6000)
        2.45  # km (approximate)
    """
    R = 6371  # 지구 반지름 (km)

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)

    a = math.sin(dlat / 2)**2 + \
        math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def point_to_line_distance(
    point: Tuple[float, float],
    line_start: Tuple[float, float],
    line_end: Tuple[float, float]
) -> float:
    """
    점에서 선분까지의 최단 거리 계산

    Args:
        point: (lat, lng) 점 좌표
        line_start: (lat, lng) 선분 시작점
        line_end: (lat, lng) 선분 끝점

    Returns:
        최단 거리 (km)

    Algorithm:
        1. 선분의 벡터 계산
        2. 점을 선분에 투영
        3. 투영점이 선분 내부인지 확인
        4. 내부: 투영점까지 거리
           외부: 가장 가까운 끝점까지 거리
    """
    p_lat, p_lng = point
    a_lat, a_lng = line_start
    b_lat, b_lng = line_end

    # 선분 벡터 AB
    ab_lat = b_lat - a_lat
    ab_lng = b_lng - a_lng

    # 점 P에서 A로 향하는 벡터 AP
    ap_lat = p_lat - a_lat
    ap_lng = p_lng - a_lng

    # AB 벡터의 길이 제곱
    ab_length_sq = ab_lat**2 + ab_lng**2

    if ab_length_sq == 0:
        # A와 B가 같은 점
        return haversine_distance(p_lat, p_lng, a_lat, a_lng)

    # P를 AB에 투영한 점의 매개변수 t (0 ≤ t ≤ 1이면 선분 내부)
    t = max(0, min(1, (ap_lat * ab_lat + ap_lng * ab_lng) / ab_length_sq))

    # 투영점 좌표
    proj_lat = a_lat + t * ab_lat
    proj_lng = a_lng + t * ab_lng

    # P에서 투영점까지 거리
    return haversine_distance(p_lat, p_lng, proj_lat, proj_lng)
