/**
 * 네비게이션 유틸리티 함수들
 * 거리 계산, 방향 계산, 경로 매칭 등
 */

/**
 * Haversine 공식을 사용한 두 좌표 간의 거리 계산 (미터 단위)
 * @param {number} lat1 - 시작점 위도
 * @param {number} lon1 - 시작점 경도
 * @param {number} lat2 - 끝점 위도
 * @param {number} lon2 - 끝점 경도
 * @returns {number} 거리 (미터)
 */
export const calculateDistance = (lat1, lon1, lat2, lon2) => {
  const R = 6371000; // 지구 반지름 (미터)
  const φ1 = (lat1 * Math.PI) / 180;
  const φ2 = (lat2 * Math.PI) / 180;
  const Δφ = ((lat2 - lat1) * Math.PI) / 180;
  const Δλ = ((lon2 - lon1) * Math.PI) / 180;

  const a =
    Math.sin(Δφ / 2) * Math.sin(Δφ / 2) +
    Math.cos(φ1) * Math.cos(φ2) * Math.sin(Δλ / 2) * Math.sin(Δλ / 2);

  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

  return R * c;
};

/**
 * 두 좌표 간의 방향(bearing) 계산 (도 단위, 0-360)
 * @param {number} lat1 - 시작점 위도
 * @param {number} lon1 - 시작점 경도
 * @param {number} lat2 - 끝점 위도
 * @param {number} lon2 - 끝점 경도
 * @returns {number} 방향 (0=북, 90=동, 180=남, 270=서)
 */
export const calculateBearing = (lat1, lon1, lat2, lon2) => {
  const φ1 = (lat1 * Math.PI) / 180;
  const φ2 = (lat2 * Math.PI) / 180;
  const Δλ = ((lon2 - lon1) * Math.PI) / 180;

  const y = Math.sin(Δλ) * Math.cos(φ2);
  const x =
    Math.cos(φ1) * Math.sin(φ2) -
    Math.sin(φ1) * Math.cos(φ2) * Math.cos(Δλ);

  const θ = Math.atan2(y, x);
  const bearing = ((θ * 180) / Math.PI + 360) % 360;

  return bearing;
};

/**
 * 점과 선분 사이의 최단 거리 계산
 * @param {Object} point - {latitude, longitude}
 * @param {Object} lineStart - {latitude, longitude}
 * @param {Object} lineEnd - {latitude, longitude}
 * @returns {number} 최단 거리 (미터)
 */
export const distanceToLineSegment = (point, lineStart, lineEnd) => {
  const x = point.latitude;
  const y = point.longitude;
  const x1 = lineStart.latitude;
  const y1 = lineStart.longitude;
  const x2 = lineEnd.latitude;
  const y2 = lineEnd.longitude;

  const A = x - x1;
  const B = y - y1;
  const C = x2 - x1;
  const D = y2 - y1;

  const dot = A * C + B * D;
  const lenSq = C * C + D * D;
  let param = -1;

  if (lenSq !== 0) param = dot / lenSq;

  let xx, yy;

  if (param < 0) {
    xx = x1;
    yy = y1;
  } else if (param > 1) {
    xx = x2;
    yy = y2;
  } else {
    xx = x1 + param * C;
    yy = y1 + param * D;
  }

  return calculateDistance(x, y, xx, yy);
};

/**
 * 경로(좌표 배열)에서 현재 위치와 가장 가까운 점 찾기
 * @param {Object} currentLocation - {latitude, longitude}
 * @param {Array} path - [{latitude, longitude}, ...]
 * @returns {Object} {nearestPoint, distance, index}
 */
export const findNearestPointOnPath = (currentLocation, path) => {
  if (!path || path.length === 0) {
    return { nearestPoint: null, distance: Infinity, index: -1 };
  }

  let minDistance = Infinity;
  let nearestPoint = path[0];
  let nearestIndex = 0;

  for (let i = 0; i < path.length - 1; i++) {
    const distance = distanceToLineSegment(
      currentLocation,
      path[i],
      path[i + 1]
    );

    if (distance < minDistance) {
      minDistance = distance;
      nearestIndex = i;

      // 선분 위의 가장 가까운 점 계산
      const projected = projectPointOntoSegment(
        currentLocation,
        path[i],
        path[i + 1]
      );
      nearestPoint = projected;
    }
  }

  return { nearestPoint, distance: minDistance, index: nearestIndex };
};

/**
 * 점을 선분에 투영
 * @param {Object} point - {latitude, longitude}
 * @param {Object} lineStart - {latitude, longitude}
 * @param {Object} lineEnd - {latitude, longitude}
 * @returns {Object} {latitude, longitude}
 */
const projectPointOntoSegment = (point, lineStart, lineEnd) => {
  const x = point.latitude;
  const y = point.longitude;
  const x1 = lineStart.latitude;
  const y1 = lineStart.longitude;
  const x2 = lineEnd.latitude;
  const y2 = lineEnd.longitude;

  const A = x - x1;
  const B = y - y1;
  const C = x2 - x1;
  const D = y2 - y1;

  const dot = A * C + B * D;
  const lenSq = C * C + D * D;
  let param = lenSq !== 0 ? dot / lenSq : -1;

  // 선분 범위 제한
  param = Math.max(0, Math.min(1, param));

  return {
    latitude: x1 + param * C,
    longitude: y1 + param * D,
  };
};

/**
 * 거리를 읽기 쉬운 문자열로 변환
 * @param {number} meters - 거리 (미터)
 * @returns {string} 예: "150m", "1.2km"
 */
export const formatDistance = (meters) => {
  if (meters < 1000) {
    return `${Math.round(meters)}m`;
  }
  return `${(meters / 1000).toFixed(1)}km`;
};

/**
 * 시간(초)을 읽기 쉬운 문자열로 변환
 * @param {number} seconds - 시간 (초)
 * @returns {string} 예: "5분", "1시간 30분"
 */
export const formatDuration = (seconds) => {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);

  if (hours > 0) {
    return minutes > 0 ? `${hours}시간 ${minutes}분` : `${hours}시간`;
  }
  return `${minutes}분`;
};

/**
 * 방향각을 방향 이름으로 변환
 * @param {number} bearing - 방향각 (0-360)
 * @returns {string} 예: "북", "북동", "동" 등
 */
export const bearingToDirection = (bearing) => {
  const directions = ['북', '북동', '동', '남동', '남', '남서', '서', '북서'];
  const index = Math.round(bearing / 45) % 8;
  return directions[index];
};

/**
 * 두 방향각의 차이를 계산하여 회전 방향 판단
 * @param {number} bearing1 - 현재 방향
 * @param {number} bearing2 - 목표 방향
 * @returns {Object} {angle: 차이각(-180 ~ 180), direction: 'left'|'right'|'straight'}
 */
export const calculateTurnDirection = (bearing1, bearing2) => {
  let diff = bearing2 - bearing1;

  // -180 ~ 180 범위로 정규화
  while (diff > 180) diff -= 360;
  while (diff < -180) diff += 360;

  let direction;
  const absDiff = Math.abs(diff);

  if (absDiff < 30) {
    direction = 'straight';
  } else if (absDiff > 150) {
    direction = 'u_turn';
  } else if (diff > 0) {
    direction = absDiff > 90 ? 'sharp_right' : diff > 45 ? 'right' : 'slight_right';
  } else {
    direction = absDiff > 90 ? 'sharp_left' : absDiff > 45 ? 'left' : 'slight_left';
  }

  return { angle: diff, direction };
};

/**
 * 방향에 따른 한국어 안내 문구
 * @param {string} direction - 방향 ('left', 'right', 'straight' 등)
 * @returns {string} 안내 문구
 */
export const getDirectionInstruction = (direction) => {
  const instructions = {
    straight: '직진',
    slight_left: '약간 좌측',
    left: '좌회전',
    sharp_left: '급좌회전',
    slight_right: '약간 우측',
    right: '우회전',
    sharp_right: '급우회전',
    u_turn: '유턴',
  };

  return instructions[direction] || '직진';
};

/**
 * 경로가 완료되었는지 확인
 * @param {Object} currentLocation - {latitude, longitude}
 * @param {Object} destination - {latitude, longitude}
 * @param {number} threshold - 도착 판정 거리 (미터, 기본 20m)
 * @returns {boolean}
 */
export const isDestinationReached = (currentLocation, destination, threshold = 20) => {
  const distance = calculateDistance(
    currentLocation.latitude,
    currentLocation.longitude,
    destination.latitude,
    destination.longitude
  );

  return distance <= threshold;
};

/**
 * 경로 이탈 여부 확인
 * @param {Object} currentLocation - {latitude, longitude}
 * @param {Array} path - [{latitude, longitude}, ...]
 * @param {number} threshold - 이탈 판정 거리 (미터, 기본 50m)
 * @returns {boolean}
 */
export const isOffRoute = (currentLocation, path, threshold = 50) => {
  const { distance } = findNearestPointOnPath(currentLocation, path);
  return distance > threshold;
};

/**
 * 경로 상의 남은 거리 계산
 * @param {Object} currentLocation - {latitude, longitude}
 * @param {Array} path - [{latitude, longitude}, ...]
 * @param {number} currentSegmentIndex - 현재 세그먼트 인덱스
 * @returns {number} 남은 거리 (미터)
 */
export const calculateRemainingDistance = (currentLocation, path, currentSegmentIndex) => {
  if (!path || path.length === 0) return 0;

  let totalDistance = 0;

  // 현재 위치에서 현재 세그먼트의 끝점까지
  if (currentSegmentIndex < path.length - 1) {
    totalDistance += calculateDistance(
      currentLocation.latitude,
      currentLocation.longitude,
      path[currentSegmentIndex + 1].latitude,
      path[currentSegmentIndex + 1].longitude
    );
  }

  // 나머지 세그먼트들
  for (let i = currentSegmentIndex + 1; i < path.length - 1; i++) {
    totalDistance += calculateDistance(
      path[i].latitude,
      path[i].longitude,
      path[i + 1].latitude,
      path[i + 1].longitude
    );
  }

  return totalDistance;
};
