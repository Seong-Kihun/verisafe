/**
 * 경로를 네비게이션 세그먼트로 변환
 * 턴바이턴 안내를 위한 세그먼트 생성 로직
 */

import {
  calculateDistance,
  calculateBearing,
  calculateTurnDirection,
  getDirectionInstruction,
} from '../utils/navigationUtils';

/**
 * 경로 데이터를 네비게이션 세그먼트로 변환
 * @param {Object} route - 백엔드에서 받은 경로 데이터
 * @returns {Array} 세그먼트 배열
 */
export const createNavigationSegments = (route) => {
  if (!route || !route.coordinates || route.coordinates.length < 2) {
    console.error('[RouteSegments] Invalid route data');
    return [];
  }

  const coordinates = route.coordinates;
  const segments = [];

  // 최소 거리 임계값 (미터) - 이보다 짧은 세그먼트는 병합
  const MIN_SEGMENT_DISTANCE = 50;
  // 방향 변화 임계값 (도) - 이보다 큰 방향 변화만 새 세그먼트로
  const BEARING_CHANGE_THRESHOLD = 30;

  let currentSegmentStart = 0;
  let previousBearing = null;

  for (let i = 1; i < coordinates.length; i++) {
    const coord1 = coordinates[i - 1];
    const coord2 = coordinates[i];

    const bearing = calculateBearing(
      coord1.latitude,
      coord1.longitude,
      coord2.latitude,
      coord2.longitude
    );

    // 첫 번째 포인트거나 방향이 크게 변경된 경우
    const shouldCreateSegment =
      i === 1 ||
      i === coordinates.length - 1 ||
      (previousBearing !== null &&
        Math.abs(bearing - previousBearing) > BEARING_CHANGE_THRESHOLD);

    if (shouldCreateSegment && i > currentSegmentStart) {
      const segmentCoords = coordinates.slice(currentSegmentStart, i + 1);
      const segmentDistance = calculateSegmentDistance(segmentCoords);

      // 충분한 거리가 있을 때만 세그먼트 생성
      if (segmentDistance >= MIN_SEGMENT_DISTANCE || i === coordinates.length - 1) {
        const segment = createSegment(
          segments.length,
          segmentCoords,
          previousBearing,
          bearing,
          route.hazards
        );

        segments.push(segment);
        currentSegmentStart = i;
      }
    }

    previousBearing = bearing;
  }

  // 마지막 세그먼트가 없으면 전체를 하나의 세그먼트로
  if (segments.length === 0) {
    segments.push(
      createSegment(0, coordinates, null, null, route.hazards, true)
    );
  }

  return segments;
};

/**
 * 단일 세그먼트 생성
 * @param {number} index - 세그먼트 인덱스
 * @param {Array} coordinates - 세그먼트의 좌표 배열
 * @param {number} previousBearing - 이전 방향
 * @param {number} currentBearing - 현재 방향
 * @param {Array} allHazards - 전체 경로의 위험 정보
 * @param {boolean} isSingleSegment - 단일 세그먼트 여부
 * @returns {Object} 세그먼트 객체
 */
const createSegment = (
  index,
  coordinates,
  previousBearing,
  currentBearing,
  allHazards = [],
  isSingleSegment = false
) => {
  const distance = calculateSegmentDistance(coordinates);
  const startPoint = coordinates[0];
  const endPoint = coordinates[coordinates.length - 1];

  // 방향 결정
  let direction = 'straight';
  let instruction = '직진';

  if (!isSingleSegment && previousBearing !== null && currentBearing !== null) {
    const turn = calculateTurnDirection(previousBearing, currentBearing);
    direction = turn.direction;
    instruction = getDirectionInstruction(direction);
  }

  // 이 세그먼트 근처의 위험 찾기
  const segmentHazards = findHazardsNearSegment(coordinates, allHazards);

  return {
    id: `segment_${index}`,
    index,
    coordinates,
    distance: Math.round(distance),
    direction,
    instruction,
    startPoint,
    endPoint,
    hazards: segmentHazards,
    bearing: currentBearing,
  };
};

/**
 * 세그먼트의 총 거리 계산
 * @param {Array} coordinates - 좌표 배열
 * @returns {number} 거리 (미터)
 */
const calculateSegmentDistance = (coordinates) => {
  let totalDistance = 0;

  for (let i = 0; i < coordinates.length - 1; i++) {
    const coord1 = coordinates[i];
    const coord2 = coordinates[i + 1];

    totalDistance += calculateDistance(
      coord1.latitude,
      coord1.longitude,
      coord2.latitude,
      coord2.longitude
    );
  }

  return totalDistance;
};

/**
 * 세그먼트 근처의 위험 찾기
 * @param {Array} coordinates - 세그먼트 좌표
 * @param {Array} hazards - 전체 위험 배열
 * @param {number} threshold - 근처 판정 거리 (미터)
 * @returns {Array} 근처 위험 배열
 */
const findHazardsNearSegment = (coordinates, hazards, threshold = 100) => {
  if (!hazards || hazards.length === 0) return [];

  const nearbyHazards = [];

  for (const hazard of hazards) {
    // 세그먼트의 각 포인트와 위험의 거리 확인
    for (const coord of coordinates) {
      const distance = calculateDistance(
        coord.latitude,
        coord.longitude,
        hazard.latitude,
        hazard.longitude
      );

      if (distance <= threshold) {
        nearbyHazards.push({
          ...hazard,
          distanceFromSegmentStart: calculateDistance(
            coordinates[0].latitude,
            coordinates[0].longitude,
            hazard.latitude,
            hazard.longitude
          ),
        });
        break; // 하나의 포인트에서 찾았으면 다음 위험으로
      }
    }
  }

  // 세그먼트 시작점으로부터 거리 순 정렬
  return nearbyHazards.sort((a, b) => a.distanceFromSegmentStart - b.distanceFromSegmentStart);
};

/**
 * 현재 위치에서 다음 세그먼트 찾기
 * @param {Object} currentLocation - {latitude, longitude}
 * @param {Array} segments - 세그먼트 배열
 * @param {number} currentSegmentIndex - 현재 세그먼트 인덱스
 * @returns {number} 다음 세그먼트 인덱스
 */
export const findCurrentSegmentIndex = (currentLocation, segments, currentSegmentIndex = 0) => {
  if (!segments || segments.length === 0) return -1;

  // 현재 세그먼트부터 검색
  for (let i = currentSegmentIndex; i < segments.length; i++) {
    const segment = segments[i];
    const distanceToEnd = calculateDistance(
      currentLocation.latitude,
      currentLocation.longitude,
      segment.endPoint.latitude,
      segment.endPoint.longitude
    );

    // 끝점까지 거리가 세그먼트 길이보다 작으면 아직 이 세그먼트 안
    if (distanceToEnd < segment.distance + 50) {
      return i;
    }
  }

  // 마지막 세그먼트 반환
  return segments.length - 1;
};

/**
 * 다음 안내 생성
 * @param {Object} currentLocation - {latitude, longitude}
 * @param {Array} segments - 세그먼트 배열
 * @param {number} currentSegmentIndex - 현재 세그먼트 인덱스
 * @returns {Object} 다음 안내 정보
 */
export const getNextInstruction = (currentLocation, segments, currentSegmentIndex) => {
  if (!segments || segments.length === 0 || currentSegmentIndex >= segments.length) {
    return null;
  }

  const currentSegment = segments[currentSegmentIndex];
  const nextSegment = segments[currentSegmentIndex + 1];

  // 현재 세그먼트 끝점까지의 거리
  const distanceToTurn = calculateDistance(
    currentLocation.latitude,
    currentLocation.longitude,
    currentSegment.endPoint.latitude,
    currentSegment.endPoint.longitude
  );

  let instructionText;
  let voiceInstruction;

  if (!nextSegment) {
    // 마지막 세그먼트 - 목적지 안내
    instructionText = '목적지';
    voiceInstruction = `${Math.round(distanceToTurn)}미터 후 목적지에 도착합니다`;
  } else {
    // 다음 세그먼트 안내
    if (distanceToTurn < 50) {
      instructionText = nextSegment.instruction;
      voiceInstruction = `곧 ${nextSegment.instruction}`;
    } else if (distanceToTurn < 200) {
      instructionText = nextSegment.instruction;
      voiceInstruction = `${Math.round(distanceToTurn)}미터 후 ${nextSegment.instruction}`;
    } else {
      instructionText = nextSegment.instruction;
      voiceInstruction = null; // 200m 이상은 음성 안내 안 함
    }
  }

  return {
    instruction: instructionText,
    direction: nextSegment?.direction || 'straight',
    distance: Math.round(distanceToTurn),
    voiceInstruction,
    segment: nextSegment || currentSegment,
  };
};

/**
 * 세그먼트의 위험 경고 생성
 * @param {Object} segment - 세그먼트
 * @param {Object} currentLocation - 현재 위치
 * @returns {Object|null} 경고 정보
 */
export const getHazardWarning = (segment, currentLocation) => {
  if (!segment || !segment.hazards || segment.hazards.length === 0) {
    return null;
  }

  // 가장 가까운 위험 찾기
  let nearestHazard = null;
  let minDistance = Infinity;

  for (const hazard of segment.hazards) {
    const distance = calculateDistance(
      currentLocation.latitude,
      currentLocation.longitude,
      hazard.latitude,
      hazard.longitude
    );

    if (distance < minDistance) {
      minDistance = distance;
      nearestHazard = { ...hazard, distance: Math.round(distance) };
    }
  }

  // 500m 이내의 위험만 경고
  if (nearestHazard && nearestHazard.distance < 500) {
    return {
      type: nearestHazard.hazard_type,
      distance: nearestHazard.distance,
      severity: nearestHazard.severity,
      message: `${nearestHazard.distance}m 앞 ${getHazardTypeName(nearestHazard.hazard_type)} 주의`,
    };
  }

  return null;
};

/**
 * 위험 유형 이름 가져오기
 * @param {string} type - 위험 유형
 * @returns {string} 한글 이름
 */
const getHazardTypeName = (type) => {
  const names = {
    armed_conflict: '무력충돌',
    conflict: '충돌',
    protest_riot: '시위/폭동',
    protest: '시위',
    checkpoint: '검문소',
    road_damage: '도로 손상',
    natural_disaster: '자연재해',
    flood: '홍수',
    landslide: '산사태',
    other: '위험',
  };

  return names[type] || '위험';
};
