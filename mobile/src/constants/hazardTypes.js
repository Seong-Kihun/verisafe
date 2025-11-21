/**
 * 위험 유형 상수 정의
 *
 * 주의: 이 파일의 hazard type ID는 백엔드의 ALLOWED_HAZARD_TYPES와 동기화되어야 합니다.
 * Backend: backend/app/routes/route.py의 ALLOWED_HAZARD_TYPES
 */

export const HAZARD_TYPES = [
  // 무력 충돌
  { id: 'armed_conflict', name: '무력충돌', icon: 'conflict', color: '#DC2626' },

  // 시위/폭동
  { id: 'protest_riot', name: '시위/폭동', icon: 'protest', color: '#F59E0B' },

  // 검문소
  { id: 'checkpoint', name: '검문소', icon: 'checkpoint', color: '#FF6B6B' },

  // 도로 손상
  { id: 'road_damage', name: '도로 손상', icon: 'roadDamage', color: '#F97316' },

  // 자연재해 관련
  { id: 'natural_disaster', name: '자연재해', icon: 'naturalDisaster', color: '#DC2626' },
  { id: 'flood', name: '홍수', icon: 'naturalDisaster', color: '#3B82F6' },
  { id: 'landslide', name: '산사태', icon: 'naturalDisaster', color: '#92400E' },

  // 안전 대피처
  { id: 'safe_haven', name: '대피처', icon: 'safeHaven', color: '#10B981' },

  // 기타
  { id: 'other', name: '기타', icon: 'other', color: '#6B7280' },
];

/**
 * 백엔드 허용 위험 유형 목록 (참고용)
 * Backend ALLOWED_HAZARD_TYPES와 동기화 필요
 */
export const ALLOWED_HAZARD_TYPE_IDS = HAZARD_TYPES.map(t => t.id);

/**
 * 시간대 필터 옵션
 */
export const TIME_FILTERS = [
  { id: 'all', name: '전체', hours: null },
  { id: '24h', name: '24시간', hours: 24 },
  { id: '48h', name: '48시간', hours: 48 },
  { id: '7d', name: '7일', hours: 168 },
];

/**
 * 위험 유형별 아이콘 매핑
 */
export const getHazardIcon = (hazardType) => {
  // 백엔드 위험 유형을 프론트엔드 유형으로 매핑
  const typeMapping = {
    'conflict': 'armed_conflict',
    'protest': 'protest_riot',
  };

  const mappedType = typeMapping[hazardType] || hazardType;
  const hazard = HAZARD_TYPES.find(h => h.id === mappedType);
  return hazard?.icon || 'other';
};

/**
 * 위험 유형별 색상 매핑
 */
export const getHazardColor = (hazardType) => {
  // 백엔드 위험 유형을 프론트엔드 유형으로 매핑
  const typeMapping = {
    'conflict': 'armed_conflict',
    'protest': 'protest_riot',
  };

  const mappedType = typeMapping[hazardType] || hazardType;
  const hazard = HAZARD_TYPES.find(h => h.id === mappedType);
  return hazard?.color || '#6B7280';
};

/**
 * 위험 유형별 이름 매핑
 */
export const getHazardName = (hazardType) => {
  // 백엔드 위험 유형을 프론트엔드 유형으로 매핑
  const typeMapping = {
    'conflict': 'armed_conflict',
    'protest': 'protest_riot',
  };

  const mappedType = typeMapping[hazardType] || hazardType;
  const hazard = HAZARD_TYPES.find(h => h.id === mappedType);
  return hazard?.name || '알 수 없음';
};
