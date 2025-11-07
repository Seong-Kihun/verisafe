/**
 * VeriSafe 색상 시스템
 * KOICA 블루를 기본으로 한 색상 팔레트
 */

export const Colors = {
  // Primary (KOICA 블루) - 더 진한 블루로 신뢰감 강화
  primary: '#0047AB',        // 더 진한 블루 (기존 #0066CC → #0047AB)
  primaryLight: '#0066CC',   // 기존 primary
  primaryDark: '#003380',    // 더 진한 버전
  
  // Secondary (강조)
  accent: '#F4D160',
  accentLight: '#F8E58A',
  accentDark: '#F0C930',
  
  // Background & Surface - 계층 명확화
  background: '#FFFFFF',      // 순수 흰색 (기존 #F8F9FA → #FFFFFF, 대비 향상)
  surface: '#F8F9FA',        // 기존 background (섹션 배경)
  surfaceElevated: '#FFFFFF', // 카드/시트 (기존 #FAFBFC → #FFFFFF)
  
  // Text - 더 명확한 대비
  textPrimary: '#0F172A',    // 더 진한 텍스트 (기존 #1E293B → #0F172A)
  textSecondary: '#64748B',  // 유지
  textTertiary: '#94A3B8',   // 유지
  textInverse: '#FFFFFF',
  
  // Border
  border: '#E2E8F0',
  borderLight: '#F1F5F9',
  borderDark: '#CBD5E1',
  
  // Status
  success: '#10B981',
  warning: '#F59E0B',
  danger: '#EF4444',
  error: '#EF4444',  // Alias for danger
  info: '#3B82F6',
  
  // Risk Levels - 더 명확한 구분 (0-10 스케일 기준)
  riskVeryLow: '#10B981',   // 0-2 (녹색)
  riskLow: '#84CC16',        // 3-4 (연두색, 기존 #F4D160 → #84CC16)
  riskMedium: '#F59E0B',    // 5-7 (주황색)
  riskHigh: '#EF4444',      // 8-10 (빨간색)
  
  // Hazard Types
  hazardArmedConflict: '#DC2626',  // 무력충돌
  hazardProtestRiot: '#F59E0B',    // 시위/폭동
  hazardCheckpoint: '#FF6B6B',     // 불법 검문소
  hazardRoadDamage: '#F97316',     // 도로 파손
  hazardNaturalDisaster: '#DC2626', // 자연재해
  hazardOther: '#6B7280',          // 기타
  
  // Map
  mapRouteSafe: '#10B981',
  mapRouteFast: '#3B82F6',
  mapRouteDanger: '#EF4444',
  
  // Overlay
  overlay: 'rgba(0, 0, 0, 0.5)',
  overlayLight: 'rgba(0, 0, 0, 0.2)',
  
  // Shadow - 계층별 그림자
  shadow: 'rgba(0, 0, 0, 0.1)',         // 기존 (하위 호환)
  shadowDark: 'rgba(0, 0, 0, 0.2)',     // 기존 (하위 호환)
  shadowSmall: 'rgba(0, 0, 0, 0.08)',   // 작은 카드
  shadowMedium: 'rgba(0, 0, 0, 0.12)',  // 중간 카드/시트
  shadowLarge: 'rgba(0, 0, 0, 0.16)',   // 큰 카드/모달
};

/**
 * 위험도에 따른 색상 반환 (0-10 스케일 기준)
 * 0-2: 매우 낮음 (녹색)
 * 3-4: 낮음 (연두색)
 * 5-7: 보통 (주황색)
 * 8-10: 높음 (빨간색)
 */
export const getRiskColor = (riskScore) => {
  if (riskScore <= 2) return Colors.riskVeryLow;
  if (riskScore <= 4) return Colors.riskLow;
  if (riskScore <= 7) return Colors.riskMedium;
  return Colors.riskHigh;
};

/**
 * 위험 유형에 따른 색상 반환
 */
export const getHazardColor = (hazardType) => {
  const colorMap = {
    'armed_conflict': Colors.hazardArmedConflict,
    'protest_riot': Colors.hazardProtestRiot,
    'checkpoint': Colors.hazardCheckpoint,
    'road_damage': Colors.hazardRoadDamage,
    'natural_disaster': Colors.hazardNaturalDisaster,
    'other': Colors.hazardOther,
  };

  return colorMap[hazardType] || Colors.hazardOther;
};

/**
 * 안전도 등급 계산 (A~F) - Phase 2 추가
 * @param {number} riskScore - 위험도 점수 (0-10)
 * @returns {string} A, B, C, D, E, F
 */
export const getSafetyGrade = (riskScore) => {
  if (riskScore <= 2) return 'A';
  if (riskScore <= 4) return 'B';
  if (riskScore <= 6) return 'C';
  if (riskScore <= 8) return 'D';
  if (riskScore <= 9) return 'E';
  return 'F';
};

/**
 * 안전도 등급에 따른 색상 반환 - Phase 2 추가
 * @param {string} grade - 안전도 등급 (A~F)
 * @returns {string} 색상 코드
 */
export const getGradeColor = (grade) => {
  if (grade === 'A' || grade === 'B') return Colors.success;
  if (grade === 'C') return Colors.warning;
  return Colors.error;
};

/**
 * 경로 타입에 따른 색상 반환 - Phase 2 추가
 * @param {string} routeType - 경로 타입 ('safe', 'fast', 'alternative')
 * @returns {string} 색상 코드
 */
export const getRouteColor = (routeType) => {
  const colorMap = {
    'safe': Colors.mapRouteSafe,      // 초록색 - 안전 경로
    'fast': Colors.mapRouteFast,      // 파란색 - 빠른 경로
    'alternative': Colors.warning,     // 주황색 - 대안 경로
  };
  return colorMap[routeType] || Colors.mapRouteFast;
};

export default Colors;

