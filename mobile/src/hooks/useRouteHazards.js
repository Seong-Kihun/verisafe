/**
 * 경로 위험 정보 관련 공통 hook
 * RouteCard, RouteResultSheet 등에서 중복 사용되는 로직을 통합
 */

import { useMemo } from 'react';

/**
 * 경로의 위험 개수를 계산하는 hook
 *
 * @param {Object} route - 경로 데이터
 * @param {Array} hazards - 위험 정보 배열 (선택적, 상세 정보가 로드된 경우)
 * @returns {number} 위험 개수
 */
export const useRouteHazardCount = (route, hazards = null) => {
  return useMemo(() => {
    // 상세 위험 정보가 있으면 우선 사용
    if (hazards && Array.isArray(hazards) && hazards.length > 0) {
      return hazards.length;
    }

    // 백엔드에서 계산된 hazard_count 사용 (기본값: 0)
    return route?.hazard_count || 0;
  }, [route?.hazard_count, hazards]);
};

/**
 * 위험 정보가 로드되었는지 확인하는 hook
 *
 * @param {Array} hazards - 위험 정보 배열
 * @returns {boolean} 위험 정보 존재 여부
 */
export const useHasRouteHazards = (hazards) => {
  return useMemo(() => {
    return hazards && Array.isArray(hazards) && hazards.length > 0;
  }, [hazards]);
};

/**
 * 경로의 위험도 요약 정보를 계산하는 hook
 *
 * @param {Object} route - 경로 데이터
 * @param {Array} hazards - 위험 정보 배열
 * @returns {Object} 요약 정보 { hazardCount, riskScore, hasHazards }
 */
export const useRouteRiskSummary = (route, hazards = null) => {
  const hazardCount = useRouteHazardCount(route, hazards);
  const hasHazards = useHasRouteHazards(hazards);

  return useMemo(() => ({
    hazardCount,
    riskScore: route?.risk_score || 0,
    hasHazards,
  }), [hazardCount, route?.risk_score, hasHazards]);
};
