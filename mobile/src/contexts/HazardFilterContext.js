/**
 * HazardFilterContext - 위험 유형 필터 전역 상태 관리
 *
 * 책임:
 * - 사용자가 선택한 위험 유형 필터 상태 관리
 * - 경로 계산 시 제외할 위험 유형 제공
 * - LayerToggleMenu와 RoutePlanningScreen 간 상태 공유
 */

import React, { createContext, useContext, useState, useCallback } from 'react';
import { ALLOWED_HAZARD_TYPE_IDS } from '../constants/hazardTypes';

const HazardFilterContext = createContext();

export const useHazardFilter = () => {
  const context = useContext(HazardFilterContext);
  if (!context) {
    throw new Error('useHazardFilter must be used within HazardFilterProvider');
  }
  return context;
};

/**
 * 위험 유형 ID 유효성 검증
 * @param {string} typeId - 검증할 위험 유형 ID
 * @returns {boolean} - 유효한 위험 유형인지 여부
 */
const isValidHazardType = (typeId) => {
  return ALLOWED_HAZARD_TYPE_IDS.includes(typeId);
};

export function HazardFilterProvider({ children }) {
  // 제외할 위험 유형 리스트
  // 기본적으로 시위(protest), 검문소(checkpoint), 도로손상(road_damage)만 표시
  const [excludedHazardTypes, setExcludedHazardTypes] = useState([
    'armed_conflict',
    'natural_disaster',
    'flood',
    'landslide',
    'safe_haven',
    'other'
  ]);

  // 위험 유형 토글 (제외 목록에 추가/제거)
  const toggleHazardType = useCallback((typeId) => {
    // 유효성 검증
    if (!isValidHazardType(typeId)) {
      console.warn(`[HazardFilterContext] Invalid hazard type: ${typeId}`);
      return;
    }

    setExcludedHazardTypes(prev => {
      if (prev.includes(typeId)) {
        // 이미 제외된 유형이면 제거 (다시 포함)
        return prev.filter(t => t !== typeId);
      } else {
        // 제외 목록에 추가
        return [...prev, typeId];
      }
    });
  }, []);

  // 특정 위험 유형이 제외되었는지 확인
  const isHazardTypeExcluded = useCallback((typeId) => {
    return excludedHazardTypes.includes(typeId);
  }, [excludedHazardTypes]);

  // 모든 필터 초기화
  const resetFilters = useCallback(() => {
    setExcludedHazardTypes([]);
  }, []);

  // 특정 위험 유형들을 한 번에 제외
  const setExcludedTypes = useCallback((types) => {
    // 유효성 검증: 유효한 타입만 필터링
    const validTypes = types.filter(typeId => {
      const isValid = isValidHazardType(typeId);
      if (!isValid) {
        console.warn(`[HazardFilterContext] Invalid hazard type ignored: ${typeId}`);
      }
      return isValid;
    });
    setExcludedHazardTypes(validTypes);
  }, []);

  const value = {
    excludedHazardTypes,
    toggleHazardType,
    isHazardTypeExcluded,
    resetFilters,
    setExcludedTypes,
  };

  return (
    <HazardFilterContext.Provider value={value}>
      {children}
    </HazardFilterContext.Provider>
  );
}

export default HazardFilterContext;
