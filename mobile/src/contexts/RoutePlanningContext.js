/**
 * RoutePlanningContext - 경로 찾기 전역 상태 관리
 * 
 * 상태:
 * - startLocation: 출발지 { lat, lng, name, address }
 * - endLocation: 목적지 { lat, lng, name, address }
 * - transportationMode: 이동 수단 'car' | 'walking' | 'bicycle'
 * - routes: 경로 목록
 * - selectedRoute: 선택된 경로
 * - isHazardBriefingOpen: 위험 정보 브리핑 모달 열림 여부
 */

import React, { createContext, useContext, useState, useCallback } from 'react';

const RoutePlanningContext = createContext();

export const useRoutePlanningContext = () => {
  const context = useContext(RoutePlanningContext);
  if (!context) {
    throw new Error('useRoutePlanningContext must be used within RoutePlanningProvider');
  }
  return context;
};

export function RoutePlanningProvider({ children }) {
  // 위치 정보
  const [startLocation, setStartLocation] = useState(null);
  const [endLocation, setEndLocation] = useState(null);
  
  // 이동 수단
  const [transportationMode, setTransportationMode] = useState('car');
  
  // 경로 정보
  const [routes, setRoutes] = useState([]);
  const [selectedRoute, setSelectedRoute] = useState(null);
  
  // UI 상태
  const [isHazardBriefingOpen, setIsHazardBriefingOpen] = useState(false);
  const [isCalculating, setIsCalculating] = useState(false);

  // 출발지 설정
  const setStart = useCallback((location) => {
    setStartLocation(location);
    setSelectedRoute(null); // 경로 초기화
  }, []);

  // 목적지 설정
  const setEnd = useCallback((location) => {
    setEndLocation(location);
    setSelectedRoute(null); // 경로 초기화
  }, []);

  // 이동 수단 변경
  const setTransportation = useCallback((mode) => {
    setTransportationMode(mode);
    setSelectedRoute(null); // 경로 초기화
  }, []);

  // 경로 목록 설정
  const setRoutesList = useCallback((routesList) => {
    setRoutes(routesList);
  }, []);

  // 경로 선택
  const selectRoute = useCallback((route) => {
    setSelectedRoute(route);
  }, []);

  // 위험 정보 브리핑 열기/닫기
  const openHazardBriefing = useCallback(() => {
    setIsHazardBriefingOpen(true);
  }, []);

  const closeHazardBriefing = useCallback(() => {
    setIsHazardBriefingOpen(false);
  }, []);

  // 계산 상태 설정
  const setCalculating = useCallback((calculating) => {
    setIsCalculating(calculating);
  }, []);

  // 초기화
  const reset = useCallback(() => {
    setStartLocation(null);
    setEndLocation(null);
    setTransportationMode('car');
    setRoutes([]);
    setSelectedRoute(null);
    setIsHazardBriefingOpen(false);
    setIsCalculating(false);
  }, []);

  const value = {
    // 상태
    startLocation,
    endLocation,
    transportationMode,
    routes,
    selectedRoute,
    isHazardBriefingOpen,
    isCalculating,
    
    // 액션
    setStart,
    setEnd,
    setTransportation,
    setRoutesList,
    selectRoute,
    openHazardBriefing,
    closeHazardBriefing,
    setCalculating,
    reset,
  };

  return (
    <RoutePlanningContext.Provider value={value}>
      {children}
    </RoutePlanningContext.Provider>
  );
}

export default RoutePlanningContext;

