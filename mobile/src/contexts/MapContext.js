/**
 * MapContext - 지도탭 전역 상태 관리
 * 
 * 상태:
 * - isSearchOpen: SearchScreen 모달 열림 여부
 * - isPlaceSheetOpen: 장소 상세 카드 열림 여부
 * - isRouteSheetOpen: 경로 결과 카드 열림 여부
 * - selectedPlace: 선택된 장소 상세 정보
 * - routeParams: 경로 계산 파라미터
 * - routeResponse: 경로 계산 결과
 */

import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import { userCountryStorage } from '../services/storage';
import { DEFAULT_COUNTRY } from '../constants/countries';

const MapContext = createContext();

export const useMapContext = () => {
  const context = useContext(MapContext);
  if (!context) {
    throw new Error('useMapContext must be used within MapProvider');
  }
  return context;
};

export function MapProvider({ children }) {
  // UI 상태
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const [isPlaceSheetOpen, setIsPlaceSheetOpen] = useState(false);
  const [isRouteSheetOpen, setIsRouteSheetOpen] = useState(false);

  // 데이터
  const [selectedPlace, setSelectedPlace] = useState(null);
  const [routeParams, setRouteParams] = useState(null);
  const [routeResponseState, setRouteResponseState] = useState(null);
  const [userLocation, setUserLocation] = useState(null);
  const [userCountry, setUserCountry] = useState(DEFAULT_COUNTRY);

  // 초기화: 저장된 국가 설정 불러오기
  useEffect(() => {
    loadUserCountry();
  }, []);

  const loadUserCountry = async () => {
    try {
      const country = await userCountryStorage.get();
      if (country) {
        setUserCountry(country);
      }
    } catch (error) {
      console.error('Failed to load user country:', error);
    }
  };

  // SearchScreen 제어
  const openSearch = useCallback(() => {
    setIsSearchOpen(true);
  }, []);

  const closeSearch = useCallback(() => {
    setIsSearchOpen(false);
  }, []);

  // PlaceDetailSheet 제어
  const openPlaceSheet = useCallback((place) => {
    // 유효성 검사
    if (!place || typeof place !== 'object') {
      console.error('[MapContext] Invalid place object:', place);
      return;
    }

    if (__DEV__) {
      console.log('[MapContext] openPlaceSheet:', place.name || place.id);
    }

    setSelectedPlace(place);
    setIsPlaceSheetOpen(true);
    setIsRouteSheetOpen(false); // 경로 시트 닫기
  }, []);

  const closePlaceSheet = useCallback(() => {
    setIsPlaceSheetOpen(false);
  }, []);

  // RouteResultSheet 제어
  const openRouteSheet = useCallback((params, response) => {
    // 유효성 검사
    if (!params || !response) {
      console.error('[MapContext] Invalid route data:', { params, response });
      return;
    }

    if (__DEV__) {
      console.log('[MapContext] openRouteSheet');
    }

    setRouteParams(params);
    setRouteResponseState(response);
    setIsRouteSheetOpen(true);
    setIsPlaceSheetOpen(false); // 장소 시트 닫기
  }, []);

  const closeRouteSheet = useCallback(() => {
    setIsRouteSheetOpen(false);
  }, []);

  // 경로 응답 직접 설정 (RoutePlanningScreen에서 사용)
  const setRouteResponse = useCallback((response) => {
    setRouteResponseState(response);
  }, []);

  // 선택된 장소 초기화
  const clearSelectedPlace = useCallback(() => {
    setSelectedPlace(null);
  }, []);

  // 사용자 위치 설정
  const updateUserLocation = useCallback((location) => {
    setUserLocation(location);
  }, []);

  // 사용자 국가 설정
  const updateUserCountry = useCallback(async (country) => {
    try {
      setUserCountry(country);
      await userCountryStorage.save(country);
      return true;
    } catch (error) {
      console.error('Failed to update user country:', error);
      return false;
    }
  }, []);

  const value = {
    // UI 상태
    isSearchOpen,
    isPlaceSheetOpen,
    isRouteSheetOpen,

    // 데이터
    selectedPlace,
    routeParams,
    routeResponse: routeResponseState,
    userLocation,
    userCountry,

    // SearchScreen 액션
    openSearch,
    closeSearch,

    // PlaceDetailSheet 액션
    openPlaceSheet,
    closePlaceSheet,

    // RouteResultSheet 액션
    openRouteSheet,
    closeRouteSheet,
    setRouteResponse,

    // 기타
    clearSelectedPlace,
    updateUserLocation,
    updateUserCountry,
  };

  return (
    <MapContext.Provider value={value}>
      {children}
    </MapContext.Provider>
  );
}

export default MapContext;

