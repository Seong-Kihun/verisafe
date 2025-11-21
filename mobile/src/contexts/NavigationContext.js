/**
 * NavigationContext - 네비게이션 전역 상태 관리
 */

import React, { createContext, useContext, useState, useCallback, useRef, useEffect } from 'react';
import { Alert } from 'react-native';
import * as Location from 'expo-location';
import navigationService from '../services/navigationService';
import voiceGuidance from '../services/voiceGuidance';
import { routeAPI } from '../services/api';
import { calculateDistance } from '../utils/navigationUtils';

const NavigationContext = createContext();

export const NavigationProvider = ({ children }) => {
  const [isNavigating, setIsNavigating] = useState(false);
  const [navigationState, setNavigationState] = useState(null);
  const [route, setRoute] = useState(null);
  const [isVoiceEnabled, setIsVoiceEnabled] = useState(true);
  const lastVoiceInstructionRef = useRef(null);
  const lastOffRouteAlertRef = useRef(null); // 경로 이탈 알림 중복 방지
  const isMountedRef = useRef(true); // 컴포넌트 마운트 상태 추적

  /**
   * 네비게이션 시작
   * @param {Object} routeData - 경로 데이터
   */
  const startNavigation = useCallback(async (routeData) => {
    try {
      console.log('[NavigationContext] Starting navigation...');

      // 경로 데이터 변환: polyline을 coordinates 형식으로 변환
      const processedRoute = {
        ...routeData,
        coordinates: routeData.polyline
          ? routeData.polyline.map(([lat, lng]) => ({ latitude: lat, longitude: lng }))
          : routeData.waypoints?.map(([lat, lng]) => ({ latitude: lat, longitude: lng })) || [],
        total_distance: routeData.distance_meters || routeData.distance * 1000,
        total_duration: routeData.duration_seconds || routeData.duration * 60,
      };

      // 경로 유효성 검증
      if (!processedRoute.coordinates || processedRoute.coordinates.length === 0) {
        throw new Error('유효하지 않은 경로 데이터입니다.');
      }

      // 현재 위치 가져오기
      const location = await Location.getCurrentPositionAsync({
        accuracy: Location.Accuracy.BestForNavigation,
      });
      const currentLocation = {
        latitude: location.coords.latitude,
        longitude: location.coords.longitude,
      };

      // 경로 시작점과 현재 위치의 거리 확인
      const routeStart = processedRoute.coordinates[0];
      const distanceToStart = calculateDistance(
        currentLocation.latitude,
        currentLocation.longitude,
        routeStart.latitude,
        routeStart.longitude
      );

      console.log(`[NavigationContext] 현재 위치: (${currentLocation.latitude}, ${currentLocation.longitude})`);
      console.log(`[NavigationContext] 경로 시작점: (${routeStart.latitude}, ${routeStart.longitude})`);
      console.log(`[NavigationContext] 거리: ${Math.round(distanceToStart)}m`);

      // 경로 시작점에서 5km 이상 떨어져 있으면 경고
      if (distanceToStart > 5000) {
        Alert.alert(
          '경고',
          `현재 위치가 경로 시작점에서 ${(distanceToStart / 1000).toFixed(1)}km 떨어져 있습니다.\n그래도 네비게이션을 시작하시겠습니까?`,
          [
            { text: '취소', style: 'cancel' },
            {
              text: '시작',
              onPress: async () => {
                await continueNavigation(processedRoute);
              }
            }
          ]
        );
        return;
      }

      await continueNavigation(processedRoute);
    } catch (error) {
      console.error('[NavigationContext] Failed to start navigation:', error);
      Alert.alert('오류', '네비게이션을 시작할 수 없습니다.\n위치 권한을 확인해주세요.');
    }
  }, [isVoiceEnabled]);

  /**
   * 네비게이션 계속 진행 (검증 후)
   */
  const continueNavigation = async (processedRoute) => {
    try {
      setRoute(processedRoute);

      // 음성 안내 시작 알림
      if (isVoiceEnabled && processedRoute.total_distance && processedRoute.total_duration) {
        await voiceGuidance.speakNavigationStart(
          processedRoute.total_distance,
          processedRoute.total_duration
        );
      }

      // 네비게이션 서비스 시작
      await navigationService.startNavigation(
        processedRoute,
        handleNavigationUpdate, // 상태 업데이트 콜백
        handleNavigationComplete, // 완료 콜백
        handleOffRoute // 경로 이탈 콜백
      );

      setIsNavigating(true);
    } catch (error) {
      console.error('[NavigationContext] Failed to continue navigation:', error);
      throw error;
    }
  };

  /**
   * 네비게이션 상태 업데이트 핸들러
   * @param {Object} state - 네비게이션 상태
   */
  const handleNavigationUpdate = useCallback((state) => {
    // 컴포넌트가 unmount되었으면 상태 업데이트 스킵
    if (!isMountedRef.current) return;

    setNavigationState(state);

    // 음성 안내 (중복 방지)
    if (isVoiceEnabled && state.nextInstruction) {
      const shouldSpeak = navigationService.shouldSpeak(state.nextInstruction);

      if (shouldSpeak && state.nextInstruction.voiceInstruction) {
        const currentInstruction = state.nextInstruction.voiceInstruction;

        // 이전 안내와 다를 때만 음성 출력
        if (currentInstruction !== lastVoiceInstructionRef.current) {
          voiceGuidance.speakDirection(
            state.nextInstruction.instruction,
            state.nextInstruction.distance
          );
          lastVoiceInstructionRef.current = currentInstruction;
        }
      }
    }

    // 위험 경고 음성 (200m 이내)
    if (isVoiceEnabled && state.hazardWarning && state.hazardWarning.distance < 200) {
      voiceGuidance.speakHazardWarning(
        state.hazardWarning.type,
        state.hazardWarning.distance
      );
    }
  }, [isVoiceEnabled]);

  /**
   * 네비게이션 완료 핸들러
   * @param {Object} result - 완료 정보
   */
  const handleNavigationComplete = useCallback((result) => {
    console.log('[NavigationContext] Navigation completed:', result);

    if (isVoiceEnabled) {
      voiceGuidance.speakArrival();
    }

    Alert.alert(
      '목적지 도착',
      `소요 시간: ${result.formattedElapsedTime}\n안전하게 도착하셨습니다!`,
      [{ text: '확인', onPress: () => stopNavigation() }]
    );
  }, [isVoiceEnabled]);

  /**
   * 경로 이탈 핸들러
   * @param {Object} offRouteInfo - 경로 이탈 정보
   */
  const handleOffRoute = useCallback(async (offRouteInfo) => {
    console.log('[NavigationContext] Off route:', offRouteInfo);

    // 경로 이탈 알림 중복 방지 (30초 이내 재알림 방지)
    const now = Date.now();
    if (lastOffRouteAlertRef.current && (now - lastOffRouteAlertRef.current) < 30000) {
      console.log('[NavigationContext] Off route alert suppressed (too soon)');
      return;
    }
    lastOffRouteAlertRef.current = now;

    if (isVoiceEnabled) {
      await voiceGuidance.speakOffRoute();
    }

    Alert.alert(
      '경로 이탈',
      `경로에서 ${offRouteInfo.distance}m 벗어났습니다.\n경로를 다시 찾으시겠습니까?`,
      [
        { text: '취소', style: 'cancel' },
        {
          text: '재탐색',
          onPress: async () => {
            await rerouteToDestination(offRouteInfo.currentLocation);
          }
        }
      ]
    );
  }, [isVoiceEnabled]);

  /**
   * 경로 재탐색
   * @param {Object} currentLocation - 현재 위치
   */
  const rerouteToDestination = useCallback(async (currentLocation) => {
    if (!route || !route.coordinates) return;

    try {
      if (isVoiceEnabled) {
        await voiceGuidance.speakRerouting();
      }

      const destination = route.coordinates[route.coordinates.length - 1];

      // 경로 재탐색
      const newRouteResponse = await routeAPI.calculateRoute({
        start: { lat: currentLocation.latitude, lng: currentLocation.longitude },
        end: { lat: destination.latitude, lng: destination.longitude },
        preference: 'safe',
        transportation_mode: route.transportation_mode || 'car',
        excluded_hazard_types: []
      });

      // 기존 네비게이션 중지
      await navigationService.stopNavigation();

      // 새 경로로 재시작 (첫 번째 경로 사용)
      if (newRouteResponse.routes && newRouteResponse.routes.length > 0) {
        await startNavigation(newRouteResponse.routes[0]);
      }
    } catch (error) {
      console.error('[NavigationContext] Reroute failed:', error);
      Alert.alert('오류', '경로 재탐색에 실패했습니다.');
    }
  }, [route, isVoiceEnabled, startNavigation]);

  /**
   * 네비게이션 중지
   */
  const stopNavigation = useCallback(async () => {
    console.log('[NavigationContext] Stopping navigation...');

    await navigationService.stopNavigation();
    await voiceGuidance.stop();

    setIsNavigating(false);
    setNavigationState(null);
    setRoute(null);
    lastVoiceInstructionRef.current = null;
    lastOffRouteAlertRef.current = null;
  }, []);

  /**
   * 음성 안내 토글
   */
  const toggleVoiceGuidance = useCallback(() => {
    const newEnabled = !isVoiceEnabled;
    setIsVoiceEnabled(newEnabled);
    voiceGuidance.setEnabled(newEnabled);
  }, [isVoiceEnabled]);

  // 컴포넌트 언마운트 시 cleanup
  useEffect(() => {
    isMountedRef.current = true;

    return () => {
      console.log('[NavigationContext] Cleanup on unmount');
      isMountedRef.current = false;

      // 네비게이션이 실행 중이면 중지
      if (isNavigating) {
        navigationService.stopNavigation().catch(err => {
          console.error('[NavigationContext] Cleanup failed:', err);
        });
        voiceGuidance.stop().catch(err => {
          console.error('[NavigationContext] Voice cleanup failed:', err);
        });
      }
    };
  }, [isNavigating]);

  const value = {
    // 상태
    isNavigating,
    navigationState,
    route,
    isVoiceEnabled,

    // 메서드
    startNavigation,
    stopNavigation,
    rerouteToDestination,
    toggleVoiceGuidance,
  };

  return (
    <NavigationContext.Provider value={value}>
      {children}
    </NavigationContext.Provider>
  );
};

/**
 * NavigationContext 사용 훅
 */
export const useNavigation = () => {
  const context = useContext(NavigationContext);

  if (!context) {
    throw new Error('useNavigation must be used within NavigationProvider');
  }

  return context;
};

export default NavigationContext;
