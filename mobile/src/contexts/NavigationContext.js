/**
 * NavigationContext - 네비게이션 전역 상태 관리
 */

import React, { createContext, useContext, useState, useCallback, useRef } from 'react';
import { Alert } from 'react-native';
import navigationService from '../services/navigationService';
import voiceGuidance from '../services/voiceGuidance';
import { routeAPI } from '../services/api';

const NavigationContext = createContext();

export const NavigationProvider = ({ children }) => {
  const [isNavigating, setIsNavigating] = useState(false);
  const [navigationState, setNavigationState] = useState(null);
  const [route, setRoute] = useState(null);
  const [isVoiceEnabled, setIsVoiceEnabled] = useState(true);
  const lastVoiceInstructionRef = useRef(null);

  /**
   * 네비게이션 시작
   * @param {Object} routeData - 경로 데이터
   */
  const startNavigation = useCallback(async (routeData) => {
    try {
      console.log('[NavigationContext] Starting navigation...');

      setRoute(routeData);

      // 음성 안내 시작 알림
      if (isVoiceEnabled && routeData.total_distance && routeData.total_duration) {
        await voiceGuidance.speakNavigationStart(
          routeData.total_distance,
          routeData.total_duration
        );
      }

      // 네비게이션 서비스 시작
      await navigationService.startNavigation(
        routeData,
        handleNavigationUpdate, // 상태 업데이트 콜백
        handleNavigationComplete, // 완료 콜백
        handleOffRoute // 경로 이탈 콜백
      );

      setIsNavigating(true);
    } catch (error) {
      console.error('[NavigationContext] Failed to start navigation:', error);
      Alert.alert('오류', '네비게이션을 시작할 수 없습니다.\n위치 권한을 확인해주세요.');
    }
  }, [isVoiceEnabled]);

  /**
   * 네비게이션 상태 업데이트 핸들러
   * @param {Object} state - 네비게이션 상태
   */
  const handleNavigationUpdate = useCallback((state) => {
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
      const newRoute = await routeAPI.searchRoute({
        start: currentLocation,
        end: destination,
        hazardTypes: [] // 기존 설정 사용 가능
      });

      // 기존 네비게이션 중지
      await navigationService.stopNavigation();

      // 새 경로로 재시작
      await startNavigation(newRoute);
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
  }, []);

  /**
   * 음성 안내 토글
   */
  const toggleVoiceGuidance = useCallback(() => {
    const newEnabled = !isVoiceEnabled;
    setIsVoiceEnabled(newEnabled);
    voiceGuidance.setEnabled(newEnabled);
  }, [isVoiceEnabled]);

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
