/**
 * networkUtils.js - 네트워크 상태 체크 유틸리티
 */

import NetInfo from '@react-native-community/netinfo';

/**
 * 현재 네트워크 연결 상태 확인
 */
export const isOnline = async () => {
  try {
    const state = await NetInfo.fetch();
    return state.isConnected && state.isInternetReachable;
  } catch (error) {
    console.error('Failed to check network status:', error);
    return false;
  }
};

/**
 * 네트워크 상태 변경 리스너 등록
 */
export const subscribeToNetworkStatus = (callback) => {
  return NetInfo.addEventListener(state => {
    const online = state.isConnected && state.isInternetReachable;
    callback(online);
  });
};

/**
 * 네트워크 타입 가져오기
 */
export const getNetworkType = async () => {
  try {
    const state = await NetInfo.fetch();
    return state.type; // wifi, cellular, none, etc.
  } catch (error) {
    console.error('Failed to get network type:', error);
    return 'unknown';
  }
};
