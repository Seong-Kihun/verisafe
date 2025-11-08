/**
 * 네비게이션 서비스
 * 실시간 위치 추적, 경로 매칭, 진행 상황 업데이트
 */

import * as Location from 'expo-location';
import {
  calculateDistance,
  calculateRemainingDistance,
  isDestinationReached,
  isOffRoute,
  findNearestPointOnPath,
  formatDistance,
  formatDuration,
} from '../utils/navigationUtils';
import {
  createNavigationSegments,
  findCurrentSegmentIndex,
  getNextInstruction,
  getHazardWarning,
} from './routeSegments';

class NavigationService {
  constructor() {
    this.isNavigating = false;
    this.route = null;
    this.segments = [];
    this.currentLocation = null;
    this.currentSegmentIndex = 0;
    this.locationSubscription = null;
    this.updateCallback = null;
    this.completionCallback = null;
    this.offRouteCallback = null;
    this.lastVoiceDistance = null;
    this.startTime = null;
  }

  /**
   * 네비게이션 시작
   * @param {Object} route - 경로 데이터
   * @param {Function} updateCallback - 상태 업데이트 콜백
   * @param {Function} completionCallback - 완료 콜백
   * @param {Function} offRouteCallback - 경로 이탈 콜백
   */
  async startNavigation(route, updateCallback, completionCallback, offRouteCallback) {
    try {
      console.log('[Navigation] Starting navigation...');

      // 위치 권한 확인
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        throw new Error('위치 권한이 필요합니다');
      }

      this.route = route;
      this.updateCallback = updateCallback;
      this.completionCallback = completionCallback;
      this.offRouteCallback = offRouteCallback;
      this.startTime = Date.now();

      // 경로를 세그먼트로 변환
      this.segments = createNavigationSegments(route);
      console.log('[Navigation] Created', this.segments.length, 'segments');

      // 현재 위치 가져오기
      const location = await Location.getCurrentPositionAsync({
        accuracy: Location.Accuracy.BestForNavigation,
      });

      this.currentLocation = {
        latitude: location.coords.latitude,
        longitude: location.coords.longitude,
      };

      // 위치 업데이트 구독
      this.locationSubscription = await Location.watchPositionAsync(
        {
          accuracy: Location.Accuracy.BestForNavigation,
          timeInterval: 2000, // 2초마다
          distanceInterval: 5, // 5m 이동 시
        },
        this.handleLocationUpdate.bind(this)
      );

      this.isNavigating = true;

      // 초기 상태 업데이트
      this.updateNavigationState();

      console.log('[Navigation] Navigation started successfully');
    } catch (error) {
      console.error('[Navigation] Failed to start navigation:', error);
      throw error;
    }
  }

  /**
   * 위치 업데이트 핸들러
   * @param {Object} location - expo-location의 위치 객체
   */
  handleLocationUpdate(location) {
    if (!this.isNavigating) return;

    this.currentLocation = {
      latitude: location.coords.latitude,
      longitude: location.coords.longitude,
      speed: location.coords.speed, // m/s
      heading: location.coords.heading, // 도
    };

    // 네비게이션 상태 업데이트
    this.updateNavigationState();
  }

  /**
   * 네비게이션 상태 업데이트 및 콜백 호출
   */
  updateNavigationState() {
    if (!this.currentLocation || !this.route || !this.segments.length) {
      return;
    }

    const path = this.route.coordinates;
    const destination = path[path.length - 1];

    // 목적지 도착 확인
    if (isDestinationReached(this.currentLocation, destination, 20)) {
      this.handleDestinationReached();
      return;
    }

    // 경로 이탈 확인
    const offRoute = isOffRoute(this.currentLocation, path, 50);
    if (offRoute) {
      this.handleOffRoute();
      return;
    }

    // 현재 세그먼트 찾기
    this.currentSegmentIndex = findCurrentSegmentIndex(
      this.currentLocation,
      this.segments,
      this.currentSegmentIndex
    );

    // 다음 안내 생성
    const nextInstruction = getNextInstruction(
      this.currentLocation,
      this.segments,
      this.currentSegmentIndex
    );

    // 위험 경고 생성
    const hazardWarning = getHazardWarning(
      this.segments[this.currentSegmentIndex],
      this.currentLocation
    );

    // 남은 거리 계산
    const remainingDistance = calculateRemainingDistance(
      this.currentLocation,
      path,
      this.currentSegmentIndex
    );

    // 남은 시간 추정 (속도 기반)
    const averageSpeed = this.currentLocation.speed || 5; // m/s, 기본 5m/s (18km/h)
    const remainingTime = remainingDistance / averageSpeed;

    // 진행률 계산
    const totalDistance = this.route.total_distance || this.calculateTotalDistance(path);
    const traveledDistance = totalDistance - remainingDistance;
    const progress = Math.min(100, Math.max(0, (traveledDistance / totalDistance) * 100));

    // 상태 객체 생성
    const navigationState = {
      currentLocation: this.currentLocation,
      currentSegmentIndex: this.currentSegmentIndex,
      totalSegments: this.segments.length,
      nextInstruction,
      hazardWarning,
      remainingDistance: Math.round(remainingDistance),
      remainingTime: Math.round(remainingTime),
      progress: Math.round(progress),
      formattedDistance: formatDistance(remainingDistance),
      formattedTime: formatDuration(remainingTime),
      speed: this.currentLocation.speed,
      heading: this.currentLocation.heading,
    };

    // 콜백 호출
    if (this.updateCallback) {
      this.updateCallback(navigationState);
    }
  }

  /**
   * 경로의 총 거리 계산
   * @param {Array} path - 좌표 배열
   * @returns {number} 총 거리 (미터)
   */
  calculateTotalDistance(path) {
    let total = 0;
    for (let i = 0; i < path.length - 1; i++) {
      total += calculateDistance(
        path[i].latitude,
        path[i].longitude,
        path[i + 1].latitude,
        path[i + 1].longitude
      );
    }
    return total;
  }

  /**
   * 목적지 도착 처리
   */
  handleDestinationReached() {
    console.log('[Navigation] Destination reached!');

    this.isNavigating = false;

    if (this.completionCallback) {
      const elapsedTime = Math.round((Date.now() - this.startTime) / 1000);
      this.completionCallback({
        success: true,
        elapsedTime,
        formattedElapsedTime: formatDuration(elapsedTime),
      });
    }

    this.stopNavigation();
  }

  /**
   * 경로 이탈 처리
   */
  handleOffRoute() {
    console.log('[Navigation] Off route detected');

    if (this.offRouteCallback) {
      const { nearestPoint, distance } = findNearestPointOnPath(
        this.currentLocation,
        this.route.coordinates
      );

      this.offRouteCallback({
        currentLocation: this.currentLocation,
        nearestPoint,
        distance: Math.round(distance),
      });
    }
  }

  /**
   * 네비게이션 중지
   */
  async stopNavigation() {
    console.log('[Navigation] Stopping navigation...');

    this.isNavigating = false;

    // 위치 업데이트 구독 해제
    if (this.locationSubscription) {
      this.locationSubscription.remove();
      this.locationSubscription = null;
    }

    // 상태 초기화
    this.route = null;
    this.segments = [];
    this.currentLocation = null;
    this.currentSegmentIndex = 0;
    this.updateCallback = null;
    this.completionCallback = null;
    this.offRouteCallback = null;
    this.lastVoiceDistance = null;
    this.startTime = null;

    console.log('[Navigation] Navigation stopped');
  }

  /**
   * 현재 네비게이션 상태 가져오기
   * @returns {Object} 현재 상태
   */
  getCurrentState() {
    return {
      isNavigating: this.isNavigating,
      currentLocation: this.currentLocation,
      currentSegmentIndex: this.currentSegmentIndex,
      totalSegments: this.segments.length,
      route: this.route,
    };
  }

  /**
   * 음성 안내가 필요한지 확인
   * @param {Object} nextInstruction - 다음 안내 정보
   * @returns {boolean}
   */
  shouldSpeak(nextInstruction) {
    if (!nextInstruction || !nextInstruction.voiceInstruction) {
      return false;
    }

    const distance = nextInstruction.distance;

    // 중복 음성 방지: 같은 거리 범위에서는 한 번만
    if (this.lastVoiceDistance !== null) {
      const diff = Math.abs(distance - this.lastVoiceDistance);
      if (diff < 20) {
        return false; // 20m 이내 변화는 무시
      }
    }

    // 음성 안내 시점: 500m, 200m, 100m, 50m
    const voiceDistances = [500, 200, 100, 50];
    const shouldSpeak = voiceDistances.some(d =>
      distance <= d && distance > d - 20
    );

    if (shouldSpeak) {
      this.lastVoiceDistance = distance;
    }

    return shouldSpeak;
  }
}

// 싱글톤 인스턴스 export
export default new NavigationService();
