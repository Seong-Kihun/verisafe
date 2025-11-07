/**
 * Geofencing service for danger zone monitoring
 * 위험 지역 모니터링을 위한 Geofencing 서비스
 */

import * as Location from 'expo-location';
import { Alert, Vibration } from 'react-native';

const DANGER_THRESHOLD = 70; // Risk score threshold
const CHECK_INTERVAL = 30000; // 30 seconds

class GeofencingService {
  constructor() {
    this.isMonitoring = false;
    this.currentLocation = null;
    this.dangerZones = [];
    this.lastAlertTime = {};
    this.checkInterval = null;
  }

  /**
   * Start monitoring for danger zones
   */
  async startMonitoring(getRiskAssessment) {
    try {
      // Request location permission
      const { status } = await Location.requestForegroundPermissionsAsync();

      if (status !== 'granted') {
        Alert.alert(
          '권한 필요',
          '위험 지역 경고를 받으려면 위치 권한이 필요합니다.'
        );
        return false;
      }

      this.isMonitoring = true;

      // Start periodic checks
      this.checkInterval = setInterval(async () => {
        try {
          const location = await Location.getCurrentPositionAsync({
            accuracy: Location.Accuracy.Balanced,
          });

          await this.checkDangerZones(
            location.coords.latitude,
            location.coords.longitude,
            getRiskAssessment
          );
        } catch (error) {
          console.error('[Geofencing] Geofence check error:', error);
        }
      }, CHECK_INTERVAL);

      console.log('[Geofencing] Danger zone monitoring started');
      return true;

    } catch (error) {
      console.error('[Geofencing] Failed to start monitoring:', error);
      return false;
    }
  }

  /**
   * Stop monitoring
   */
  async stopMonitoring() {
    try {
      if (this.checkInterval) {
        clearInterval(this.checkInterval);
        this.checkInterval = null;
      }
      this.isMonitoring = false;
      console.log('[Geofencing] Danger zone monitoring stopped');
      return true;
    } catch (error) {
      console.error('[Geofencing] Failed to stop monitoring:', error);
      return false;
    }
  }

  /**
   * Check if current location is in a danger zone
   */
  async checkDangerZones(latitude, longitude, getRiskAssessment) {
    try {
      // Get risk assessment for current location
      const assessment = await getRiskAssessment(latitude, longitude);

      if (assessment && assessment.riskScore >= DANGER_THRESHOLD) {
        // Check if we recently alerted for this zone
        const zoneKey = `${Math.round(latitude * 100)}_${Math.round(longitude * 100)}`;
        const now = Date.now();
        const lastAlert = this.lastAlertTime[zoneKey] || 0;

        // Only alert once every 5 minutes for the same zone
        if (now - lastAlert > 5 * 60 * 1000) {
          this.triggerDangerAlert(assessment);
          this.lastAlertTime[zoneKey] = now;
        }
      }

    } catch (error) {
      console.error('[Geofencing] Failed to check danger zones:', error);
    }
  }

  /**
   * Trigger danger zone alert
   */
  triggerDangerAlert(assessment) {
    // Strong vibration pattern
    Vibration.vibrate([0, 500, 200, 500, 200, 500]);

    // Show full-screen alert
    Alert.alert(
      '⚠️ 위험 지역 진입 경고',
      `현재 위치의 위험도가 높습니다!\n\n위험도: ${assessment.riskScore}/100\n주요 위험: ${assessment.primaryThreat || '알 수 없음'}\n\n안전한 경로로 우회하거나 즉시 대피하세요.`,
      [
        {
          text: '경로 재탐색',
          onPress: () => {
            // Trigger route recalculation avoiding this area
            console.log('[Geofencing] Recalculate route');
          },
        },
        {
          text: '확인',
          style: 'cancel',
        },
      ],
      {
        cancelable: false, // Force user to acknowledge
      }
    );

    console.log(`[Geofence] Danger alert triggered: Risk ${assessment.riskScore}`);
  }

  /**
   * Get monitoring status
   */
  getStatus() {
    return {
      isMonitoring: this.isMonitoring,
      dangerZonesCount: this.dangerZones.length,
    };
  }
}

export default new GeofencingService();
