/**
 * Safety check-in service
 * 안전 체크인 서비스
 *
 * 경로 시작 시 자동으로 체크인을 등록하고,
 * 예상 도착 시간 이후에 도착 확인을 하지 않으면
 * 긴급 연락망에 자동 알림을 전송합니다.
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import { Alert } from 'react-native';
import { emergencyContactsStorage } from './storage';

const STORAGE_KEY = '@verisafe_active_checkin';

class SafetyCheckinService {
  constructor() {
    this.activeCheckin = null;
    this.checkInterval = null;
  }

  /**
   * Register a safety check-in
   */
  async register(route, estimatedArrivalTime) {
    try {
      const checkin = {
        id: `checkin_${Date.now()}`,
        routeId: route.id || null,
        origin: route.origin,
        destination: route.destination,
        estimatedArrivalTime,
        status: 'active',
        createdAt: new Date().toISOString(),
      };

      // Save to local storage
      await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(checkin));
      this.activeCheckin = checkin;

      // Start monitoring
      this.startMonitoring();

      console.log('[Safety Check-in] Registered:', checkin);
      return checkin;

    } catch (error) {
      console.error('Failed to register safety check-in:', error);
      return null;
    }
  }

  /**
   * Confirm safe arrival
   */
  async confirm() {
    try {
      if (!this.activeCheckin) {
        return false;
      }

      // Update status
      this.activeCheckin.status = 'confirmed';
      this.activeCheckin.confirmedAt = new Date().toISOString();

      // Clear from storage
      await AsyncStorage.removeItem(STORAGE_KEY);

      // Stop monitoring
      this.stopMonitoring();

      console.log('[Safety Check-in] Confirmed');
      this.activeCheckin = null;
      return true;

    } catch (error) {
      console.error('Failed to confirm check-in:', error);
      return false;
    }
  }

  /**
   * Cancel check-in
   */
  async cancel() {
    try {
      if (!this.activeCheckin) {
        return false;
      }

      this.activeCheckin.status = 'cancelled';
      await AsyncStorage.removeItem(STORAGE_KEY);
      this.stopMonitoring();

      console.log('[Safety Check-in] Cancelled');
      this.activeCheckin = null;
      return true;

    } catch (error) {
      console.error('Failed to cancel check-in:', error);
      return false;
    }
  }

  /**
   * Start monitoring for overdue check-ins
   */
  startMonitoring() {
    if (this.checkInterval) {
      return;
    }

    // Check every minute
    this.checkInterval = setInterval(() => {
      this.checkOverdue();
    }, 60 * 1000);
  }

  /**
   * Stop monitoring
   */
  stopMonitoring() {
    if (this.checkInterval) {
      clearInterval(this.checkInterval);
      this.checkInterval = null;
    }
  }

  /**
   * Check if check-in is overdue
   */
  async checkOverdue() {
    try {
      if (!this.activeCheckin || this.activeCheckin.status !== 'active') {
        return;
      }

      const eta = new Date(this.activeCheckin.estimatedArrivalTime);
      const now = new Date();
      const etaPlus30 = new Date(eta);
      etaPlus30.setMinutes(etaPlus30.getMinutes() + 30);

      // If current time is past ETA + 30 minutes
      if (now > etaPlus30) {
        await this.triggerMissedAlert();
      }

    } catch (error) {
      console.error('Error checking overdue:', error);
    }
  }

  /**
   * Trigger alert for missed check-in
   */
  async triggerMissedAlert() {
    try {
      // Update status
      this.activeCheckin.status = 'missed';
      await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(this.activeCheckin));

      // Get emergency contacts
      const contacts = await emergencyContactsStorage.getAll();

      // Show alert to user
      Alert.alert(
        '⚠️ 안전 체크인 놓침',
        `예정된 도착 시간이 지났습니다.\n\n긴급 연락망 ${contacts.length}명에게 알림이 전송됩니다.`,
        [
          {
            text: '안전 확인',
            onPress: () => this.confirm(),
          },
          {
            text: '나중에',
            style: 'cancel',
          },
        ]
      );

      console.log('[Safety Check-in] MISSED - Alerting contacts');

      // In production, this would send actual SMS/Push notifications
      // For now, just log it
      console.log(`[Safety Check-in] Would alert ${contacts.length} emergency contacts`);

    } catch (error) {
      console.error('Failed to trigger missed alert:', error);
    }
  }

  /**
   * Get active check-in
   */
  async getActive() {
    try {
      if (this.activeCheckin) {
        return this.activeCheckin;
      }

      const stored = await AsyncStorage.getItem(STORAGE_KEY);
      if (stored) {
        this.activeCheckin = JSON.parse(stored);
        this.startMonitoring();
        return this.activeCheckin;
      }

      return null;

    } catch (error) {
      console.error('Failed to get active check-in:', error);
      return null;
    }
  }
}

export default new SafetyCheckinService();
