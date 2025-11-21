/**
 * 긴급 제스처 감지 서비스
 * 화면 탭, 볼륨 버튼 등 제스처로 긴급 알림 발송
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import { Alert } from 'react-native';
import * as Location from 'expo-location';
import { sendSOSSMS } from './sms';

// 설정 키
const GESTURE_SETTINGS_KEY = '@verisafe_gesture_settings';
const EMERGENCY_CONTACTS_KEY = '@verisafe_emergency_contacts';

// 기본 설정 (기본적으로 비활성화 - 긴급연락처 화면에서 활성화 필요)
const DEFAULT_SETTINGS = {
  tapGestureEnabled: false, // 기본적으로 꺼짐
  volumeGestureEnabled: false, // 기본적으로 꺼짐
  tapCount: 5, // 몇 번 탭해야 하는지
  tapTimeout: 3000, // 몇 초 안에 탭해야 하는지 (ms)
  volumePressCount: 3, // 볼륨 버튼 몇 번 눌러야 하는지
  volumeTimeout: 2000, // 볼륨 버튼 몇 초 안에 눌러야 하는지 (ms)
};

// 제스처 설정 관리
export class GestureSettings {
  /**
   * 제스처 설정 불러오기
   */
  static async load() {
    try {
      const stored = await AsyncStorage.getItem(GESTURE_SETTINGS_KEY);
      if (stored) {
        return { ...DEFAULT_SETTINGS, ...JSON.parse(stored) };
      }
      return DEFAULT_SETTINGS;
    } catch (error) {
      console.error('[GestureSettings] Failed to load settings:', error);
      return DEFAULT_SETTINGS;
    }
  }

  /**
   * 제스처 설정 저장
   */
  static async save(settings) {
    try {
      await AsyncStorage.setItem(GESTURE_SETTINGS_KEY, JSON.stringify(settings));
      return true;
    } catch (error) {
      console.error('[GestureSettings] Failed to save settings:', error);
      return false;
    }
  }

  /**
   * 특정 설정 업데이트
   */
  static async update(key, value) {
    try {
      const settings = await this.load();
      settings[key] = value;
      await this.save(settings);
      return true;
    } catch (error) {
      console.error('[GestureSettings] Failed to update setting:', error);
      return false;
    }
  }
}

// 탭 제스처 감지기
export class TapGestureDetector {
  constructor() {
    this.tapCount = 0;
    this.lastTapTime = 0;
    this.timeout = null;
    this.onTrigger = null;
  }

  /**
   * 탭 이벤트 처리
   * @param {number} requiredTaps - 필요한 탭 횟수
   * @param {number} timeWindow - 시간 제한 (ms)
   * @returns {boolean} - 제스처 트리거 여부
   */
  handleTap(requiredTaps = 5, timeWindow = 3000) {
    const now = Date.now();

    // 시간 초과 시 초기화
    if (now - this.lastTapTime > timeWindow) {
      this.tapCount = 0;
    }

    this.tapCount++;
    this.lastTapTime = now;

    console.log(`[TapGesture] Tap ${this.tapCount}/${requiredTaps}`);

    // 필요한 탭 횟수 도달
    if (this.tapCount >= requiredTaps) {
      this.tapCount = 0;
      if (this.onTrigger) {
        this.onTrigger();
      }
      return true;
    }

    // 타임아웃 설정 (시간 초과 시 초기화)
    if (this.timeout) {
      clearTimeout(this.timeout);
    }
    this.timeout = setTimeout(() => {
      if (this.tapCount < requiredTaps) {
        console.log('[TapGesture] Timeout - resetting');
        this.tapCount = 0;
      }
    }, timeWindow);

    return false;
  }

  /**
   * 트리거 콜백 설정
   */
  setOnTrigger(callback) {
    this.onTrigger = callback;
  }

  /**
   * 리셋
   */
  reset() {
    this.tapCount = 0;
    this.lastTapTime = 0;
    if (this.timeout) {
      clearTimeout(this.timeout);
      this.timeout = null;
    }
  }
}

// 볼륨 버튼 제스처 감지기
export class VolumeGestureDetector {
  constructor() {
    this.pressCount = 0;
    this.lastPressTime = 0;
    this.timeout = null;
    this.onTrigger = null;
  }

  /**
   * 볼륨 버튼 이벤트 처리
   * @param {number} requiredPresses - 필요한 누름 횟수
   * @param {number} timeWindow - 시간 제한 (ms)
   * @returns {boolean} - 제스처 트리거 여부
   */
  handlePress(requiredPresses = 3, timeWindow = 2000) {
    const now = Date.now();

    // 시간 초과 시 초기화
    if (now - this.lastPressTime > timeWindow) {
      this.pressCount = 0;
    }

    this.pressCount++;
    this.lastPressTime = now;

    console.log(`[VolumeGesture] Press ${this.pressCount}/${requiredPresses}`);

    // 필요한 누름 횟수 도달
    if (this.pressCount >= requiredPresses) {
      this.pressCount = 0;
      if (this.onTrigger) {
        this.onTrigger();
      }
      return true;
    }

    // 타임아웃 설정
    if (this.timeout) {
      clearTimeout(this.timeout);
    }
    this.timeout = setTimeout(() => {
      if (this.pressCount < requiredPresses) {
        console.log('[VolumeGesture] Timeout - resetting');
        this.pressCount = 0;
      }
    }, timeWindow);

    return false;
  }

  /**
   * 트리거 콜백 설정
   */
  setOnTrigger(callback) {
    this.onTrigger = callback;
  }

  /**
   * 리셋
   */
  reset() {
    this.pressCount = 0;
    this.lastPressTime = 0;
    if (this.timeout) {
      clearTimeout(this.timeout);
      this.timeout = null;
    }
  }
}

/**
 * 긴급 SOS 발송
 */
export const triggerEmergencySOS = async () => {
  try {
    console.log('[EmergencyGesture] Triggering SOS...');

    // 확인 알림 (실수 방지)
    return new Promise((resolve) => {
      Alert.alert(
        '🆘 긴급 SOS',
        '긴급 연락처에 SOS 메시지를 발송하시겠습니까?',
        [
          {
            text: '취소',
            style: 'cancel',
            onPress: () => resolve(false),
          },
          {
            text: '즉시 발송',
            style: 'destructive',
            onPress: async () => {
              try {
                // 긴급 연락처 불러오기
                const contactsData = await AsyncStorage.getItem(EMERGENCY_CONTACTS_KEY);
                if (!contactsData) {
                  Alert.alert('오류', '등록된 긴급 연락처가 없습니다.');
                  resolve(false);
                  return;
                }

                const contacts = JSON.parse(contactsData);
                if (!contacts || contacts.length === 0) {
                  Alert.alert('오류', '등록된 긴급 연락처가 없습니다.');
                  resolve(false);
                  return;
                }

                // 현재 위치 가져오기
                let location = null;
                try {
                  const { status } = await Location.requestForegroundPermissionsAsync();
                  if (status === 'granted') {
                    const loc = await Location.getCurrentPositionAsync({
                      accuracy: Location.Accuracy.High,
                    });
                    location = {
                      latitude: loc.coords.latitude,
                      longitude: loc.coords.longitude,
                    };
                  }
                } catch (locError) {
                  console.error('[EmergencyGesture] Failed to get location:', locError);
                  // 위치를 가져오지 못해도 계속 진행 (기본 위치 사용)
                  location = {
                    latitude: 4.8550,
                    longitude: 31.5850,
                  };
                }

                // SOS SMS 발송
                const success = await sendSOSSMS(contacts, location, '사용자');

                if (success) {
                  Alert.alert('✅ SOS 발송 완료', '긴급 연락처에 SOS 메시지를 발송했습니다.');
                  resolve(true);
                } else {
                  Alert.alert('오류', 'SOS 메시지 발송에 실패했습니다.');
                  resolve(false);
                }
              } catch (error) {
                console.error('[EmergencyGesture] Failed to send SOS:', error);
                Alert.alert('오류', 'SOS 메시지 발송 중 오류가 발생했습니다.');
                resolve(false);
              }
            },
          },
        ],
        { cancelable: false }
      );
    });
  } catch (error) {
    console.error('[EmergencyGesture] Failed to trigger SOS:', error);
    Alert.alert('오류', '긴급 알림 발송에 실패했습니다.');
    return false;
  }
};
