/**
 * SMS 발송 서비스
 * expo-sms를 사용하여 긴급 연락처에 SMS 전송
 */

import * as SMS from 'expo-sms';
import * as Linking from 'expo-linking';
import { Alert, Platform } from 'react-native';

/**
 * SMS 발송 가능 여부 확인
 */
export const isSMSAvailable = async () => {
  try {
    const isAvailable = await SMS.isAvailableAsync();
    return isAvailable;
  } catch (error) {
    console.error('[SMS] Failed to check SMS availability:', error);
    return false;
  }
};

/**
 * 긴급 연락처에 SOS SMS 발송
 *
 * @param {Array} contacts - 긴급 연락처 배열
 * @param {Object} userLocation - 사용자 위치 { latitude, longitude }
 * @param {string} userName - 사용자 이름
 * @returns {Promise<boolean>} - 성공 여부
 */
export const sendSOSSMS = async (contacts, userLocation, userName = '사용자') => {
  try {
    // 입력 유효성 검사
    if (!contacts || !Array.isArray(contacts) || contacts.length === 0) {
      console.error('[SMS] Invalid contacts:', contacts);
      Alert.alert('오류', '등록된 긴급 연락처가 없습니다.');
      return false;
    }

    if (!userLocation || typeof userLocation.latitude !== 'number' || typeof userLocation.longitude !== 'number') {
      console.error('[SMS] Invalid userLocation:', userLocation);
      Alert.alert('오류', '현재 위치를 확인할 수 없습니다.');
      return false;
    }

    // SMS 사용 가능 여부 확인
    const available = await isSMSAvailable();

    if (!available) {
      console.warn('[SMS] SMS is not available on this device');
      // SMS를 사용할 수 없으면 전화 앱 열기로 대체
      return await fallbackToPhoneApp(contacts, userLocation, userName);
    }

    // 전화번호 추출 (우선순위 순)
    const phoneNumbers = contacts
      .sort((a, b) => (a.priority || 999) - (b.priority || 999))
      .map(c => c.phone)
      .filter(phone => phone && typeof phone === 'string' && phone.trim().length > 0);

    if (phoneNumbers.length === 0) {
      console.error('[SMS] No valid phone numbers found');
      Alert.alert('오류', '유효한 전화번호가 없습니다.');
      return false;
    }

    // SMS 메시지 작성 (에러 발생 가능)
    let message;
    try {
      message = createSOSMessage(userLocation, userName);
    } catch (msgError) {
      console.error('[SMS] Failed to create message:', msgError);
      Alert.alert('오류', 'SOS 메시지 생성에 실패했습니다.');
      return false;
    }

    // SMS 전송
    const { result } = await SMS.sendSMSAsync(phoneNumbers, message);

    if (result === 'sent') {
      console.log('[SMS] SOS SMS sent successfully to', phoneNumbers.length, 'contacts');
      return true;
    } else if (result === 'cancelled') {
      console.log('[SMS] User cancelled SMS');
      return false;
    } else {
      console.warn('[SMS] SMS send result:', result);
      return false;
    }
  } catch (error) {
    console.error('[SMS] Failed to send SOS SMS:', error);
    Alert.alert('오류', 'SMS 전송에 실패했습니다.');
    return false;
  }
};

/**
 * SOS 메시지 생성
 */
const createSOSMessage = (userLocation, userName) => {
  // userLocation 유효성 검사
  if (!userLocation || typeof userLocation.latitude !== 'number' || typeof userLocation.longitude !== 'number') {
    console.error('[SMS] Invalid userLocation:', userLocation);
    throw new Error('유효하지 않은 위치 정보입니다.');
  }

  const { latitude, longitude } = userLocation;
  const timestamp = new Date().toLocaleString('ko-KR', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });

  // Google Maps 링크 생성
  const googleMapsLink = `https://www.google.com/maps?q=${latitude},${longitude}`;

  const message = `🆘 긴급 SOS 알림 🆘

${userName}님이 긴급 상황에 처했습니다.

📅 시간: ${timestamp}
📍 위치:
위도: ${latitude.toFixed(6)}
경도: ${longitude.toFixed(6)}

🗺️ 지도에서 위치 확인:
${googleMapsLink}

즉시 확인하고 연락해주세요!

- VeriSafe 긴급 알림 시스템`;

  return message;
};

/**
 * SMS를 사용할 수 없을 때 전화 앱 열기
 * (예: 시뮬레이터에서 테스트할 때)
 */
const fallbackToPhoneApp = async (contacts, userLocation, userName) => {
  try {
    if (!contacts || contacts.length === 0) {
      console.warn('[SMS] No contacts available for fallback');
      return false;
    }

    // 첫 번째 연락처에게 전화 걸기
    const sortedContacts = [...contacts].sort((a, b) => a.priority - b.priority);
    const firstContact = sortedContacts[0];

    if (!firstContact || !firstContact.phone) {
      console.error('[SMS] First contact has no phone number');
      return false;
    }

    const phoneNumber = firstContact.phone;

    // Promise를 사용하여 사용자의 선택을 기다림
    return new Promise((resolve) => {
      Alert.alert(
        'SMS 사용 불가',
        `이 기기에서는 SMS를 사용할 수 없습니다.\n${firstContact.name}(${phoneNumber})에게 직접 전화하시겠습니까?`,
        [
          {
            text: '취소',
            style: 'cancel',
            onPress: () => resolve(false)
          },
          {
            text: '전화 걸기',
            onPress: async () => {
              try {
                const url = `tel:${phoneNumber}`;
                const supported = await Linking.canOpenURL(url);
                if (supported) {
                  await Linking.openURL(url);
                  resolve(true);
                } else {
                  Alert.alert('오류', '전화를 걸 수 없습니다.');
                  resolve(false);
                }
              } catch (error) {
                console.error('[SMS] Error opening phone app:', error);
                Alert.alert('오류', '전화를 걸 수 없습니다.');
                resolve(false);
              }
            },
          },
        ],
        { cancelable: false } // 바깥 클릭으로 닫지 못하게
      );
    });
  } catch (error) {
    console.error('[SMS] Fallback error:', error);
    return false;
  }
};

/**
 * 단일 연락처에게 SMS 발송 (테스트용)
 */
export const sendSingleSMS = async (phoneNumber, message) => {
  try {
    const available = await isSMSAvailable();
    if (!available) {
      Alert.alert('알림', 'SMS를 사용할 수 없습니다.');
      return false;
    }

    const { result } = await SMS.sendSMSAsync([phoneNumber], message);
    return result === 'sent';
  } catch (error) {
    console.error('[SMS] Failed to send single SMS:', error);
    return false;
  }
};
