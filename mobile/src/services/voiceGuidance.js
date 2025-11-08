/**
 * 음성 안내 서비스
 * expo-speech를 사용한 턴바이턴 음성 안내
 */

import * as Speech from 'expo-speech';

class VoiceGuidanceService {
  constructor() {
    this.isEnabled = true;
    this.isSpeaking = false;
    this.language = 'ko-KR';
    this.rate = 0.9; // 말하기 속도 (0.5 - 2.0)
    this.pitch = 1.0; // 음높이
  }

  /**
   * 음성 안내 설정
   * @param {Object} options - {enabled, language, rate, pitch}
   */
  configure(options = {}) {
    if (options.enabled !== undefined) {
      this.isEnabled = options.enabled;
    }
    if (options.language) {
      this.language = options.language;
    }
    if (options.rate !== undefined) {
      this.rate = Math.max(0.5, Math.min(2.0, options.rate));
    }
    if (options.pitch !== undefined) {
      this.pitch = Math.max(0.5, Math.min(2.0, options.pitch));
    }
  }

  /**
   * 음성 안내 실행
   * @param {string} text - 안내 문구
   * @param {Object} options - 추가 옵션
   */
  async speak(text, options = {}) {
    if (!this.isEnabled || !text) {
      return;
    }

    try {
      // 현재 재생 중인 음성 중지
      if (this.isSpeaking) {
        await Speech.stop();
      }

      this.isSpeaking = true;

      const speechOptions = {
        language: options.language || this.language,
        pitch: options.pitch || this.pitch,
        rate: options.rate || this.rate,
        onDone: () => {
          this.isSpeaking = false;
        },
        onError: (error) => {
          console.error('[VoiceGuidance] Speech error:', error);
          this.isSpeaking = false;
        },
      };

      await Speech.speak(text, speechOptions);
      console.log('[VoiceGuidance] Speaking:', text);
    } catch (error) {
      console.error('[VoiceGuidance] Failed to speak:', error);
      this.isSpeaking = false;
    }
  }

  /**
   * 방향 안내 음성
   * @param {string} instruction - 방향 안내 ("좌회전", "직진" 등)
   * @param {number} distance - 거리 (미터)
   */
  async speakDirection(instruction, distance) {
    if (distance < 50) {
      await this.speak(`곧 ${instruction}`);
    } else if (distance < 100) {
      await this.speak(`${Math.round(distance)}미터 후 ${instruction}`);
    } else if (distance < 500) {
      await this.speak(`${Math.round(distance / 10) * 10}미터 후 ${instruction}`);
    } else {
      const roundedDistance = Math.round(distance / 100) * 100;
      await this.speak(`${roundedDistance}미터 후 ${instruction}`);
    }
  }

  /**
   * 위험 경고 음성
   * @param {string} hazardType - 위험 유형
   * @param {number} distance - 거리 (미터)
   */
  async speakHazardWarning(hazardType, distance) {
    const roundedDistance = Math.round(distance / 10) * 10;
    await this.speak(`주의! ${roundedDistance}미터 앞에 ${hazardType}이 있습니다`);
  }

  /**
   * 목적지 도착 안내
   */
  async speakArrival() {
    await this.speak('목적지에 도착했습니다');
  }

  /**
   * 경로 이탈 안내
   */
  async speakOffRoute() {
    await this.speak('경로를 벗어났습니다');
  }

  /**
   * 경로 재탐색 안내
   */
  async speakRerouting() {
    await this.speak('경로를 다시 찾고 있습니다');
  }

  /**
   * 네비게이션 시작 안내
   * @param {number} distance - 총 거리 (미터)
   * @param {number} duration - 예상 시간 (초)
   */
  async speakNavigationStart(distance, duration) {
    const km = (distance / 1000).toFixed(1);
    const minutes = Math.round(duration / 60);

    await this.speak(
      `목적지까지 ${km}킬로미터, 약 ${minutes}분 소요됩니다. 안내를 시작합니다`
    );
  }

  /**
   * 음성 중지
   */
  async stop() {
    try {
      await Speech.stop();
      this.isSpeaking = false;
    } catch (error) {
      console.error('[VoiceGuidance] Failed to stop speech:', error);
    }
  }

  /**
   * 음성 안내 활성화/비활성화
   * @param {boolean} enabled
   */
  setEnabled(enabled) {
    this.isEnabled = enabled;
    if (!enabled && this.isSpeaking) {
      this.stop();
    }
  }

  /**
   * 음성 안내 활성화 상태 확인
   * @returns {boolean}
   */
  isVoiceEnabled() {
    return this.isEnabled;
  }
}

// 싱글톤 인스턴스 export
export default new VoiceGuidanceService();
