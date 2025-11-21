/**
 * EmergencyGestureDetector.js
 * 앱 전체에서 긴급 제스처를 감지하는 컴포넌트
 */

import React, { useEffect, useRef, useState } from 'react';
import {
  View,
  StyleSheet,
  Dimensions,
  TouchableWithoutFeedback,
  Text,
  Animated,
  Platform,
} from 'react-native';
import {
  GestureSettings,
  TapGestureDetector,
  VolumeGestureDetector,
  triggerEmergencySOS,
} from '../services/emergencyGesture';

// 볼륨 버튼 감지 (네이티브 모듈)
let VolumeManager = null;
try {
  VolumeManager = require('react-native-volume-manager').default;
} catch (e) {
  console.log('[EmergencyGestureDetector] Volume manager not available - using Expo Go or module not installed');
}

const { width, height } = Dimensions.get('window');

// 탭 감지 영역 크기 (화면 모서리)
const TAP_ZONE_SIZE = 60;

export default function EmergencyGestureDetector({ children }) {
  const [settings, setSettings] = useState(null);
  const [tapProgress, setTapProgress] = useState(0); // 진행 상태 (0-1)
  const tapDetector = useRef(new TapGestureDetector()).current;
  const volumeDetector = useRef(new VolumeGestureDetector()).current;
  const progressAnim = useRef(new Animated.Value(0)).current;
  const lastVolumeRef = useRef(null);

  // 설정 불러오기
  useEffect(() => {
    loadSettings();
  }, []);

  // 볼륨 변화 감지 설정
  useEffect(() => {
    if (!VolumeManager || !settings || !settings.volumeGestureEnabled) {
      return;
    }

    let volumeListener = null;

    const setupVolumeListener = async () => {
      try {
        // 현재 볼륨 가져오기
        const { volume } = await VolumeManager.getVolume();
        lastVolumeRef.current = volume;

        // 볼륨 변화 리스너 추가
        volumeListener = VolumeManager.addVolumeListener((result) => {
          const { volume: newVolume } = result;

          // 볼륨이 변경되었으면 (버튼이 눌렸으면)
          if (lastVolumeRef.current !== null && newVolume !== lastVolumeRef.current) {
            console.log('[EmergencyGestureDetector] Volume button pressed');
            volumeDetector.handlePress(settings.volumePressCount, settings.volumeTimeout);
          }

          lastVolumeRef.current = newVolume;
        });

        console.log('[EmergencyGestureDetector] Volume listener setup complete');
      } catch (error) {
        console.error('[EmergencyGestureDetector] Failed to setup volume listener:', error);
      }
    };

    setupVolumeListener();

    // 클린업
    return () => {
      if (volumeListener) {
        volumeListener.remove();
      }
    };
  }, [settings]);

  const loadSettings = async () => {
    const loaded = await GestureSettings.load();
    setSettings(loaded);

    // 트리거 콜백 설정
    tapDetector.setOnTrigger(() => {
      console.log('[EmergencyGestureDetector] Tap gesture triggered!');
      triggerEmergencySOS();
    });

    volumeDetector.setOnTrigger(() => {
      console.log('[EmergencyGestureDetector] Volume gesture triggered!');
      triggerEmergencySOS();
    });
  };

  /**
   * 탭 처리
   */
  const handleTap = () => {
    if (!settings || !settings.tapGestureEnabled) {
      return;
    }

    const triggered = tapDetector.handleTap(settings.tapCount, settings.tapTimeout);

    // 진행 상태 업데이트
    const progress = Math.min(tapDetector.tapCount / settings.tapCount, 1);
    setTapProgress(progress);

    // 진행 상태 애니메이션
    Animated.timing(progressAnim, {
      toValue: progress,
      duration: 200,
      useNativeDriver: false,
    }).start();

    // 완료되면 리셋
    if (triggered) {
      setTimeout(() => {
        setTapProgress(0);
        progressAnim.setValue(0);
      }, 1000);
    }

    // 시간 초과 시 리셋
    setTimeout(() => {
      if (tapDetector.tapCount < settings.tapCount) {
        setTapProgress(0);
        progressAnim.setValue(0);
      }
    }, settings.tapTimeout);
  };

  if (!settings || !settings.tapGestureEnabled) {
    return <>{children}</>;
  }

  // 진행 상태 색상
  const progressColor = progressAnim.interpolate({
    inputRange: [0, 0.5, 1],
    outputRange: ['rgba(255, 59, 48, 0.3)', 'rgba(255, 149, 0, 0.5)', 'rgba(52, 199, 89, 0.7)'],
  });

  return (
    <View style={styles.container}>
      {children}

      {/* 좌상단 탭 영역 */}
      <TouchableWithoutFeedback onPress={handleTap}>
        <View style={[styles.tapZone, styles.topLeft]}>
          {tapProgress > 0 && (
            <Animated.View
              style={[
                styles.progressIndicator,
                {
                  backgroundColor: progressColor,
                  opacity: progressAnim,
                },
              ]}
            >
              <Text style={styles.progressText}>
                {tapDetector.tapCount}/{settings.tapCount}
              </Text>
            </Animated.View>
          )}
        </View>
      </TouchableWithoutFeedback>

      {/* 우상단 탭 영역 */}
      <TouchableWithoutFeedback onPress={handleTap}>
        <View style={[styles.tapZone, styles.topRight]}>
          {tapProgress > 0 && (
            <Animated.View
              style={[
                styles.progressIndicator,
                {
                  backgroundColor: progressColor,
                  opacity: progressAnim,
                },
              ]}
            />
          )}
        </View>
      </TouchableWithoutFeedback>

      {/* 좌하단 탭 영역 */}
      <TouchableWithoutFeedback onPress={handleTap}>
        <View style={[styles.tapZone, styles.bottomLeft]}>
          {tapProgress > 0 && (
            <Animated.View
              style={[
                styles.progressIndicator,
                {
                  backgroundColor: progressColor,
                  opacity: progressAnim,
                },
              ]}
            />
          )}
        </View>
      </TouchableWithoutFeedback>

      {/* 우하단 탭 영역 */}
      <TouchableWithoutFeedback onPress={handleTap}>
        <View style={[styles.tapZone, styles.bottomRight]}>
          {tapProgress > 0 && (
            <Animated.View
              style={[
                styles.progressIndicator,
                {
                  backgroundColor: progressColor,
                  opacity: progressAnim,
                },
              ]}
            />
          )}
        </View>
      </TouchableWithoutFeedback>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  tapZone: {
    position: 'absolute',
    width: TAP_ZONE_SIZE,
    height: TAP_ZONE_SIZE,
    zIndex: 9999,
  },
  topLeft: {
    top: 0,
    left: 0,
  },
  topRight: {
    top: 0,
    right: 0,
  },
  bottomLeft: {
    bottom: 0,
    left: 0,
  },
  bottomRight: {
    bottom: 0,
    right: 0,
  },
  progressIndicator: {
    flex: 1,
    borderRadius: TAP_ZONE_SIZE / 2,
    justifyContent: 'center',
    alignItems: 'center',
  },
  progressText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '700',
  },
});
