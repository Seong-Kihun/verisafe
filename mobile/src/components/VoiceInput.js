/**
 * VoiceInput.js - 음성 입력 컴포넌트
 * Expo Speech Recognition을 사용하여 음성을 텍스트로 변환
 */

import React, { useState, useEffect } from 'react';
import { View, TouchableOpacity, Text, StyleSheet, Platform, Alert } from 'react-native';
import * as Speech from 'expo-speech';
import Icon from './icons/Icon';
import { Colors, Typography, Spacing } from '../styles';

export default function VoiceInput({ onResult, style }) {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');

  /**
   * 음성 인식 시작
   *
   * 참고: expo-speech는 TTS(Text-to-Speech)만 지원하고
   * STT(Speech-to-Text)는 지원하지 않습니다.
   *
   * 실제 구현을 위해서는:
   * - iOS: expo-speech-recognition 또는 react-native-voice
   * - Android: @react-native-voice/voice
   * - Web: Web Speech API
   *
   * 현재는 UI만 구현하고 실제 음성 인식은 향후 추가
   */
  const startListening = async () => {
    try {
      // 권한 체크
      if (Platform.OS === 'ios' || Platform.OS === 'android') {
        // 실제 구현 시 권한 요청
        // const { status } = await requestMicrophonePermission();
        // if (status !== 'granted') {
        //   Alert.alert('권한 필요', '음성 인식을 사용하려면 마이크 권한이 필요합니다.');
        //   return;
        // }
      }

      setIsListening(true);

      // 실제 구현 예시 (react-native-voice 사용 시):
      // await Voice.start('ko-KR');
      // Voice.onSpeechResults = (e) => {
      //   const text = e.value[0];
      //   setTranscript(text);
      //   onResult(text);
      // };

      // 데모용: 3초 후 자동 종료
      setTimeout(() => {
        stopListening();
        // 데모 텍스트
        const demoText = '검문소에서 대기 시간이 길어요';
        setTranscript(demoText);
        onResult(demoText);
      }, 3000);
    } catch (error) {
      console.error('Failed to start listening:', error);
      Alert.alert('오류', '음성 인식을 시작할 수 없습니다.');
      setIsListening(false);
    }
  };

  /**
   * 음성 인식 중지
   */
  const stopListening = async () => {
    try {
      // 실제 구현 시:
      // await Voice.stop();
      // Voice.removeAllListeners();

      setIsListening(false);
    } catch (error) {
      console.error('Failed to stop listening:', error);
    }
  };

  /**
   * 버튼 클릭 핸들러
   */
  const handlePress = () => {
    if (isListening) {
      stopListening();
    } else {
      startListening();
    }
  };

  return (
    <View style={[styles.container, style]}>
      <TouchableOpacity
        style={[
          styles.button,
          isListening && styles.buttonListening,
        ]}
        onPress={handlePress}
        activeOpacity={0.7}
      >
        <Icon
          name={isListening ? 'stop' : 'mic'}
          size={24}
          color={isListening ? Colors.textInverse : Colors.primary}
        />
        <Text
          style={[
            styles.buttonText,
            isListening && styles.buttonTextListening,
          ]}
        >
          {isListening ? '듣는 중...' : '음성 입력'}
        </Text>
      </TouchableOpacity>

      {isListening && (
        <View style={styles.waveContainer}>
          <View style={[styles.wave, styles.wave1]} />
          <View style={[styles.wave, styles.wave2]} />
          <View style={[styles.wave, styles.wave3]} />
        </View>
      )}

      {!isListening && transcript && (
        <View style={styles.transcriptContainer}>
          <Icon name="check-box" size={16} color={Colors.success} />
          <Text style={styles.transcriptText} numberOfLines={2}>
            "{transcript}"
          </Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    marginVertical: Spacing.md,
  },
  button: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: `${Colors.primary}10`,
    paddingVertical: Spacing.md,
    paddingHorizontal: Spacing.lg,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: Colors.primary,
    gap: Spacing.sm,
  },
  buttonListening: {
    backgroundColor: Colors.primary,
  },
  buttonText: {
    ...Typography.body,
    color: Colors.primary,
    fontWeight: '600',
  },
  buttonTextListening: {
    color: Colors.textInverse,
  },
  waveContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: Spacing.md,
    gap: Spacing.xs,
  },
  wave: {
    width: 4,
    backgroundColor: Colors.primary,
    borderRadius: 2,
  },
  wave1: {
    height: 20,
    animation: 'wave 0.8s ease-in-out infinite',
  },
  wave2: {
    height: 30,
    animation: 'wave 0.8s ease-in-out infinite 0.2s',
  },
  wave3: {
    height: 20,
    animation: 'wave 0.8s ease-in-out infinite 0.4s',
  },
  transcriptContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: Spacing.md,
    padding: Spacing.sm,
    backgroundColor: `${Colors.success}10`,
    borderRadius: 8,
    gap: Spacing.xs,
  },
  transcriptText: {
    ...Typography.body,
    color: Colors.success,
    flex: 1,
    fontStyle: 'italic',
  },
});
