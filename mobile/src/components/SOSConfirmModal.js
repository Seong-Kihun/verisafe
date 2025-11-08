/**
 * SOSConfirmModal - SOS 발동 확인 모달
 *
 * 기능:
 * - 5초 카운트다운 (자동 발송)
 * - 취소 버튼으로 중단 가능
 * - 긴급 연락처 수 표시
 * - 발송될 정보 미리보기 (위치, 메시지)
 */

import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Modal,
  TouchableOpacity,
  ActivityIndicator,
  Animated,
} from 'react-native';
import { Colors, Spacing, Typography } from '../styles';
import Icon from './icons/Icon';

export default function SOSConfirmModal({
  visible,
  onConfirm,
  onCancel,
  emergencyContactsCount = 0,
  userLocation,
}) {
  const [countdown, setCountdown] = useState(5);
  const [isSending, setIsSending] = useState(false);
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const countdownInterval = useRef(null);

  // 펄스 애니메이션
  useEffect(() => {
    if (!visible || isSending) return;

    const pulse = Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 1.1,
          duration: 500,
          useNativeDriver: true,
        }),
        Animated.timing(pulseAnim, {
          toValue: 1,
          duration: 500,
          useNativeDriver: true,
        }),
      ])
    );
    pulse.start();

    return () => pulse.stop();
  }, [visible, isSending, pulseAnim]);

  // 카운트다운
  useEffect(() => {
    if (!visible) {
      setCountdown(5);
      setIsSending(false);
      if (countdownInterval.current) {
        clearInterval(countdownInterval.current);
      }
      return;
    }

    setCountdown(5);
    countdownInterval.current = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          clearInterval(countdownInterval.current);
          handleAutoConfirm();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => {
      if (countdownInterval.current) {
        clearInterval(countdownInterval.current);
      }
    };
  }, [visible]);

  const handleAutoConfirm = async () => {
    setIsSending(true);
    await onConfirm();
    // onConfirm이 모달을 닫는 책임을 가짐
  };

  const handleCancel = () => {
    if (countdownInterval.current) {
      clearInterval(countdownInterval.current);
    }
    setCountdown(5);
    setIsSending(false);
    onCancel();
  };

  return (
    <Modal
      visible={visible}
      transparent
      animationType="fade"
      onRequestClose={handleCancel}
    >
      <View style={styles.overlay}>
        <View style={styles.modal}>
          {isSending ? (
            // 전송 중
            <View style={styles.sendingContainer}>
              <ActivityIndicator size="large" color={Colors.danger} />
              <Text style={styles.sendingText}>긴급 SOS 발송 중...</Text>
              <Text style={styles.sendingSubtext}>
                {emergencyContactsCount}명의 긴급 연락처로 전송합니다
              </Text>
            </View>
          ) : (
            // 확인 화면
            <>
              <Animated.View style={[styles.iconContainer, { transform: [{ scale: pulseAnim }] }]}>
                <View style={styles.iconCircle}>
                  <Icon name="warning" size={48} color={Colors.textInverse} />
                </View>
              </Animated.View>

              <Text style={styles.title}>긴급 SOS</Text>
              <Text style={styles.description}>
                {emergencyContactsCount > 0
                  ? `${emergencyContactsCount}명의 긴급 연락처로\n위치와 도움 요청 메시지를 전송합니다`
                  : '등록된 긴급 연락처가 없습니다'}
              </Text>

              {/* 카운트다운 */}
              <View style={styles.countdownContainer}>
                <Text style={styles.countdownNumber}>{countdown}</Text>
                <Text style={styles.countdownLabel}>초 후 자동 발송</Text>
              </View>

              {/* 전송될 정보 미리보기 */}
              {userLocation && (
                <View style={styles.infoPreview}>
                  <View style={styles.infoRow}>
                    <Icon name="locationOn" size={16} color={Colors.textSecondary} />
                    <Text style={styles.infoText}>
                      현재 위치: {userLocation.latitude.toFixed(5)}, {userLocation.longitude.toFixed(5)}
                    </Text>
                  </View>
                  <View style={styles.infoRow}>
                    <Icon name="time" size={16} color={Colors.textSecondary} />
                    <Text style={styles.infoText}>
                      발송 시간: {new Date().toLocaleTimeString('ko-KR')}
                    </Text>
                  </View>
                </View>
              )}

              {/* 버튼 */}
              <View style={styles.buttons}>
                <TouchableOpacity
                  style={styles.cancelButton}
                  onPress={handleCancel}
                  activeOpacity={0.8}
                >
                  <Text style={styles.cancelButtonText}>취소</Text>
                </TouchableOpacity>

                <TouchableOpacity
                  style={styles.confirmButton}
                  onPress={handleAutoConfirm}
                  activeOpacity={0.8}
                  disabled={emergencyContactsCount === 0}
                >
                  <Text style={styles.confirmButtonText}>즉시 발송</Text>
                </TouchableOpacity>
              </View>

              {emergencyContactsCount === 0 && (
                <Text style={styles.warningText}>
                  긴급 연락처를 먼저 등록해주세요
                </Text>
              )}
            </>
          )}
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: Spacing.xl,
  },
  modal: {
    backgroundColor: Colors.surface,
    borderRadius: 24,
    padding: Spacing.xl,
    width: '100%',
    maxWidth: 400,
    alignItems: 'center',
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.3,
    shadowRadius: 16,
    elevation: 12,
  },
  iconContainer: {
    marginBottom: Spacing.lg,
  },
  iconCircle: {
    width: 96,
    height: 96,
    borderRadius: 48,
    backgroundColor: Colors.danger,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: Colors.danger,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.4,
    shadowRadius: 8,
    elevation: 8,
  },
  title: {
    ...Typography.h2,
    color: Colors.danger,
    marginBottom: Spacing.sm,
    fontWeight: '700',
  },
  description: {
    ...Typography.body,
    color: Colors.textSecondary,
    textAlign: 'center',
    marginBottom: Spacing.lg,
    lineHeight: 22,
  },
  countdownContainer: {
    alignItems: 'center',
    marginBottom: Spacing.lg,
    paddingVertical: Spacing.md,
    paddingHorizontal: Spacing.xl,
    backgroundColor: Colors.danger + '10',
    borderRadius: 16,
    borderWidth: 2,
    borderColor: Colors.danger + '30',
  },
  countdownNumber: {
    fontSize: 48,
    fontWeight: '700',
    color: Colors.danger,
    lineHeight: 56,
  },
  countdownLabel: {
    ...Typography.bodySmall,
    color: Colors.danger,
    fontWeight: '600',
  },
  infoPreview: {
    width: '100%',
    backgroundColor: Colors.background,
    borderRadius: 12,
    padding: Spacing.md,
    marginBottom: Spacing.lg,
  },
  infoRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: Spacing.xs,
  },
  infoText: {
    ...Typography.bodySmall,
    color: Colors.textSecondary,
    marginLeft: Spacing.xs,
    flex: 1,
  },
  buttons: {
    flexDirection: 'row',
    width: '100%',
    marginTop: Spacing.sm,
  },
  cancelButton: {
    flex: 1,
    backgroundColor: Colors.surface,
    borderRadius: 12,
    paddingVertical: Spacing.md,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: Colors.border,
    marginRight: Spacing.sm,
  },
  cancelButtonText: {
    ...Typography.button,
    color: Colors.textSecondary,
    fontWeight: '600',
  },
  confirmButton: {
    flex: 1,
    backgroundColor: Colors.danger,
    borderRadius: 12,
    paddingVertical: Spacing.md,
    alignItems: 'center',
    marginLeft: Spacing.sm,
  },
  confirmButtonText: {
    ...Typography.button,
    color: Colors.textInverse,
    fontWeight: '700',
  },
  sendingContainer: {
    alignItems: 'center',
    paddingVertical: Spacing.xl,
  },
  sendingText: {
    ...Typography.h3,
    color: Colors.textPrimary,
    marginTop: Spacing.lg,
    marginBottom: Spacing.xs,
  },
  sendingSubtext: {
    ...Typography.body,
    color: Colors.textSecondary,
  },
  warningText: {
    ...Typography.bodySmall,
    color: Colors.warning,
    marginTop: Spacing.md,
    textAlign: 'center',
  },
});
