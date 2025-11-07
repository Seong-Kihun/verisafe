/**
 * ReportSuccessModal.js - 제보 성공 모달
 * 제출 성공 시 애니메이션과 함께 표시
 */

import React, { useEffect, useRef } from 'react';
import { View, Text, Modal, StyleSheet, Animated, TouchableOpacity } from 'react-native';
import Icon from './icons/Icon';
import { Colors, Typography, Spacing } from '../styles';

export default function ReportSuccessModal({ visible, onClose, impactCount }) {
  const scaleAnim = useRef(new Animated.Value(0)).current;
  const fadeAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    if (visible) {
      // 체크 마크 확장 애니메이션
      Animated.sequence([
        Animated.spring(scaleAnim, {
          toValue: 1,
          tension: 50,
          friction: 7,
          useNativeDriver: true,
        }),
        Animated.timing(fadeAnim, {
          toValue: 1,
          duration: 300,
          useNativeDriver: true,
        }),
      ]).start();

      // 2초 후 자동 닫기
      const timer = setTimeout(() => {
        onClose();
      }, 3000);

      return () => clearTimeout(timer);
    } else {
      // 리셋
      scaleAnim.setValue(0);
      fadeAnim.setValue(0);
    }
  }, [visible]);

  return (
    <Modal
      visible={visible}
      transparent
      animationType="fade"
      onRequestClose={onClose}
    >
      <View style={styles.overlay}>
        <View style={styles.container}>
          {/* 성공 아이콘 */}
          <Animated.View
            style={[
              styles.iconContainer,
              {
                transform: [{ scale: scaleAnim }],
              },
            ]}
          >
            <View style={styles.iconCircle}>
              <Icon name="check-box" size={64} color={Colors.textInverse} />
            </View>
          </Animated.View>

          {/* 메시지 */}
          <Animated.View
            style={[
              styles.messageContainer,
              {
                opacity: fadeAnim,
              },
            ]}
          >
            <Text style={styles.title}>제보 완료!</Text>
            <Text style={styles.message}>
              귀중한 정보 감사합니다
            </Text>

            {impactCount > 0 && (
              <View style={styles.impactContainer}>
                <Icon name="person" size={20} color={Colors.primary} />
                <Text style={styles.impactText}>
                  {impactCount}명이 이 정보를 유용하게 봤습니다
                </Text>
              </View>
            )}

            <View style={styles.infoContainer}>
              <Icon name="info" size={16} color={Colors.textSecondary} />
              <Text style={styles.infoText}>
                검증 후 지도에 표시됩니다
              </Text>
            </View>
          </Animated.View>

          {/* 닫기 버튼 */}
          <Animated.View
            style={{
              opacity: fadeAnim,
            }}
          >
            <TouchableOpacity
              style={styles.closeButton}
              onPress={onClose}
              activeOpacity={0.8}
            >
              <Text style={styles.closeButtonText}>확인</Text>
            </TouchableOpacity>
          </Animated.View>
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  container: {
    backgroundColor: Colors.surface,
    borderRadius: 20,
    padding: Spacing.xl,
    alignItems: 'center',
    maxWidth: 320,
    width: '80%',
    elevation: 8,
    shadowColor: Colors.shadowDark,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 12,
  },
  iconContainer: {
    marginBottom: Spacing.lg,
  },
  iconCircle: {
    width: 100,
    height: 100,
    borderRadius: 50,
    backgroundColor: Colors.success,
    alignItems: 'center',
    justifyContent: 'center',
    elevation: 4,
    shadowColor: Colors.success,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
  },
  messageContainer: {
    alignItems: 'center',
    marginBottom: Spacing.lg,
  },
  title: {
    ...Typography.h2,
    color: Colors.textPrimary,
    marginBottom: Spacing.xs,
  },
  message: {
    ...Typography.body,
    color: Colors.textSecondary,
    textAlign: 'center',
    marginBottom: Spacing.md,
  },
  impactContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: `${Colors.primary}10`,
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.sm,
    borderRadius: 20,
    marginBottom: Spacing.sm,
    gap: Spacing.xs,
  },
  impactText: {
    ...Typography.caption,
    color: Colors.primary,
    fontWeight: '600',
  },
  infoContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.xs,
    marginTop: Spacing.sm,
  },
  infoText: {
    ...Typography.caption,
    color: Colors.textSecondary,
  },
  closeButton: {
    backgroundColor: Colors.primary,
    paddingHorizontal: Spacing.xl,
    paddingVertical: Spacing.md,
    borderRadius: 12,
    minWidth: 120,
  },
  closeButtonText: {
    ...Typography.button,
    color: Colors.textInverse,
    textAlign: 'center',
  },
});
