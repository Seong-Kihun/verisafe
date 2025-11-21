/**
 * RouteInputBar - 경로 입력 바 컴포넌트
 * 출발지와 목적지를 표시하는 듀얼 입력창
 *
 * 책임:
 * 1. 출발지와 목적지를 분리된 입력창으로 표시
 * 2. 각 위치를 클릭하여 수정 가능
 * 3. 경로가 선택된 상태에서 지도 상단에 표시
 */

import React from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
} from 'react-native';
import { Colors, Typography, Spacing } from '../styles';
import Icon from './icons/Icon';

export default function RouteInputBar({
  startLocation,
  endLocation,
  onStartPress,
  onEndPress,
  onSwap,
  onClose
}) {
  return (
    <View style={styles.container}>
      <View style={styles.inputBar}>
        {/* 위치 교환 버튼 */}
        {startLocation && endLocation && onSwap && (
          <TouchableOpacity
            style={styles.swapButton}
            onPress={onSwap}
            activeOpacity={0.7}
          >
            <Icon name="swap" size={20} color={Colors.textSecondary} />
          </TouchableOpacity>
        )}

        {/* 입력 영역 */}
        <View style={styles.inputContainer}>
          {/* 출발지 입력 */}
          <TouchableOpacity
            style={styles.inputRow}
            onPress={onStartPress}
            activeOpacity={0.7}
          >
            <View style={[styles.dot, styles.startDot]} />
            <Text
              style={styles.inputText}
              numberOfLines={1}
            >
              {startLocation?.name || '출발지'}
            </Text>
          </TouchableOpacity>

          {/* 구분선 */}
          <View style={styles.divider} />

          {/* 목적지 입력 */}
          <TouchableOpacity
            style={styles.inputRow}
            onPress={onEndPress}
            activeOpacity={0.7}
          >
            <View style={[styles.dot, styles.endDot]} />
            <Text
              style={styles.inputText}
              numberOfLines={1}
            >
              {endLocation?.name || '목적지'}
            </Text>
          </TouchableOpacity>
        </View>

        {/* 닫기 버튼 */}
        {onClose && (
          <TouchableOpacity
            style={styles.closeButton}
            onPress={onClose}
            activeOpacity={0.7}
          >
            <Icon name="close" size={20} color={Colors.textSecondary} />
          </TouchableOpacity>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  inputBar: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    borderRadius: 12,
    paddingVertical: Spacing.sm,
    paddingLeft: Spacing.xs,
    paddingRight: Spacing.xs,
    // Shadow (shadowMedium 적용)
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.12,
    shadowRadius: 8,
    elevation: 4,
  },
  swapButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: Colors.surface,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: Colors.border,
    marginRight: Spacing.xs,
  },
  inputContainer: {
    flex: 1,
  },
  inputRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: Spacing.sm,
    minHeight: 40,
  },
  dot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: Spacing.md,
  },
  startDot: {
    backgroundColor: Colors.primary,
  },
  endDot: {
    backgroundColor: Colors.error,
  },
  inputText: {
    ...Typography.body,
    color: Colors.textPrimary,
    flex: 1,
  },
  divider: {
    height: 1,
    backgroundColor: Colors.border,
    marginLeft: 12 + Spacing.md, // dot 너비 + 여백
  },
  closeButton: {
    width: 36,
    height: 36,
    justifyContent: 'center',
    alignItems: 'center',
    marginLeft: Spacing.xs,
  },
});
