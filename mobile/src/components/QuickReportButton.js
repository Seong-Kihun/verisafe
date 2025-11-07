/**
 * QuickReportButton.js - 빠른 제보 플로팅 버튼
 * 지도 위에 표시되는 빠른 제보 버튼
 */

import React from 'react';
import { TouchableOpacity, Text, StyleSheet, View } from 'react-native';
import Icon from './icons/Icon';
import { Colors, Typography, Spacing } from '../styles';

export default function QuickReportButton({ onPress, style }) {
  return (
    <TouchableOpacity
      style={[styles.button, style]}
      onPress={onPress}
      activeOpacity={0.9}
    >
      <View style={styles.iconContainer}>
        <Icon name="report" size={24} color={Colors.textInverse} />
      </View>
      <Text style={styles.text}>빠른 제보</Text>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  button: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: Colors.danger,
    paddingVertical: Spacing.md,
    paddingHorizontal: Spacing.lg,
    borderRadius: 28,
    elevation: 6,
    shadowColor: Colors.shadowDark,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    gap: Spacing.sm,
  },
  iconContainer: {
    width: 28,
    height: 28,
    alignItems: 'center',
    justifyContent: 'center',
  },
  text: {
    ...Typography.button,
    color: Colors.textInverse,
    fontWeight: '700',
  },
});
