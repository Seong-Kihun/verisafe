/**
 * SeverityPicker.js - 심각도 선택 컴포넌트
 * 위험의 심각도를 선택 (경미, 중간, 심각)
 */

import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import Icon from './icons/Icon';
import { Colors, Typography, Spacing } from '../styles';

const SEVERITY_LEVELS = [
  {
    value: 'low',
    label: '경미',
    description: '우회 가능',
    color: '#10B981',
    icon: 'info',
  },
  {
    value: 'medium',
    label: '중간',
    description: '주의 필요',
    color: '#F59E0B',
    icon: 'warning',
  },
  {
    value: 'high',
    label: '심각',
    description: '즉시 회피',
    color: '#DC2626',
    icon: 'dangerous',
  },
];

export default function SeverityPicker({ value, onChange }) {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>심각도</Text>

      <View style={styles.optionsContainer}>
        {SEVERITY_LEVELS.map((level) => {
          const isSelected = value === level.value;

          return (
            <TouchableOpacity
              key={level.value}
              style={[
                styles.option,
                isSelected && styles.optionSelected,
                isSelected && { borderColor: level.color },
              ]}
              onPress={() => onChange(level.value)}
              activeOpacity={0.7}
            >
              <View
                style={[
                  styles.iconContainer,
                  { backgroundColor: `${level.color}20` },
                ]}
              >
                <Icon name={level.icon} size={24} color={level.color} />
              </View>

              <View style={styles.textContainer}>
                <Text
                  style={[
                    styles.label,
                    isSelected && { color: level.color },
                  ]}
                >
                  {level.label}
                </Text>
                <Text style={styles.description}>{level.description}</Text>
              </View>

              {isSelected && (
                <View style={[styles.checkmark, { backgroundColor: level.color }]}>
                  <Icon name="check-box" size={20} color={Colors.textInverse} />
                </View>
              )}
            </TouchableOpacity>
          );
        })}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    marginVertical: Spacing.md,
  },
  title: {
    ...Typography.body,
    fontWeight: '600',
    color: Colors.textPrimary,
    marginBottom: Spacing.sm,
  },
  optionsContainer: {
    gap: Spacing.sm,
  },
  option: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: Spacing.md,
    backgroundColor: Colors.surface,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: Colors.border,
  },
  optionSelected: {
    backgroundColor: Colors.surfaceElevated,
    borderWidth: 2,
  },
  iconContainer: {
    width: 48,
    height: 48,
    borderRadius: 24,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: Spacing.md,
  },
  textContainer: {
    flex: 1,
  },
  label: {
    ...Typography.body,
    fontWeight: '600',
    color: Colors.textPrimary,
    marginBottom: 2,
  },
  description: {
    ...Typography.caption,
    color: Colors.textSecondary,
  },
  checkmark: {
    width: 28,
    height: 28,
    borderRadius: 14,
    alignItems: 'center',
    justifyContent: 'center',
  },
});
