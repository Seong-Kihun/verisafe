/**
 * TransportationModeSelector - 이동 수단 선택 컴포넌트
 */

import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { Colors, Spacing, Typography } from '../styles';
import Icon from './icons/Icon';

const MODES = [
  { id: 'car', name: '차량', icon: 'car' },
  { id: 'walking', name: '도보', icon: 'walking' },
  { id: 'bicycle', name: '자전거', icon: 'bicycle' },
];

export default function TransportationModeSelector({ 
  selectedMode, 
  onSelect 
}) {
  return (
    <View style={styles.container}>
      {MODES.map((mode) => (
        <TouchableOpacity
          key={mode.id}
          style={[
            styles.button,
            selectedMode === mode.id && styles.buttonActive
          ]}
          onPress={() => onSelect(mode.id)}
          activeOpacity={0.8}
        >
          <Icon 
            name={mode.icon} 
            size={20} 
            color={selectedMode === mode.id ? Colors.primary : Colors.textSecondary} 
          />
          <Text
            style={[
              styles.text,
              selectedMode === mode.id && styles.textActive
            ]}
          >
            {mode.name}
          </Text>
        </TouchableOpacity>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    marginBottom: Spacing.lg,
    gap: Spacing.sm,
  },
  button: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: Colors.borderLight,
    borderRadius: 12,
    paddingVertical: Spacing.md,
    paddingHorizontal: Spacing.sm,
    borderWidth: 2,
    borderColor: Colors.border,
    minHeight: Spacing.buttonHeight,  // 48px
    gap: Spacing.xs,
  },
  buttonActive: {
    backgroundColor: Colors.primary + '20',
    borderColor: Colors.primary,
  },
  text: {
    ...Typography.buttonSmall,
    color: Colors.textPrimary,
  },
  textActive: {
    color: Colors.primary,
    fontWeight: '600',
  },
});

