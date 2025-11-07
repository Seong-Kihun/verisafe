/**
 * LocationInput - 출발지/목적지 입력 컴포넌트
 */

import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { Colors, Spacing, Typography } from '../styles';
import Icon from './icons/Icon';

export default function LocationInput({ 
  label, 
  value, 
  placeholder, 
  onPress,
  icon = 'location'
}) {
  return (
    <TouchableOpacity 
      style={styles.container} 
      onPress={onPress}
      activeOpacity={0.8}
    >
      <View style={styles.iconContainer}>
        <Icon name={icon} size={20} color={Colors.primary} />
      </View>
      <View style={styles.content}>
        <Text style={styles.label}>{label}</Text>
        <Text 
          style={[styles.text, !value && styles.placeholder]}
          numberOfLines={1}
        >
          {value ? value.name || value.address : placeholder}
        </Text>
        {value && value.address && (
          <Text style={styles.address} numberOfLines={1}>
            {value.address}
          </Text>
        )}
      </View>
      <Text style={styles.chevron}>›</Text>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: Colors.surface,
    borderRadius: 12,
    padding: Spacing.md,
    marginBottom: Spacing.sm,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  iconContainer: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: Colors.primary + '20',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: Spacing.md,
  },
  content: {
    flex: 1,
  },
  label: {
    ...Typography.labelSmall,
    color: Colors.textSecondary,
    marginBottom: Spacing.xs,
  },
  text: {
    ...Typography.body,
    color: Colors.textPrimary,
    fontWeight: '500',
  },
  placeholder: {
    color: Colors.textTertiary,
    fontWeight: '400',
  },
  address: {
    ...Typography.bodySmall,
    color: Colors.textSecondary,
    marginTop: 2,
  },
  chevron: {
    fontSize: 24,
    color: Colors.textTertiary,
    marginLeft: Spacing.sm,
  },
});

