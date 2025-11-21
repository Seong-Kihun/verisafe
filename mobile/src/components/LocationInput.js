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
  onMapPress,
  icon = 'location'
}) {
  return (
    <View style={styles.wrapper}>
      <TouchableOpacity
        style={styles.container}
        onPress={onPress}
        activeOpacity={0.8}
        accessible={true}
        accessibilityRole="button"
        accessibilityLabel={`${label} 입력`}
        accessibilityValue={{ text: value ? (value.name || value.address) : placeholder }}
        accessibilityHint="두 번 탭하여 장소를 검색하세요"
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

      {onMapPress && (
        <TouchableOpacity
          style={styles.mapButton}
          onPress={onMapPress}
          activeOpacity={0.8}
          accessible={true}
          accessibilityRole="button"
          accessibilityLabel="지도에서 선택"
          accessibilityHint="두 번 탭하여 지도에서 장소를 직접 선택하세요"
        >
          <Icon name="map" size={20} color={Colors.primary} />
          <Text style={styles.mapButtonText}>지도에서 선택</Text>
        </TouchableOpacity>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  wrapper: {
    marginBottom: Spacing.sm,
  },
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: Colors.surface,
    borderRadius: 12,
    padding: Spacing.md,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  mapButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: Colors.primary + '10',
    borderRadius: 8,
    paddingVertical: Spacing.sm,
    paddingHorizontal: Spacing.md,
    marginTop: Spacing.xs,
    gap: Spacing.xs,
  },
  mapButtonText: {
    ...Typography.labelSmall,
    color: Colors.primary,
    fontWeight: '600',
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

