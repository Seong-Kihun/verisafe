/**
 * LayerToggleMenu.js - 위험 정보 레이어 토글 메뉴
 * 우측 상단 레이어 버튼 클릭 시 표시되는 모달
 */

import React from 'react';
import { View, Text, TouchableOpacity, Modal, StyleSheet, ScrollView } from 'react-native';
import Icon from './icons/Icon';
import { Colors, Typography, Spacing } from '../styles';
import { getHazardColor } from '../styles/colors';

const HAZARD_TYPES = [
  { id: 'armed_conflict', name: '무력충돌', icon: 'conflict', color: '#DC2626' },
  { id: 'conflict', name: '충돌', icon: 'conflict', color: '#EF4444' },
  { id: 'protest_riot', name: '시위/폭동', icon: 'protest', color: '#F59E0B' },
  { id: 'protest', name: '시위', icon: 'protest', color: '#F97316' },
  { id: 'checkpoint', name: '검문소', icon: 'checkpoint', color: '#FF6B6B' },
  { id: 'road_damage', name: '도로 손상', icon: 'roadDamage', color: '#F97316' },
  { id: 'natural_disaster', name: '자연재해', icon: 'naturalDisaster', color: '#DC2626' },
  { id: 'flood', name: '홍수', icon: 'naturalDisaster', color: '#3B82F6' },
  { id: 'landslide', name: '산사태', icon: 'naturalDisaster', color: '#92400E' },
  { id: 'other', name: '기타', icon: 'other', color: '#6B7280' },
];

// Phase 4: 시간대 필터
const TIME_FILTERS = [
  { id: 'all', name: '전체', hours: null },
  { id: '24h', name: '24시간', hours: 24 },
  { id: '48h', name: '48시간', hours: 48 },
  { id: '7d', name: '7일', hours: 168 },
];

export default function LayerToggleMenu({
  visible,
  onClose,
  activeTypes,
  onToggle,
  timeFilter = 'all',
  onTimeFilterChange,
}) {
  return (
    <Modal
      visible={visible}
      transparent
      animationType="fade"
      onRequestClose={onClose}
    >
      <TouchableOpacity
        style={styles.overlay}
        onPress={onClose}
        activeOpacity={1}
      >
        <View style={styles.menuContainer}>
          <TouchableOpacity activeOpacity={1}>
            <View style={styles.menu}>
              <View style={styles.header}>
                <Text style={styles.title}>레이어 선택</Text>
                <TouchableOpacity onPress={onClose} style={styles.closeButton}>
                  <Icon name="close" size={24} color={Colors.textSecondary} />
                </TouchableOpacity>
              </View>

              <ScrollView style={styles.itemsContainer}>
                {/* Phase 4: 시간대 필터 */}
                <View style={styles.timeFilterSection}>
                  <Text style={styles.sectionTitle}>시간대 필터</Text>
                  <View style={styles.timeFilterButtons}>
                    {TIME_FILTERS.map(filter => {
                      const isActive = timeFilter === filter.id;
                      return (
                        <TouchableOpacity
                          key={filter.id}
                          style={[
                            styles.timeFilterButton,
                            isActive && styles.timeFilterButtonActive
                          ]}
                          onPress={() => onTimeFilterChange && onTimeFilterChange(filter.id)}
                          activeOpacity={0.7}
                        >
                          <Text style={[
                            styles.timeFilterText,
                            isActive && styles.timeFilterTextActive
                          ]}>
                            {filter.name}
                          </Text>
                        </TouchableOpacity>
                      );
                    })}
                  </View>
                </View>

                {/* 위험 유형 */}
                <Text style={styles.sectionTitle}>위험 유형</Text>
                {HAZARD_TYPES.map(type => {
                  const isActive = activeTypes.includes(type.id);
                  return (
                    <TouchableOpacity
                      key={type.id}
                      style={styles.item}
                      onPress={() => onToggle(type.id)}
                      activeOpacity={0.7}
                    >
                      <View style={styles.itemLeft}>
                        <Icon
                          name={type.icon}
                          size={24}
                          color={type.color}
                        />
                        <Text style={styles.itemText}>{type.name}</Text>
                      </View>
                      <View style={styles.itemRight}>
                        <View style={[styles.badge, { backgroundColor: type.color }]} />
                        <Icon
                          name={isActive ? 'check-box' : 'check-box-outline-blank'}
                          size={24}
                          color={isActive ? type.color : Colors.textTertiary}
                        />
                      </View>
                    </TouchableOpacity>
                  );
                })}
              </ScrollView>

              <View style={styles.footer}>
                <TouchableOpacity
                  style={styles.applyButton}
                  onPress={onClose}
                  activeOpacity={0.8}
                >
                  <Text style={styles.applyButtonText}>적용</Text>
                </TouchableOpacity>
              </View>
            </View>
          </TouchableOpacity>
        </View>
      </TouchableOpacity>
    </Modal>
  );
}

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'flex-start',
    alignItems: 'flex-end',
  },
  menuContainer: {
    marginTop: 60,
    marginRight: Spacing.lg,
  },
  menu: {
    backgroundColor: Colors.surfaceElevated,
    borderRadius: 12,
    minWidth: 240,
    maxWidth: 300,
    maxHeight: 500,
    elevation: 8,
    shadowColor: Colors.shadowDark,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.25,
    shadowRadius: 8,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: Spacing.lg,
    paddingVertical: Spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: Colors.border,
  },
  title: {
    ...Typography.h3,
    color: Colors.textPrimary,
  },
  closeButton: {
    padding: Spacing.xs,
  },
  itemsContainer: {
    maxHeight: 360,
  },
  item: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: Spacing.lg,
    paddingVertical: Spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: Colors.borderLight,
  },
  itemLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  itemText: {
    ...Typography.body,
    color: Colors.textPrimary,
    marginLeft: Spacing.md,
  },
  itemRight: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
  },
  badge: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  footer: {
    paddingHorizontal: Spacing.lg,
    paddingVertical: Spacing.md,
    borderTopWidth: 1,
    borderTopColor: Colors.border,
  },
  applyButton: {
    backgroundColor: Colors.primary,
    paddingVertical: Spacing.md,
    borderRadius: 8,
    alignItems: 'center',
  },
  applyButtonText: {
    ...Typography.button,
    color: Colors.textInverse,
  },
  // Phase 4: 시간대 필터 스타일
  timeFilterSection: {
    paddingHorizontal: Spacing.lg,
    paddingVertical: Spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: Colors.border,
  },
  sectionTitle: {
    ...Typography.label,
    color: Colors.textSecondary,
    marginBottom: Spacing.sm,
    paddingHorizontal: Spacing.lg,
  },
  timeFilterButtons: {
    flexDirection: 'row',
    gap: Spacing.xs,
    flexWrap: 'wrap',
  },
  timeFilterButton: {
    paddingVertical: Spacing.xs,
    paddingHorizontal: Spacing.md,
    borderRadius: 16,
    backgroundColor: Colors.borderLight,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  timeFilterButtonActive: {
    backgroundColor: Colors.primary + '20',
    borderColor: Colors.primary,
  },
  timeFilterText: {
    ...Typography.labelSmall,
    color: Colors.textSecondary,
    fontWeight: '600',
  },
  timeFilterTextActive: {
    color: Colors.primary,
  },
});
