/**
 * RouteComparison - 경로 비교 UI 컴포넌트 (Phase 2 개선)
 *
 * 책임:
 * 1. 여러 경로를 탭으로 전환 (Safe/Fast/Alternative)
 * 2. 각 경로의 정보를 카드 형태로 표시
 * 3. 위험도/시간/거리를 바 그래프로 시각화
 * 4. Step 1-3의 디자인 토큰 적용
 *
 * Phase 2 개선사항:
 * - 추천 경로 배지 강조
 * - 안전도 등급 표시
 * - 위험 구간 수 표시
 */

import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
} from 'react-native';
import { Colors, Spacing, Typography, getRiskColor, getSafetyGrade, getGradeColor } from '../styles';
import Icon from './icons/Icon';

const ROUTE_TYPES = [
  { id: 'safe', label: '안전 경로', icon: 'safe' },
  { id: 'fast', label: '빠른 경로', icon: 'fast' },
];

export default function RouteComparison({ 
  routes = [], 
  selectedRoute, 
  onSelect 
}) {
  const [activeTab, setActiveTab] = useState('safe');

  // 경로 타입별로 그룹화
  const routesByType = {
    safe: routes.find(r => r.type === 'safe') || null,
    fast: routes.find(r => r.type === 'fast') || null,
  };

  const currentRoute = routesByType[activeTab];

  // 바 그래프를 위한 최대값 계산
  const maxDuration = Math.max(...routes.map(r => r.duration || 0), 1);
  const maxDistance = Math.max(...routes.map(r => r.distance || 0), 1);
  const maxRisk = 10;

  // Phase 2: 위험 구간 수 계산
  const getHazardZoneCount = (route) => {
    return Math.floor(route.risk_score / 2);
  };

  // Phase 2: 추천 경로 결정 (안전 경로를 기본 추천)
  const isRecommended = (route) => {
    return route.type === 'safe';
  };

  const renderBarGraph = (label, value, maxValue, color) => {
    const percentage = (value / maxValue) * 100;
    return (
      <View style={styles.barContainer}>
        <Text style={styles.barLabel}>{label}</Text>
        <View style={styles.barWrapper}>
          <View style={[styles.bar, { width: `${percentage}%`, backgroundColor: color }]} />
          <Text style={styles.barValue}>{value}</Text>
        </View>
      </View>
    );
  };

  if (!currentRoute) {
    return (
      <View style={styles.emptyContainer}>
        <Text style={styles.emptyText}>경로 정보가 없습니다.</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* 탭 전환 */}
      <View style={styles.tabContainer}>
        {ROUTE_TYPES.map((type) => {
          const route = routesByType[type.id];
          const isActive = activeTab === type.id;
          const isAvailable = route !== null;
          
          return (
            <TouchableOpacity
              key={type.id}
              style={[
                styles.tab,
                isActive && styles.tabActive,
                !isAvailable && styles.tabDisabled
              ]}
              onPress={() => isAvailable && setActiveTab(type.id)}
              disabled={!isAvailable}
              activeOpacity={0.7}
            >
              <Icon 
                name={type.icon} 
                size={18} 
                color={isActive ? Colors.primary : Colors.textSecondary} 
              />
              <Text style={[
                styles.tabText,
                isActive && styles.tabTextActive
              ]}>
                {type.label}
              </Text>
            </TouchableOpacity>
          );
        })}
      </View>

      {/* 경로 정보 카드 */}
      <View style={styles.card}>
        {/* Phase 2: 추천 경로 배지 */}
        {isRecommended(currentRoute) && (
          <View style={styles.recommendedBadge}>
            <Icon name="safe" size={16} color={Colors.textInverse} />
            <Text style={styles.recommendedText}>추천 경로</Text>
          </View>
        )}

        <View style={styles.cardHeader}>
          <View style={styles.cardHeaderLeft}>
            <Icon
              name={currentRoute.type}
              size={24}
              color={Colors.primary}
            />
            <View style={styles.cardHeaderText}>
              <Text style={styles.cardTitle}>
                {ROUTE_TYPES.find(t => t.id === activeTab)?.label}
              </Text>
              <Text style={styles.cardSubtitle}>
                {currentRoute.transportation_mode === 'car' ? '차량' :
                 currentRoute.transportation_mode === 'walking' ? '도보' : '자전거'}
              </Text>
            </View>
          </View>
          {/* Phase 2: 안전도 등급 배지 */}
          <View style={[styles.gradeBadge, { backgroundColor: getGradeColor(getSafetyGrade(currentRoute.risk_score)) + '20' }]}>
            <Text style={[styles.gradeText, { color: getGradeColor(getSafetyGrade(currentRoute.risk_score)) }]}>
              {getSafetyGrade(currentRoute.risk_score)}
            </Text>
          </View>
        </View>

        {/* 바 그래프 */}
        <View style={styles.graphContainer}>
          {renderBarGraph(
            '시간',
            `${currentRoute.duration}분`,
            maxDuration,
            Colors.primary
          )}
          {renderBarGraph(
            '거리',
            `${currentRoute.distance.toFixed(1)}km`,
            maxDistance,
            Colors.info
          )}
          {renderBarGraph(
            '위험도',
            `${currentRoute.risk_score}/10`,
            maxRisk,
            getRiskColor(currentRoute.risk_score)
          )}
        </View>

        {/* Phase 2: 위험 구간 수 표시 */}
        <View style={styles.hazardInfo}>
          <Icon
            name="warning"
            size={18}
            color={getHazardZoneCount(currentRoute) > 3 ? Colors.error : Colors.warning}
          />
          <Text style={[styles.hazardText, {
            color: getHazardZoneCount(currentRoute) > 3 ? Colors.error : Colors.warning
          }]}>
            위험 구간 {getHazardZoneCount(currentRoute)}개
          </Text>
        </View>

      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    marginBottom: Spacing.lg,
  },
  tabContainer: {
    flexDirection: 'row',
    marginBottom: Spacing.md,
    gap: Spacing.sm,
  },
  tab: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: Colors.borderLight,
    borderRadius: 12,
    paddingVertical: Spacing.sm,
    paddingHorizontal: Spacing.sm,
    borderWidth: 2,
    borderColor: Colors.border,
    gap: Spacing.xs,
  },
  tabActive: {
    backgroundColor: Colors.primary + '20',
    borderColor: Colors.primary,
  },
  tabDisabled: {
    opacity: 0.5,
  },
  tabText: {
    ...Typography.labelSmall,
    color: Colors.textSecondary,
  },
  tabTextActive: {
    color: Colors.primary,
    fontWeight: '600',
  },
  card: {
    backgroundColor: Colors.surfaceElevated,
    borderRadius: 16,
    padding: Spacing.lg,
    borderWidth: 2,
    borderColor: Colors.border,
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.12,
    shadowRadius: 8,
    elevation: 4,
    position: 'relative',
  },
  // Phase 2: 추천 경로 배지
  recommendedBadge: {
    position: 'absolute',
    top: -10,
    right: Spacing.lg,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: Colors.success,
    paddingVertical: Spacing.xs,
    paddingHorizontal: Spacing.md,
    borderRadius: 16,
    gap: Spacing.xs,
    shadowColor: Colors.success,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
    elevation: 4,
    zIndex: 10,
  },
  recommendedText: {
    ...Typography.labelSmall,
    color: Colors.textInverse,
    fontWeight: '700',
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: Spacing.lg,
  },
  cardHeaderLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
    gap: Spacing.sm,
  },
  cardHeaderText: {
    flex: 1,
  },
  cardTitle: {
    ...Typography.h3,
    color: Colors.textPrimary,
    marginBottom: 2,
  },
  cardSubtitle: {
    ...Typography.bodySmall,
    color: Colors.textSecondary,
  },
  // Phase 2: 안전도 등급 배지
  gradeBadge: {
    width: 44,
    height: 44,
    borderRadius: 22,
    alignItems: 'center',
    justifyContent: 'center',
  },
  gradeText: {
    ...Typography.h3,
    fontSize: 22,
    fontWeight: '700',
  },
  graphContainer: {
    marginBottom: Spacing.lg,
    gap: Spacing.md,
  },
  barContainer: {
    marginBottom: Spacing.sm,
  },
  barLabel: {
    ...Typography.labelSmall,
    color: Colors.textSecondary,
    marginBottom: Spacing.xs,
  },
  barWrapper: {
    flexDirection: 'row',
    alignItems: 'center',
    height: 24,
    backgroundColor: Colors.borderLight,
    borderRadius: 12,
    overflow: 'hidden',
  },
  bar: {
    height: '100%',
    borderRadius: 12,
  },
  barValue: {
    ...Typography.labelSmall,
    color: Colors.textPrimary,
    marginLeft: Spacing.sm,
    fontWeight: '600',
  },
  selectButton: {
    backgroundColor: Colors.borderLight,
    borderRadius: 12,
    paddingVertical: Spacing.md,
    alignItems: 'center',
    borderWidth: 2,
    borderColor: Colors.border,
  },
  selectButtonActive: {
    backgroundColor: Colors.primary + '20',
    borderColor: Colors.primary,
  },
  selectButtonText: {
    ...Typography.buttonSmall,
    color: Colors.textPrimary,
    fontWeight: '600',
  },
  selectButtonTextActive: {
    color: Colors.primary,
  },
  emptyContainer: {
    padding: Spacing.xl,
    alignItems: 'center',
  },
  emptyText: {
    ...Typography.body,
    color: Colors.textSecondary,
  },
  // Phase 2: 위험 구간 정보
  hazardInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: Spacing.xs,
    paddingVertical: Spacing.sm,
    paddingHorizontal: Spacing.md,
    backgroundColor: Colors.background,
    borderRadius: 12,
  },
  hazardText: {
    ...Typography.bodySmall,
    fontWeight: '600',
  },
});

