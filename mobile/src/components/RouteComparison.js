/**
 * RouteComparison - 경로 비교 UI 컴포넌트 (개선 버전)
 *
 * 책임:
 * 1. 모든 경로를 가로 스크롤 카드로 표시
 * 2. 각 경로의 정보를 카드 형태로 시각화
 * 3. 위험도/시간/거리를 바 그래프로 시각화
 * 4. 선택된 경로 하이라이트
 *
 * 개선사항:
 * - 탭 방식 제거, 모든 경로를 스크롤 가능한 카드로 표시
 * - 각 경로 카드에 안전도 등급, 경로 타입 표시
 * - 추천 경로 배지 강조
 */

import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
} from 'react-native';
import { Colors, Spacing, Typography, getRiskColor, getSafetyGrade, getGradeColor, getRouteColor } from '../styles';
import Icon from './icons/Icon';

export default function RouteComparison({
  routes = [],
  selectedRoute,
  onSelect
}) {
  if (routes.length === 0) {
    return (
      <View style={styles.emptyContainer}>
        <Text style={styles.emptyText}>경로 정보가 없습니다.</Text>
      </View>
    );
  }

  // 바 그래프를 위한 최대값 계산
  const maxDuration = Math.max(...routes.map(r => r.duration || 0), 1);
  const maxDistance = Math.max(...routes.map(r => r.distance || 0), 1);
  const maxRisk = 10;

  // 경로 타입별 라벨
  const getRouteTypeLabel = (type) => {
    const labels = {
      'safe': '안전 경로',
      'fast': '빠른 경로',
      'alternative': '대안 경로',
    };
    return labels[type] || '경로';
  };

  // 경로 타입별 아이콘
  const getRouteTypeIcon = (type) => {
    const icons = {
      'safe': 'safe',
      'fast': 'fast',
      'alternative': 'route',
    };
    return icons[type] || 'route';
  };

  // 추천 경로 결정 (가장 안전한 경로)
  const recommendedRoute = routes.reduce((best, route) => {
    if (!best || route.risk_score < best.risk_score) {
      return route;
    }
    return best;
  }, null);

  // 위험 구간 수 계산
  // Note: map 내부에서 사용하므로 hook 사용 불가, 함수로 유지
  const getHazardZoneCount = (route) => {
    return route?.hazard_count || 0;
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

  return (
    <View style={styles.container}>
      <Text style={styles.sectionTitle}>모든 경로 ({routes.length}개)</Text>
      <Text style={styles.sectionSubtitle}>
        좌우로 스크롤하여 경로를 비교하고 선택하세요
      </Text>

      {/* 가로 스크롤 카드 */}
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
      >
        {routes.map((route, index) => {
          const isSelected = selectedRoute?.id === route.id;
          const isRecommended = route.id === recommendedRoute?.id;
          const hazardCount = route.hazard_count || 0;
          const safetyGrade = getSafetyGrade(route.risk_score, hazardCount);
          const gradeColor = getGradeColor(safetyGrade);
          const routeColor = getRouteColor(route.type);

          return (
            <TouchableOpacity
              key={route.id}
              style={[
                styles.card,
                isSelected && styles.cardSelected,
              ]}
              onPress={() => onSelect && onSelect(route)}
              activeOpacity={0.7}
            >
              {/* 추천 경로 배지 */}
              {isRecommended && (
                <View style={styles.recommendedBadge}>
                  <Icon name="safe" size={14} color={Colors.textInverse} />
                  <Text style={styles.recommendedText}>추천</Text>
                </View>
              )}

              <View style={styles.cardHeader}>
                <View style={styles.cardHeaderLeft}>
                  <Icon
                    name={getRouteTypeIcon(route.type)}
                    size={22}
                    color={routeColor}
                  />
                  <View style={styles.cardHeaderText}>
                    <Text style={styles.cardTitle}>
                      {getRouteTypeLabel(route.type)}
                    </Text>
                    <Text style={styles.cardSubtitle}>
                      경로 {index + 1}
                    </Text>
                  </View>
                </View>
                {/* 안전도 등급 배지 */}
                <View style={[styles.gradeBadge, { backgroundColor: gradeColor + '20' }]}>
                  <Text style={[styles.gradeText, { color: gradeColor }]}>
                    {safetyGrade}
                  </Text>
                </View>
              </View>

              {/* 주요 정보 */}
              <View style={styles.infoRow}>
                <View style={styles.infoItem}>
                  <Icon name="time" size={16} color={Colors.textSecondary} />
                  <Text style={styles.infoLabel}>시간</Text>
                  <Text style={styles.infoValue}>{route.duration}분</Text>
                </View>
                <View style={styles.infoItem}>
                  <Icon name="route" size={16} color={Colors.textSecondary} />
                  <Text style={styles.infoLabel}>거리</Text>
                  <Text style={styles.infoValue}>{route.distance.toFixed(1)}km</Text>
                </View>
                <View style={styles.infoItem}>
                  <Icon name="warning" size={16} color={getRiskColor(route.risk_score)} />
                  <Text style={styles.infoLabel}>위험도</Text>
                  <Text style={[styles.infoValue, { color: getRiskColor(route.risk_score) }]}>
                    {route.risk_score}/10
                  </Text>
                </View>
              </View>

              {/* 바 그래프 */}
              <View style={styles.graphContainer}>
                {renderBarGraph(
                  '시간',
                  `${route.duration}분`,
                  maxDuration,
                  Colors.primary
                )}
                {renderBarGraph(
                  '거리',
                  `${route.distance.toFixed(1)}km`,
                  maxDistance,
                  Colors.info
                )}
                {renderBarGraph(
                  '위험도',
                  `${route.risk_score}/10`,
                  maxRisk,
                  getRiskColor(route.risk_score)
                )}
              </View>

              {/* 위험 구간 정보 */}
              <View style={styles.hazardInfo}>
                <Icon
                  name="warning"
                  size={16}
                  color={getHazardZoneCount(route) > 3 ? Colors.error : Colors.warning}
                />
                <Text style={[styles.hazardText, {
                  color: getHazardZoneCount(route) > 3 ? Colors.error : Colors.warning
                }]}>
                  위험 구간 {getHazardZoneCount(route)}개
                </Text>
              </View>

              {/* 선택 버튼 */}
              {isSelected ? (
                <View style={styles.selectedIndicator}>
                  <Icon name="check" size={18} color={Colors.textInverse} />
                  <Text style={styles.selectedText}>선택됨</Text>
                </View>
              ) : (
                <View style={styles.selectButton}>
                  <Text style={styles.selectButtonText}>선택</Text>
                </View>
              )}
            </TouchableOpacity>
          );
        })}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    marginBottom: Spacing.lg,
  },
  sectionTitle: {
    ...Typography.h3,
    color: Colors.textPrimary,
    marginBottom: Spacing.xs,
  },
  sectionSubtitle: {
    ...Typography.bodySmall,
    color: Colors.textSecondary,
    marginBottom: Spacing.md,
  },
  scrollView: {
    marginHorizontal: -Spacing.lg, // 카드가 화면 밖으로 살짝 나가도록
  },
  scrollContent: {
    paddingHorizontal: Spacing.lg,
    // gap 대신 마지막 카드를 제외하고 marginRight 사용 (호환성)
  },
  card: {
    width: 280, // 고정 너비
    backgroundColor: Colors.surfaceElevated,
    borderRadius: 16,
    padding: Spacing.lg,
    borderWidth: 2,
    borderColor: Colors.border,
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 3,
    position: 'relative',
    marginRight: Spacing.md, // 카드 간 간격
  },
  cardSelected: {
    borderColor: Colors.primary,
    borderWidth: 3,
    shadowOpacity: 0.2,
    elevation: 6,
  },
  // 추천 경로 배지
  recommendedBadge: {
    position: 'absolute',
    top: -8,
    right: Spacing.lg,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: Colors.success,
    paddingVertical: 4,
    paddingHorizontal: Spacing.sm,
    borderRadius: 12,
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
    fontSize: 11,
    marginLeft: 4, // Icon과의 간격 (gap 대신)
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: Spacing.md,
  },
  cardHeaderLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  cardHeaderText: {
    flex: 1,
    marginLeft: Spacing.sm, // Icon과의 간격 (gap 대신)
  },
  cardTitle: {
    ...Typography.h4,
    color: Colors.textPrimary,
    marginBottom: 2,
  },
  cardSubtitle: {
    ...Typography.bodySmall,
    color: Colors.textSecondary,
    fontSize: 12,
  },
  // 안전도 등급 배지
  gradeBadge: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
  gradeText: {
    ...Typography.h3,
    fontSize: 20,
    fontWeight: '700',
  },
  // 주요 정보
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: Spacing.md,
    paddingBottom: Spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: Colors.borderLight,
  },
  infoItem: {
    flex: 1,
    alignItems: 'center',
  },
  infoLabel: {
    ...Typography.labelSmall,
    color: Colors.textSecondary,
    fontSize: 11,
    marginTop: 4, // Icon과의 간격 (gap 대신)
  },
  infoValue: {
    ...Typography.label,
    color: Colors.textPrimary,
    fontWeight: '700',
    fontSize: 14,
    marginTop: 4, // infoLabel과의 간격 (gap 대신)
  },
  // 바 그래프
  graphContainer: {
    marginBottom: Spacing.md,
    // gap 제거: barContainer에 이미 marginBottom 있음
  },
  barContainer: {
    marginBottom: Spacing.sm, // gap 역할
  },
  barLabel: {
    ...Typography.labelSmall,
    color: Colors.textSecondary,
    marginBottom: 4,
    fontSize: 11,
  },
  barWrapper: {
    flexDirection: 'row',
    alignItems: 'center',
    height: 20,
    backgroundColor: Colors.borderLight,
    borderRadius: 10,
    overflow: 'hidden',
  },
  bar: {
    height: '100%',
    borderRadius: 10,
  },
  barValue: {
    ...Typography.labelSmall,
    color: Colors.textPrimary,
    marginLeft: Spacing.xs,
    fontWeight: '600',
    fontSize: 11,
  },
  // 위험 구간 정보
  hazardInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: Spacing.xs,
    paddingHorizontal: Spacing.sm,
    backgroundColor: Colors.background,
    borderRadius: 8,
    marginBottom: Spacing.md,
  },
  hazardText: {
    ...Typography.bodySmall,
    fontWeight: '600',
    fontSize: 12,
    marginLeft: 4, // Icon과의 간격 (gap 대신)
  },
  // 선택 버튼
  selectButton: {
    backgroundColor: Colors.borderLight,
    borderRadius: 8,
    paddingVertical: Spacing.sm,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: Colors.border,
  },
  selectButtonText: {
    ...Typography.buttonSmall,
    color: Colors.textPrimary,
    fontWeight: '600',
  },
  selectedIndicator: {
    backgroundColor: Colors.primary,
    borderRadius: 8,
    paddingVertical: Spacing.sm,
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'center',
  },
  selectedText: {
    ...Typography.buttonSmall,
    color: Colors.textInverse,
    fontWeight: '700',
    marginLeft: Spacing.xs, // Icon과의 간격 (gap 대신)
  },
  emptyContainer: {
    padding: Spacing.xl,
    alignItems: 'center',
  },
  emptyText: {
    ...Typography.body,
    color: Colors.textSecondary,
  },
});
