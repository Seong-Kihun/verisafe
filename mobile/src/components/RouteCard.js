/**
 * RouteCard - 경로 카드 컴포넌트 (Phase 2 개선)
 *
 * 개선사항:
 * - 안전도 등급 (A~F) 표시
 * - 위험 구간 수 표시
 */

import React, { useRef, useEffect } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Animated } from 'react-native';
import { Colors, Spacing, Typography, getRiskColor, getSafetyGrade, getGradeColor } from '../styles';
import Icon from './icons/Icon';

export default function RouteCard({ route, onSelect, isSelected }) {
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const scaleAnim = useRef(new Animated.Value(0.95)).current;

  useEffect(() => {
    // Fade in 애니메이션
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 200,
        useNativeDriver: true,
      }),
      Animated.spring(scaleAnim, {
        toValue: 1,
        tension: 50,
        friction: 7,
        useNativeDriver: true,
      }),
    ]).start();
  }, []);

  const getRiskLabel = (score) => {
    if (score <= 3) return '안전';
    if (score <= 6) return '보통';
    return '위험';
  };

  const getRouteTypeIcon = (type) => {
    const icons = {
      safe: 'safe',
      fast: 'fast',
      alternative: 'route',
    };
    return icons[type] || 'location';
  };

  const getRouteTypeLabel = (type) => {
    const labels = {
      safe: '안전',
      fast: '빠른',
      alternative: '대안',
    };
    return labels[type] || '일반';
  };

  const getRouteTypeBadgeColor = (type) => {
    const colors = {
      safe: Colors.success,
      fast: Colors.primary,
      alternative: Colors.info,
    };
    return colors[type] || Colors.textSecondary;
  };

  // 위험 구간 수 계산
  const getHazardZoneCount = (route) => {
    // 향후 개선: route.hazards 배열에서 실제 위험 구간 수를 계산
    // 현재는 risk_score를 기반으로 추정값 반환
    return route.hazards?.length || Math.floor(route.risk_score / 20);
  };

  const safetyGrade = getSafetyGrade(route.risk_score);
  const gradeColor = getGradeColor(safetyGrade);
  const hazardCount = getHazardZoneCount(route);

  return (
    <Animated.View
      style={{
        opacity: fadeAnim,
        transform: [{ scale: scaleAnim }],
      }}
    >
      <TouchableOpacity
        style={[styles.container, isSelected && styles.containerSelected]}
        onPress={() => onSelect(route)}
        activeOpacity={0.8}
      >
      <View style={styles.header}>
        <View style={styles.leftSection}>
          <View style={[styles.typeIconContainer, { backgroundColor: getRouteTypeBadgeColor(route.type) + '20' }]}>
            <Icon
              name={getRouteTypeIcon(route.type)}
              size={24}
              color={getRouteTypeBadgeColor(route.type)}
            />
          </View>
          <View style={styles.info}>
            <View style={styles.typeRow}>
              <Text style={styles.type}>
                {getRouteTypeLabel(route.type)} 경로
              </Text>
              {route.type === 'safe' && (
                <View style={[styles.typeBadge, { backgroundColor: Colors.success }]}>
                  <Text style={styles.typeBadgeText}>추천</Text>
                </View>
              )}
            </View>
            <View style={styles.modeContainer}>
              <Icon
                name={route.transportation_mode}
                size={16}
                color={Colors.textSecondary}
              />
              <Text style={styles.mode}>
                {route.transportation_mode === 'car' ? '차량' : route.transportation_mode === 'walking' ? '도보' : '자전거'}
              </Text>
            </View>
          </View>
        </View>

        {/* Phase 2: 안전도 등급 배지 */}
        <View style={[styles.gradeBadge, { backgroundColor: gradeColor + '20' }]}>
          <Text style={[styles.gradeText, { color: gradeColor }]}>{safetyGrade}</Text>
        </View>
      </View>

      <View style={styles.details}>
        <View style={styles.detailItem}>
          <Icon name="time" size={16} color={Colors.textSecondary} />
          <Text style={styles.detailText}>{route.duration}분</Text>
        </View>
        <View style={styles.detailItem}>
          <Icon name="distance" size={16} color={Colors.textSecondary} />
          <Text style={styles.detailText}>{route.distance.toFixed(1)}km</Text>
        </View>
        {/* Phase 2: 위험 구간 수 표시 */}
        <View style={styles.detailItem}>
          <Icon name="warning" size={16} color={hazardCount > 3 ? Colors.error : Colors.warning} />
          <Text style={[styles.detailText, { color: hazardCount > 3 ? Colors.error : Colors.warning }]}>
            위험 {hazardCount}개
          </Text>
        </View>
      </View>
      </TouchableOpacity>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: Colors.surfaceElevated,
    borderRadius: 16,  // 12 → 16
    padding: Spacing.lg,  // md → lg (12 → 16)
    marginBottom: Spacing.md,
    borderWidth: 2,
    borderColor: Colors.border,
    // 그림자 추가 (shadowMedium 적용)
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.12,
    shadowRadius: 8,
    elevation: 4,
  },
  containerSelected: {
    borderColor: Colors.primary,
    backgroundColor: Colors.primary + '10',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: Spacing.sm,
  },
  leftSection: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  typeIconContainer: {
    marginRight: Spacing.sm,
    width: 48,
    height: 48,
    borderRadius: 24,
    alignItems: 'center',
    justifyContent: 'center',
  },
  typeRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.xs,
  },
  typeBadge: {
    paddingHorizontal: Spacing.xs,
    paddingVertical: 2,
    borderRadius: 8,
  },
  typeBadgeText: {
    ...Typography.labelSmall,
    fontSize: 10,
    color: Colors.textInverse,
    fontWeight: '700',
  },
  info: {
    flex: 1,
  },
  modeContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.xs,
  },
  type: {
    ...Typography.h3,
    color: Colors.textPrimary,
    marginBottom: 2,
  },
  mode: {
    ...Typography.bodySmall,
    color: Colors.textSecondary,
  },
  // Phase 2: 안전도 등급 배지
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
  details: {
    flexDirection: 'row',
    marginTop: Spacing.sm,
    paddingTop: Spacing.sm,
    borderTopWidth: 1,
    borderTopColor: Colors.border,
  },
  detailItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.xs,
    marginRight: Spacing.md,
  },
  detailText: {
    ...Typography.bodySmall,
    color: Colors.textPrimary,
  },
});

