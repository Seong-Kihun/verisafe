/**
 * RouteCarousel - 경로 목록 카드 캐러셀
 * 지도탭에서 여러 경로를 수평 스크롤로 확인할 수 있는 컴포넌트
 */

import React, { useRef, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Dimensions,
} from 'react-native';
import { Colors, Typography, Spacing } from '../styles';
import Icon from './icons/Icon';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const CARD_WIDTH = SCREEN_WIDTH * 0.75;
const CARD_SPACING = Spacing.md;

export default function RouteCarousel({
  routes,
  selectedRoute,
  onSelectRoute,
  onStartNavigation,
  onShowDetail,
}) {
  const scrollViewRef = useRef(null);

  // 선택된 경로가 변경되면 해당 카드로 스크롤
  useEffect(() => {
    if (selectedRoute && routes.length > 0) {
      const selectedIndex = routes.findIndex(r => r.id === selectedRoute.id);
      if (selectedIndex >= 0 && scrollViewRef.current) {
        scrollViewRef.current.scrollTo({
          x: selectedIndex * (CARD_WIDTH + CARD_SPACING),
          animated: true,
        });
      }
    }
  }, [selectedRoute, routes]);

  if (!routes || routes.length === 0) {
    return null;
  }

  return (
    <View style={styles.container}>
      <ScrollView
        ref={scrollViewRef}
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.scrollContent}
        snapToInterval={CARD_WIDTH + CARD_SPACING}
        decelerationRate="fast"
        snapToAlignment="start"
      >
        {routes.map((route, index) => {
          const isSelected = selectedRoute?.id === route.id;

          return (
            <TouchableOpacity
              key={route.id || index}
              style={[
                styles.card,
                isSelected && styles.cardSelected,
              ]}
              onPress={() => onSelectRoute(route)}
              activeOpacity={0.8}
              accessible={true}
              accessibilityRole="button"
              accessibilityLabel={`경로 ${index + 1}, 거리 ${typeof route.distance === 'number' ? route.distance.toFixed(1) : route.distance}킬로미터, 소요시간 ${formatDuration(route.duration)}, 위험도 ${route.risk_score}점, 위험 구간 ${route.hazard_count || 0}개`}
              accessibilityHint={isSelected ? "현재 선택된 경로입니다" : "두 번 탭하여 이 경로를 선택하세요"}
              accessibilityState={{ selected: isSelected }}
            >
              {/* 경로 번호 뱃지 */}
              <View style={styles.cardHeader}>
                <View style={[styles.routeBadge, isSelected && styles.routeBadgeSelected]}>
                  <Text style={[styles.routeBadgeText, isSelected && styles.routeBadgeTextSelected]}>
                    경로 {index + 1}
                  </Text>
                </View>
                {isSelected && (
                  <View style={styles.selectedIndicator}>
                    <Icon name="check" size={16} color={Colors.textInverse} />
                  </View>
                )}
              </View>

              {/* 경로 정보 */}
              <View style={styles.routeInfo}>
                {/* 거리 & 시간 */}
                <View style={styles.infoRow}>
                  <View style={styles.infoItem}>
                    <Icon name="locationOn" size={18} color={Colors.primary} />
                    <Text style={styles.infoValue}>
                      {typeof route.distance === 'number'
                        ? `${route.distance.toFixed(1)}km`
                        : route.distance}
                    </Text>
                  </View>
                  <View style={styles.infoDivider} />
                  <View style={styles.infoItem}>
                    <Icon name="time" size={18} color={Colors.primary} />
                    <Text style={styles.infoValue}>
                      {formatDuration(route.duration)}
                    </Text>
                  </View>
                </View>

                {/* 위험도 */}
                <View style={styles.riskRow}>
                  <View style={styles.riskIndicator}>
                    <Icon name="warning" size={16} color={getRiskColor(route.risk_score)} />
                    <Text style={[styles.riskText, { color: getRiskColor(route.risk_score) }]}>
                      위험도 {route.risk_score}/10
                    </Text>
                  </View>
                  <Text style={styles.hazardCount}>
                    위험 구간 {route.hazard_count || 0}개
                  </Text>
                </View>
              </View>

              {/* 버튼 영역 (모든 카드에 표시) */}
              <View style={styles.buttonRow}>
                {/* 상세 정보 버튼 */}
                {onShowDetail && (
                  <TouchableOpacity
                    style={styles.detailButton}
                    onPress={() => {
                      onSelectRoute(route);
                      onShowDetail(route);
                    }}
                    activeOpacity={0.8}
                    accessible={true}
                    accessibilityRole="button"
                    accessibilityLabel={`경로 ${index + 1} 상세 정보 보기`}
                    accessibilityHint="두 번 탭하여 경로의 상세 위험 정보를 확인하세요"
                  >
                    <Icon name="info" size={18} color={Colors.primary} />
                    <Text style={styles.detailButtonText}>상세 정보</Text>
                  </TouchableOpacity>
                )}

                {/* 안내 시작 버튼 */}
                {onStartNavigation && (
                  <TouchableOpacity
                    style={styles.navButton}
                    onPress={() => {
                      onSelectRoute(route);
                      onStartNavigation(route);
                    }}
                    activeOpacity={0.8}
                    accessible={true}
                    accessibilityRole="button"
                    accessibilityLabel={`경로 ${index + 1} 안내 시작`}
                    accessibilityHint="두 번 탭하여 내비게이션을 시작하세요"
                  >
                    <Icon name="navigation" size={20} color={Colors.textInverse} />
                    <Text style={styles.navButtonText}>안내 시작</Text>
                  </TouchableOpacity>
                )}
              </View>
            </TouchableOpacity>
          );
        })}
      </ScrollView>
    </View>
  );
}

/**
 * 소요 시간 포맷팅 (분 -> "X분" or "X시간 Y분")
 */
const formatDuration = (minutes) => {
  if (typeof minutes !== 'number') return minutes;
  if (minutes < 60) return `${minutes}분`;
  const hours = Math.floor(minutes / 60);
  const mins = minutes % 60;
  return mins > 0 ? `${hours}시간 ${mins}분` : `${hours}시간`;
};

/**
 * 위험도에 따른 색상 반환
 */
const getRiskColor = (riskScore) => {
  if (riskScore <= 3) return Colors.success;
  if (riskScore <= 6) return Colors.warning;
  return Colors.danger;
};

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    paddingBottom: Spacing.md,
    zIndex: 1100, // FloatingActionButton(1000)보다 높게 설정
  },
  scrollContent: {
    paddingHorizontal: (SCREEN_WIDTH - CARD_WIDTH) / 2,
    paddingVertical: Spacing.sm,
  },
  card: {
    width: CARD_WIDTH,
    backgroundColor: Colors.surface,
    borderRadius: 16,
    padding: Spacing.md,
    marginRight: CARD_SPACING,
    // Shadow
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.15,
    shadowRadius: 12,
    elevation: 8,
    borderWidth: 2,
    borderColor: 'transparent',
  },
  cardSelected: {
    borderColor: Colors.primary,
    backgroundColor: Colors.surface,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: Spacing.sm,
  },
  routeBadge: {
    backgroundColor: Colors.border,
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.xs,
    borderRadius: 12,
  },
  routeBadgeSelected: {
    backgroundColor: Colors.primary,
  },
  routeBadgeText: {
    ...Typography.labelSmall,
    color: Colors.textSecondary,
    fontWeight: '600',
  },
  routeBadgeTextSelected: {
    color: Colors.textInverse,
  },
  selectedIndicator: {
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: Colors.success,
    justifyContent: 'center',
    alignItems: 'center',
  },
  routeInfo: {
    marginBottom: Spacing.sm,
  },
  infoRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: Spacing.sm,
  },
  infoItem: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.xs,
  },
  infoDivider: {
    width: 1,
    height: 20,
    backgroundColor: Colors.border,
    marginHorizontal: Spacing.sm,
  },
  infoValue: {
    ...Typography.body,
    color: Colors.textPrimary,
    fontWeight: '600',
  },
  riskRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  riskIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.xs,
  },
  riskText: {
    ...Typography.body,
    fontWeight: '600',
  },
  hazardCount: {
    ...Typography.bodySmall,
    color: Colors.textSecondary,
  },
  buttonRow: {
    flexDirection: 'row',
    gap: Spacing.sm,
    marginTop: Spacing.sm,
  },
  detailButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: Colors.surface,
    paddingVertical: Spacing.md,
    borderRadius: 12,
    borderWidth: 1.5,
    borderColor: Colors.primary,
    gap: Spacing.xs,
  },
  detailButtonText: {
    ...Typography.body,
    color: Colors.primary,
    fontWeight: '600',
  },
  navButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: Colors.primary,
    paddingVertical: Spacing.md,
    borderRadius: 12,
    gap: Spacing.xs,
  },
  navButtonText: {
    ...Typography.body,
    color: Colors.textInverse,
    fontWeight: '700',
  },
});
