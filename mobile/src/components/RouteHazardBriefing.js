/**
 * RouteHazardBriefing - 경로 위험 정보 브리핑 시트 (드래그 가능)
 */

import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
  Animated,
  PanResponder,
  Dimensions,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Colors, Spacing, Typography } from '../styles';
import { routeAPI } from '../services/api';

const { height: SCREEN_HEIGHT } = Dimensions.get('window');
const MIN_SHEET_HEIGHT = 200; // 최소 높이
const MAX_SHEET_HEIGHT = SCREEN_HEIGHT * 0.8; // 최대 높이 (화면의 80%)

export default function RouteHazardBriefing({ route, isVisible, onClose }) {
  const insets = useSafeAreaInsets();
  const [hazards, setHazards] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // 드래그 애니메이션
  const panY = useRef(new Animated.Value(0)).current;
  const sheetHeight = useRef(MIN_SHEET_HEIGHT);

  useEffect(() => {
    if (isVisible && route && route.polyline) {
      loadHazards();
      // 시트 초기 위치 설정
      panY.setValue(0);
      sheetHeight.current = MIN_SHEET_HEIGHT;
    }
  }, [isVisible, route]);

  const panResponder = useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => true,
      onMoveShouldSetPanResponder: (_, gestureState) => {
        return Math.abs(gestureState.dy) > 5;
      },
      onPanResponderGrant: () => {
        panY.setOffset(panY._value);
      },
      onPanResponderMove: (_, gestureState) => {
        const newHeight = sheetHeight.current - gestureState.dy;
        const clampedHeight = Math.max(MIN_SHEET_HEIGHT, Math.min(MAX_SHEET_HEIGHT, newHeight));
        panY.setValue(-(sheetHeight.current - clampedHeight));
      },
      onPanResponderRelease: (_, gestureState) => {
        panY.flattenOffset();
        const newHeight = sheetHeight.current - gestureState.dy;
        
        // 스냅 처리
        if (gestureState.dy > 50) {
          // 아래로 드래그 - 닫기
          Animated.timing(panY, {
            toValue: MAX_SHEET_HEIGHT,
            duration: 200,
            useNativeDriver: true,
          }).start(() => {
            onClose();
            panY.setValue(0);
          });
        } else if (gestureState.dy < -50) {
          // 위로 드래그 - 최대 높이로
          sheetHeight.current = MAX_SHEET_HEIGHT;
          Animated.spring(panY, {
            toValue: -(MAX_SHEET_HEIGHT - MIN_SHEET_HEIGHT),
            useNativeDriver: true,
          }).start();
        } else {
          // 원래 위치로
          sheetHeight.current = MIN_SHEET_HEIGHT;
          Animated.spring(panY, {
            toValue: 0,
            useNativeDriver: true,
          }).start();
        }
      },
    })
  ).current;

  const loadHazards = async () => {
    if (!route || !route.polyline) return;

    setLoading(true);
    setError(null);

    try {
      const response = await routeAPI.getRouteHazards(route.id, route.polyline);
      setHazards(response.data);
    } catch (err) {
      console.error('Failed to load route hazards:', err);
      setError('위험 정보를 불러올 수 없습니다.');
    } finally {
      setLoading(false);
    }
  };

  const getHazardTypeLabel = (type) => {
    const labels = {
      'armed_conflict': '무력충돌',
      'protest_riot': '시위/폭동',
      'checkpoint': '검문소',
      'road_damage': '도로 손상',
      'natural_disaster': '자연재해',
      'other': '기타 위험',
      'crime': '범죄',
    };
    return labels[type] || type;
  };

  const getRiskLevelLabel = (score) => {
    if (score >= 70) return '매우 위험';
    if (score >= 50) return '위험';
    if (score >= 30) return '주의';
    return '안전';
  };

  if (!isVisible || !route) return null;

  const translateY = panY.interpolate({
    inputRange: [-(MAX_SHEET_HEIGHT - MIN_SHEET_HEIGHT), 0],
    outputRange: [-(MAX_SHEET_HEIGHT - MIN_SHEET_HEIGHT), 0],
    extrapolate: 'clamp',
  });

  return (
    <View style={styles.container}>
      <Animated.View
        style={[
          styles.sheet,
          {
            paddingBottom: insets.bottom,
            transform: [{ translateY }],
            maxHeight: MAX_SHEET_HEIGHT,
          },
        ]}
        {...panResponder.panHandlers}
      >
        <View style={styles.handleContainer}>
          <View style={styles.handle} />
        </View>
        
        <View style={styles.header}>
            <Text style={styles.title}>경로 위험 정보</Text>
            <TouchableOpacity onPress={onClose} style={styles.closeButton}>
              <Text style={styles.closeButtonText}>✕</Text>
            </TouchableOpacity>
          </View>

          {loading ? (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="large" color={Colors.primary} />
              <Text style={styles.loadingText}>위험 정보를 불러오는 중...</Text>
            </View>
          ) : error ? (
            <View style={styles.errorContainer}>
              <Text style={styles.errorText}>{error}</Text>
            </View>
          ) : hazards ? (
            <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
              {/* 요약 정보 */}
              <View style={styles.summaryCard}>
                <Text style={styles.summaryTitle}>📊 요약</Text>
                <View style={styles.summaryRow}>
                  <Text style={styles.summaryLabel}>총 위험 정보:</Text>
                  <Text style={styles.summaryValue}>
                    {hazards.summary?.total_hazards || 0}개
                  </Text>
                </View>
                {hazards.summary?.highest_risk_type && (
                  <View style={styles.summaryRow}>
                    <Text style={styles.summaryLabel}>가장 많은 위험:</Text>
                    <Text style={styles.summaryValue}>
                      {getHazardTypeLabel(hazards.summary.highest_risk_type)}
                    </Text>
                  </View>
                )}
              </View>

              {/* 위험 유형별 그룹화 */}
              {hazards.hazards_by_type && Object.keys(hazards.hazards_by_type).length > 0 && (
                <View style={styles.section}>
                  <Text style={styles.sectionTitle}>⚠️ 위험 유형별</Text>
                  {Object.entries(hazards.hazards_by_type).map(([type, items]) => (
                    <View key={type} style={styles.typeCard}>
                      <View style={styles.typeHeader}>
                        <Text style={styles.typeName}>
                          {getHazardTypeLabel(type)}
                        </Text>
                        <Text style={styles.typeCount}>{items.length}개</Text>
                      </View>
                    </View>
                  ))}
                </View>
              )}

              {/* 상세 위험 정보 */}
              {hazards.hazards && hazards.hazards.length > 0 && (
                <View style={styles.section}>
                  <Text style={styles.sectionTitle}>📍 상세 위치</Text>
                  {hazards.hazards.map((hazard, index) => (
                    <View key={index} style={styles.hazardCard}>
                      <View style={styles.hazardHeader}>
                        <Text style={styles.hazardType}>
                          {getHazardTypeLabel(hazard.hazard_type)}
                        </Text>
                        <View style={[
                          styles.riskBadge,
                          { backgroundColor: hazard.risk_score >= 70 ? Colors.error + '20' : Colors.warning + '20' }
                        ]}>
                          <Text style={[
                            styles.riskText,
                            { color: hazard.risk_score >= 70 ? Colors.error : Colors.warning }
                          ]}>
                            {getRiskLevelLabel(hazard.risk_score)}
                          </Text>
                        </View>
                      </View>
                      {hazard.description && (
                        <Text style={styles.hazardDescription}>
                          {hazard.description}
                        </Text>
                      )}
                      <Text style={styles.hazardDistance}>
                        경로로부터 {hazard.distance_from_route?.toFixed(0) || 0}m
                      </Text>
                    </View>
                  ))}
                </View>
              )}

              {(!hazards.hazards || hazards.hazards.length === 0) && (
                <View style={styles.emptyContainer}>
                  <Text style={styles.emptyIcon}>✅</Text>
                  <Text style={styles.emptyText}>이 경로는 안전합니다!</Text>
                  <Text style={styles.emptySubtext}>
                    경로 근방에 위험 정보가 없습니다.
                  </Text>
                </View>
              )}
            </ScrollView>
          ) : null}
        </Animated.View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    zIndex: 2000,
  },
  sheet: {
    backgroundColor: Colors.surfaceElevated,
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    minHeight: MIN_SHEET_HEIGHT,
    // 그림자 강화 (shadowLarge 적용)
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: -4 },
    shadowOpacity: 0.16,
    shadowRadius: 12,
    elevation: 8,
  },
  handleContainer: {
    paddingVertical: Spacing.sm,
    alignItems: 'center',
  },
  handle: {
    width: 40,
    height: 4,
    backgroundColor: Colors.border,
    borderRadius: 2,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: Spacing.lg,
    paddingBottom: Spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: Colors.border,
  },
  title: {
    ...Typography.h2,
    color: Colors.textPrimary,
  },
  closeButton: {
    width: 32,
    height: 32,
    justifyContent: 'center',
    alignItems: 'center',
  },
  closeButtonText: {
    fontSize: 24,
    color: Colors.textSecondary,
  },
  loadingContainer: {
    padding: Spacing.xl,
    alignItems: 'center',
  },
  loadingText: {
    ...Typography.body,
    color: Colors.textSecondary,
    marginTop: Spacing.md,
  },
  errorContainer: {
    padding: Spacing.xl,
    alignItems: 'center',
  },
  errorText: {
    ...Typography.body,
    color: Colors.error,
  },
  content: {
    paddingHorizontal: Spacing.lg,
    paddingTop: Spacing.lg,  // md → lg (여백 증가)
  },
  summaryCard: {
    backgroundColor: Colors.primary + '10',
    borderRadius: 16,  // 12 → 16
    padding: Spacing.lg,  // md → lg (여백 증가)
    marginBottom: Spacing.lg,
  },
  summaryTitle: {
    ...Typography.h3,
    color: Colors.textPrimary,
    marginBottom: Spacing.sm,
  },
  summaryRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: Spacing.xs,
  },
  summaryLabel: {
    ...Typography.body,
    color: Colors.textSecondary,
  },
  summaryValue: {
    ...Typography.body,
    color: Colors.textPrimary,
    fontWeight: '600',
  },
  section: {
    marginBottom: Spacing.lg,
  },
  sectionTitle: {
    ...Typography.h3,
    color: Colors.textPrimary,
    marginBottom: Spacing.md,
  },
  typeCard: {
    backgroundColor: Colors.surfaceElevated,
    borderRadius: 12,  // 8 → 12
    padding: Spacing.md,
    marginBottom: Spacing.sm,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  typeHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  typeName: {
    ...Typography.body,
    color: Colors.textPrimary,
    fontWeight: '500',
  },
  typeCount: {
    ...Typography.label,
    color: Colors.primary,
    fontWeight: '600',
  },
  hazardCard: {
    backgroundColor: Colors.surfaceElevated,
    borderRadius: 12,  // 8 → 12
    padding: Spacing.md,
    marginBottom: Spacing.sm,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  hazardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: Spacing.xs,
  },
  hazardType: {
    ...Typography.body,
    color: Colors.textPrimary,
    fontWeight: '600',
  },
  riskBadge: {
    paddingHorizontal: Spacing.sm,
    paddingVertical: 4,
    borderRadius: 12,
  },
  riskText: {
    ...Typography.labelSmall,
    fontWeight: '600',
  },
  hazardDescription: {
    ...Typography.bodySmall,
    color: Colors.textSecondary,
    marginBottom: Spacing.xs,
  },
  hazardDistance: {
    ...Typography.bodySmall,
    color: Colors.textTertiary,
  },
  emptyContainer: {
    padding: Spacing.xl,
    alignItems: 'center',
  },
  emptyIcon: {
    fontSize: 48,
    marginBottom: Spacing.md,
  },
  emptyText: {
    ...Typography.h3,
    color: Colors.textPrimary,
    marginBottom: Spacing.xs,
  },
  emptySubtext: {
    ...Typography.body,
    color: Colors.textSecondary,
    textAlign: 'center',
  },
});

