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
import { useMapContext } from '../contexts/MapContext';
import { useRoutePlanningContext } from '../contexts/RoutePlanningContext';
import { useNavigation } from '@react-navigation/native';

const { height: SCREEN_HEIGHT } = Dimensions.get('window');
const MIN_SHEET_HEIGHT = SCREEN_HEIGHT * 0.3; // 최소 높이 (화면의 30%)
const MID_SHEET_HEIGHT = SCREEN_HEIGHT * 0.6; // 중간 높이 (화면의 60%)
const MAX_SHEET_HEIGHT = SCREEN_HEIGHT * 0.85; // 최대 높이 (화면의 85%)

export default function RouteHazardBriefing({ route, isVisible, onClose }) {
  const navigation = useNavigation();
  const { openPlaceSheet } = useMapContext();
  const { closeHazardBriefing, setShouldReopenBriefing } = useRoutePlanningContext();
  const insets = useSafeAreaInsets();
  const [hazards, setHazards] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // 드래그 애니메이션
  const panY = useRef(new Animated.Value(MAX_SHEET_HEIGHT - MIN_SHEET_HEIGHT)).current;
  const sheetHeight = useRef(MIN_SHEET_HEIGHT);

  useEffect(() => {
    if (isVisible && route && route.polyline) {
      loadHazards();
      // 시트 초기 위치 설정 (30%만 보이도록)
      panY.setValue(MAX_SHEET_HEIGHT - MIN_SHEET_HEIGHT);
      sheetHeight.current = MIN_SHEET_HEIGHT;
    }
  }, [isVisible, route]);

  const panResponder = useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => true,
      onMoveShouldSetPanResponder: (_, gestureState) => {
        // 세로 드래그가 가로 드래그보다 클 때만 반응
        return Math.abs(gestureState.dy) > Math.abs(gestureState.dx) && Math.abs(gestureState.dy) > 3;
      },
      onPanResponderGrant: () => {
        // 드래그 시작 시 현재 값을 offset으로 설정
        panY.setOffset(panY._value);
        panY.setValue(0);
      },
      onPanResponderMove: Animated.event(
        [null, { dy: panY }],
        {
          useNativeDriver: false,
          listener: (_, gestureState) => {
            // 범위 제한 적용
            const newValue = panY._offset + gestureState.dy;
            if (newValue < 0) {
              // 위로 너무 많이 드래그 시 저항 추가 (rubber band effect)
              panY.setValue(newValue * 0.4);
            } else if (newValue > MAX_SHEET_HEIGHT - MIN_SHEET_HEIGHT) {
              // 아래로 너무 많이 드래그 시 저항 추가
              const excess = newValue - (MAX_SHEET_HEIGHT - MIN_SHEET_HEIGHT);
              panY.setValue((MAX_SHEET_HEIGHT - MIN_SHEET_HEIGHT) + excess * 0.4);
            }
          }
        }
      ),
      onPanResponderRelease: (_, gestureState) => {
        panY.flattenOffset();
        const currentTranslateY = panY._value;

        // 현재 보이는 시트 높이 계산
        const currentVisibleHeight = MAX_SHEET_HEIGHT - currentTranslateY;

        // 가장 가까운 스냅 포인트 찾기
        let targetHeight;
        const snapPoints = [MIN_SHEET_HEIGHT, MID_SHEET_HEIGHT, MAX_SHEET_HEIGHT];

        // 빠른 스와이프 감지 (velocity 기반)
        const isQuickSwipe = Math.abs(gestureState.vy) > 0.8;

        if (isQuickSwipe) {
          // 빠른 스와이프: 방향에 따라 다음/이전 스냅 포인트로
          if (gestureState.dy > 0) {
            // 아래로 스와이프
            if (gestureState.dy > 150 && gestureState.vy > 1.2) {
              // 강하게 아래로 - 닫기
              Animated.timing(panY, {
                toValue: SCREEN_HEIGHT,
                duration: 280,
                useNativeDriver: true,
              }).start(() => {
                onClose();
                panY.setValue(MAX_SHEET_HEIGHT - MIN_SHEET_HEIGHT);
                sheetHeight.current = MIN_SHEET_HEIGHT;
              });
              return;
            } else {
              // 한 단계 아래로 (높이 감소)
              targetHeight = snapPoints.reverse().find(h => h < currentVisibleHeight) || MIN_SHEET_HEIGHT;
              snapPoints.reverse(); // 원래대로
            }
          } else {
            // 위로 스와이프 - 한 단계 위로 (높이 증가)
            targetHeight = snapPoints.find(h => h > currentVisibleHeight) || MAX_SHEET_HEIGHT;
          }
        } else {
          // 천천히 드래그: 가장 가까운 스냅 포인트로
          targetHeight = snapPoints.reduce((prev, curr) => {
            return Math.abs(curr - currentVisibleHeight) < Math.abs(prev - currentVisibleHeight) ? curr : prev;
          });
        }

        // 스냅 애니메이션 - 더 부드러운 스프링 효과
        sheetHeight.current = targetHeight;
        const targetTranslateY = MAX_SHEET_HEIGHT - targetHeight;
        Animated.spring(panY, {
          toValue: targetTranslateY,
          velocity: gestureState.vy,  // 드래그 속도를 애니메이션에 반영
          tension: 68,  // 80 → 68 (더 부드럽게)
          friction: 14, // 12 → 14 (바운스 감소)
          useNativeDriver: true,
        }).start();
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

  const handleHazardPress = (hazard) => {
    // 1. 경로 위험 정보 모달 닫기
    closeHazardBriefing();

    // 2. 재오픈 플래그 설정
    setShouldReopenBriefing(true);

    // 3. 지도 화면으로 이동 (이미 지도 화면이면 유지)
    navigation.navigate('MapStack', { screen: 'MapMain' });

    // 4. 위험 위치 정보로 PlaceSheet 열기
    openPlaceSheet({
      id: hazard.id,
      name: getHazardTypeLabel(hazard.hazard_type),
      address: hazard.description || '',
      latitude: hazard.latitude,
      longitude: hazard.longitude,
      category: 'danger',
      description: hazard.description,
      risk_score: hazard.risk_score,
      hazard_type: hazard.hazard_type,
      type: 'hazard',
    });
  };

  if (!isVisible || !route) return null;

  const translateY = panY.interpolate({
    inputRange: [0, MAX_SHEET_HEIGHT - MIN_SHEET_HEIGHT],
    outputRange: [0, MAX_SHEET_HEIGHT - MIN_SHEET_HEIGHT],
    extrapolate: 'clamp',
  });

  return (
    <View style={styles.container}>
      <Animated.View
        style={[
          styles.sheet,
          {
            height: MAX_SHEET_HEIGHT, // 전체 높이
            paddingBottom: insets.bottom,
            transform: [{ translateY }],
          },
        ]}
      >
        <View style={styles.handleContainer} {...panResponder.panHandlers}>
          <View style={styles.handle} />
        </View>

        <View style={styles.header} {...panResponder.panHandlers}>
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
            <ScrollView
              style={styles.content}
              showsVerticalScrollIndicator={false}
            >
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
                    <TouchableOpacity
                      key={index}
                      style={styles.hazardCard}
                      onPress={() => handleHazardPress(hazard)}
                      activeOpacity={0.7}
                    >
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
                    </TouchableOpacity>
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
    // 그림자 강화 (shadowLarge 적용)
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: -4 },
    shadowOpacity: 0.16,
    shadowRadius: 12,
    elevation: 8,
  },
  handleContainer: {
    paddingVertical: Spacing.md,
    alignItems: 'center',
    cursor: 'grab',
  },
  handle: {
    width: 48,
    height: 5,
    backgroundColor: Colors.textTertiary,
    borderRadius: 3,
    opacity: 0.5,
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
    flex: 1,
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

