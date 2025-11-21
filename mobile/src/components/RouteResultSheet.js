/**
 * RouteResultSheet - 경로 결과 하단 시트 (Phase 2 개선)
 *
 * 책임:
 * 1. 안전/빠른 경로 탭 표시
 * 2. 각 경로의 거리, 시간, 위험도 표시
 * 3. 큰 "안내 시작" 버튼 (Google Maps 스타일)
 * 4. ETA (도착 예정 시간) 표시
 * 5. 안전도 등급 (A~F) 표시
 * 6. 위험 구간 수 표시
 */

import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  Alert,
  Share,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import { Colors, Spacing, Typography } from '../styles';
import { useMapContext } from '../contexts/MapContext';
import { useNavigation as useNavContext } from '../contexts/NavigationContext';
import { routeAPI } from '../services/api';
import Icon from './icons/Icon';
import { useRouteHazardCount } from '../hooks/useRouteHazards';

export default function RouteResultSheet() {
  const insets = useSafeAreaInsets();
  const { routeResponse, closeRouteSheet } = useMapContext();
  const { startNavigation } = useNavContext();
  const navigation = useNavigation();
  const [selectedMode, setSelectedMode] = useState('safe');
  const [routeHazards, setRouteHazards] = useState({});

  // 경로 위험 정보 가져오기
  useEffect(() => {
    const fetchRouteHazards = async () => {
      if (!routeResponse || !routeResponse.routes) return;

      const hazardsData = {};
      for (const route of routeResponse.routes) {
        try {
          const response = await routeAPI.getRouteHazards(route.id, route.polyline);
          hazardsData[route.id] = response.data.hazards || [];
        } catch (error) {
          console.error(`[RouteResultSheet] Failed to fetch hazards for ${route.id}:`, error);
          hazardsData[route.id] = [];
        }
      }
      setRouteHazards(hazardsData);
    };

    fetchRouteHazards();
  }, [routeResponse]);

  if (!routeResponse || !routeResponse.routes || routeResponse.routes.length === 0) {
    return null;
  }

  const safeRoute = routeResponse.routes.find(r => r.type === 'safe');
  const fastRoute = routeResponse.routes.find(r => r.type === 'fast');

  // 경로가 1개만 있는지 확인
  const hasOnlyOneRoute = routeResponse.routes.length === 1;
  const singleRoute = hasOnlyOneRoute ? routeResponse.routes[0] : null;

  const getRiskColor = (score) => {
    if (score < 25) return '#4CAF50'; // green
    if (score < 50) return '#FF9800'; // orange
    return '#F44336'; // red
  };

  // 안전도 등급 계산 (A~F)
  const getSafetyGrade = (riskScore) => {
    if (riskScore <= 2) return 'A';
    if (riskScore <= 4) return 'B';
    if (riskScore <= 6) return 'C';
    if (riskScore <= 8) return 'D';
    if (riskScore <= 9) return 'E';
    return 'F';
  };

  // 안전도 등급 색상
  const getGradeColor = (grade) => {
    if (grade === 'A' || grade === 'B') return Colors.success;
    if (grade === 'C') return Colors.warning;
    return Colors.error;
  };

  // ETA 계산 (현재 시간 + duration)
  const getETA = (durationMinutes) => {
    const now = new Date();
    const eta = new Date(now.getTime() + durationMinutes * 60000);
    const hours = eta.getHours();
    const minutes = eta.getMinutes();
    const period = hours < 12 ? '오전' : '오후';
    const displayHours = hours % 12 || 12;
    return `${period} ${displayHours}:${minutes.toString().padStart(2, '0')}`;
  };

  // 위험 구간 수 계산 (실제 API 데이터 사용)
  // Note: 이 로직은 useRouteHazardCount hook과 동일하지만,
  // routeHazards 상태를 사용하기 때문에 함수로 유지
  const getHazardZoneCount = (route) => {
    const hazards = routeHazards[route.id];
    // useRouteHazardCount와 동일한 로직 사용
    if (hazards && Array.isArray(hazards) && hazards.length > 0) {
      return hazards.length;
    }
    return route?.hazard_count || 0;
  };

  // 실제 위험도 계산 (hazards 기반)
  const getActualRiskScore = (route) => {
    const hazards = routeHazards[route.id];
    if (!hazards || hazards.length === 0) {
      return route.risk_score || 0;
    }

    // 위험 수준에 따른 가중치
    const severityWeights = {
      'critical': 3,
      'high': 2,
      'medium': 1,
      'low': 0.5
    };

    let totalWeight = 0;
    hazards.forEach(hazard => {
      const weight = severityWeights[hazard.severity] || 1;
      totalWeight += weight;
    });

    // 0-10 스케일로 정규화 (hazards 수와 심각도를 고려)
    const normalized = Math.min(10, Math.round(totalWeight / 2));
    return normalized;
  };

  const handleStartNavigation = async () => {
    const currentRoute = selectedMode === 'safe' ? safeRoute : fastRoute;
    if (currentRoute) {
      try {
        // 경로에 위험 정보 추가
        const routeWithHazards = {
          ...currentRoute,
          hazards: routeHazards[currentRoute.id] || []
        };

        // 네비게이션 시작 (위치 추적, 음성 안내 등 초기화)
        await startNavigation(routeWithHazards);

        // 네비게이션 화면으로 이동
        navigation.navigate('NavigationScreen');

        // 경로 결과 시트 닫기
        closeRouteSheet();
      } catch (error) {
        console.error('[RouteResultSheet] Navigation start failed:', error);
        Alert.alert(
          '오류',
          '네비게이션을 시작할 수 없습니다.\n위치 권한을 확인해주세요.',
          [{ text: '확인' }]
        );
      }
    }
  };

  // Phase 3: 경로 공유 기능
  const handleShareRoute = async () => {
    const currentRoute = selectedMode === 'safe' ? safeRoute : fastRoute;
    if (!currentRoute) return;

    const actualRiskScore = getActualRiskScore(currentRoute);
    const hazardCount = getHazardZoneCount(currentRoute);
    const safetyGrade = getSafetyGrade(actualRiskScore, hazardCount);
    const eta = getETA(currentRoute.duration);

    const shareText = `🗺️ VeriSafe 안전 경로

📍 경로 정보
• 소요 시간: ${currentRoute.duration}분
• 도착 시간: ${eta}
• 거리: ${currentRoute.distance.toFixed(1)}km
• 안전도 등급: ${safetyGrade}
• 위험 구간: ${hazardCount}개

🛡️ ${currentRoute.type === 'safe' ? '가장 안전한 경로' : '가장 빠른 경로'}입니다.

VeriSafe로 안전하게 이동하세요!`;

    try {
      await Share.share({
        message: shareText,
      });
    } catch (error) {
      console.error('[RouteResultSheet] 공유 오류:', error);
    }
  };

  const currentRoute = hasOnlyOneRoute ? singleRoute : (selectedMode === 'safe' ? safeRoute : fastRoute);

  if (!currentRoute) {
    return null;
  }

  // 실제 위험도 사용
  const actualRiskScore = getActualRiskScore(currentRoute);
  const hazardCount = getHazardZoneCount(currentRoute);
  const safetyGrade = getSafetyGrade(actualRiskScore, hazardCount);
  const gradeColor = getGradeColor(safetyGrade);
  const eta = getETA(currentRoute.duration);

  return (
    <View style={[styles.container, { paddingBottom: Spacing.md }]}>
      <View style={styles.sheetHandle} />

      {/* 모드 탭 또는 단일 경로 안내 */}
      {hasOnlyOneRoute ? (
        <View style={styles.singleRouteNotice}>
          <Icon name="checkCircle" size={20} color={Colors.success} />
          <Text style={styles.singleRouteText}>
            이 지역에서 가장 안전한 경로를 찾았습니다
          </Text>
        </View>
      ) : (
        <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.modeTabs}>
          {safeRoute && (
            <TouchableOpacity
              style={[styles.modeTab, selectedMode === 'safe' && styles.modeTabActive]}
              onPress={() => setSelectedMode('safe')}
            >
              <Icon name="safe" size={20} color={selectedMode === 'safe' ? Colors.primary : Colors.textSecondary} />
              <Text style={[styles.modeLabel, selectedMode === 'safe' && styles.modeLabelActive]}>안전</Text>
              <Text style={[styles.modeTime, selectedMode === 'safe' && styles.modeTimeActive]}>{safeRoute.duration}분</Text>
            </TouchableOpacity>
          )}

          {fastRoute && (
            <TouchableOpacity
              style={[styles.modeTab, selectedMode === 'fast' && styles.modeTabActive]}
              onPress={() => setSelectedMode('fast')}
            >
              <Icon name="fast" size={20} color={selectedMode === 'fast' ? Colors.primary : Colors.textSecondary} />
              <Text style={[styles.modeLabel, selectedMode === 'fast' && styles.modeLabelActive]}>빠름</Text>
              <Text style={[styles.modeTime, selectedMode === 'fast' && styles.modeTimeActive]}>{fastRoute.duration}분</Text>
            </TouchableOpacity>
          )}
        </ScrollView>
      )}

      {/* 경로 상세 정보 */}
      <View style={styles.routeDetails}>
        {/* 시간 + ETA (Google Maps 스타일) */}
        <View style={styles.timeSection}>
          <Text style={styles.duration}>{currentRoute.duration}분</Text>
          <Text style={styles.eta}>{eta} 도착</Text>
        </View>

        {/* 안전도 등급 + 거리 + 위험 구간 수 */}
        <View style={styles.infoGrid}>
          <View style={styles.infoItem}>
            <View style={[styles.gradeBadge, { backgroundColor: gradeColor + '20' }]}>
              <Text style={[styles.gradeText, { color: gradeColor }]}>{safetyGrade}</Text>
            </View>
            <Text style={styles.infoLabel}>안전도</Text>
          </View>

          <View style={styles.infoItem}>
            <Icon name="distance" size={24} color={Colors.primary} />
            <Text style={styles.infoValue}>{currentRoute.distance.toFixed(1)}km</Text>
            <Text style={styles.infoLabel}>거리</Text>
          </View>

          <View style={styles.infoItem}>
            <Icon name="warning" size={24} color={hazardCount > 3 ? Colors.error : Colors.warning} />
            <Text style={styles.infoValue}>{hazardCount}개</Text>
            <Text style={styles.infoLabel}>위험 구간</Text>
          </View>
        </View>

        {/* 경로 타입 배지 */}
        <View style={styles.routeTypeBadge}>
          <Icon
            name={currentRoute.type === 'safe' ? 'safe' : 'fast'}
            size={16}
            color={Colors.primary}
          />
          <Text style={styles.routeTypeText}>
            {currentRoute.type === 'safe' ? '가장 안전한 경로' : '가장 빠른 경로'}
          </Text>
        </View>
      </View>

      {/* 큰 "안내 시작" 버튼 + 공유 버튼 (Phase 3) */}
      <View style={styles.actionButtonsContainer}>
        <TouchableOpacity
          style={styles.startButton}
          onPress={handleStartNavigation}
          activeOpacity={0.8}
        >
          <Icon name="navigation" size={24} color={Colors.textInverse} />
          <Text style={styles.startButtonText}>안내 시작</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.shareButton}
          onPress={handleShareRoute}
          activeOpacity={0.8}
        >
          <Icon name="share" size={24} color={Colors.primary} />
        </TouchableOpacity>
      </View>

      {/* 닫기 버튼 */}
      <TouchableOpacity style={styles.closeButton} onPress={closeRouteSheet}>
        <Text style={styles.closeButtonText}>닫기</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    bottom: 70, // 탭바 높이만큼 위로
    left: 0,
    right: 0,
    backgroundColor: Colors.surface,
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    shadowColor: Colors.shadowDark,
    shadowOffset: { width: 0, height: -2 },
    shadowOpacity: 0.2,
    shadowRadius: 8,
    elevation: 5,
    maxHeight: '70%',
  },
  sheetHandle: {
    width: 40,
    height: 4,
    backgroundColor: Colors.border,
    borderRadius: 2,
    alignSelf: 'center',
    marginTop: Spacing.sm,
    marginBottom: Spacing.md,
  },
  modeTabs: {
    paddingHorizontal: Spacing.lg,
    marginBottom: Spacing.lg,
  },
  modeTab: {
    alignItems: 'center',
    paddingVertical: Spacing.md,
    paddingHorizontal: Spacing.lg,
    marginRight: Spacing.md,
    borderRadius: 16,
    backgroundColor: Colors.borderLight,
    borderWidth: 2,
    borderColor: 'transparent',
    gap: Spacing.xs,
  },
  modeTabActive: {
    backgroundColor: Colors.primary + '15',
    borderColor: Colors.primary,
  },
  modeLabel: {
    ...Typography.labelSmall,
    color: Colors.textSecondary,
    fontWeight: '600',
  },
  modeLabelActive: {
    color: Colors.primary,
  },
  modeTime: {
    ...Typography.h3,
    color: Colors.textSecondary,
    fontSize: 16,
  },
  modeTimeActive: {
    color: Colors.primary,
  },
  routeDetails: {
    paddingHorizontal: Spacing.lg,
    marginBottom: Spacing.lg,
  },
  // 시간 + ETA 섹션 (Google Maps 스타일)
  timeSection: {
    marginBottom: Spacing.lg,
  },
  duration: {
    ...Typography.h1,
    fontSize: 36,
    color: Colors.textPrimary,
    fontWeight: '700',
    marginBottom: Spacing.xs,
  },
  eta: {
    ...Typography.body,
    color: Colors.textSecondary,
    fontSize: 16,
  },
  // 정보 그리드 (안전도, 거리, 위험 구간)
  infoGrid: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingVertical: Spacing.md,
    backgroundColor: Colors.background,
    borderRadius: 16,
    marginBottom: Spacing.md,
  },
  infoItem: {
    alignItems: 'center',
    gap: Spacing.xs,
  },
  gradeBadge: {
    width: 48,
    height: 48,
    borderRadius: 24,
    alignItems: 'center',
    justifyContent: 'center',
  },
  gradeText: {
    ...Typography.h2,
    fontSize: 24,
    fontWeight: '700',
  },
  infoValue: {
    ...Typography.h3,
    color: Colors.textPrimary,
    fontWeight: '600',
  },
  infoLabel: {
    ...Typography.labelSmall,
    color: Colors.textSecondary,
  },
  // 경로 타입 배지
  routeTypeBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: Spacing.xs,
    backgroundColor: Colors.primary + '10',
    paddingVertical: Spacing.sm,
    paddingHorizontal: Spacing.md,
    borderRadius: 20,
    alignSelf: 'center',
  },
  routeTypeText: {
    ...Typography.labelSmall,
    color: Colors.primary,
    fontWeight: '600',
  },
  // Phase 3: 액션 버튼 컨테이너
  actionButtonsContainer: {
    flexDirection: 'row',
    marginHorizontal: Spacing.lg,
    marginBottom: Spacing.md,
    gap: Spacing.sm,
  },
  // 큰 "안내 시작" 버튼 (Google Maps 스타일)
  startButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    flex: 1,
    paddingVertical: Spacing.lg,
    backgroundColor: Colors.primary,
    borderRadius: 16,
    gap: Spacing.sm,
    shadowColor: Colors.primary,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 6,
  },
  // Phase 3: 공유 버튼
  shareButton: {
    width: 56,
    height: 56,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: Colors.surface,
    borderRadius: 16,
    borderWidth: 2,
    borderColor: Colors.primary,
  },
  startButtonText: {
    ...Typography.button,
    fontSize: 18,
    color: Colors.textInverse,
    fontWeight: '700',
  },
  closeButton: {
    marginHorizontal: Spacing.lg,
    marginBottom: Spacing.md,
    paddingVertical: Spacing.md,
    backgroundColor: Colors.borderLight,
    borderRadius: 12,
    alignItems: 'center',
  },
  closeButtonText: {
    ...Typography.button,
    color: Colors.textPrimary,
  },
  // 단일 경로 안내 메시지
  singleRouteNotice: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
    paddingVertical: Spacing.md,
    paddingHorizontal: Spacing.lg,
    backgroundColor: Colors.success + '10',
    marginHorizontal: Spacing.lg,
    marginTop: Spacing.md,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: Colors.success + '30',
  },
  singleRouteText: {
    ...Typography.body,
    color: Colors.success,
    fontWeight: '600',
    flex: 1,
  },
});

