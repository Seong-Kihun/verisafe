/**
 * SafetyIndicator - 현재 위치 안전도 실시간 표시
 *
 * 카카오 네비의 실시간 교통 정보처럼
 * 현재 위치의 안전도를 색상과 텍스트로 명확히 전달
 */

import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Animated } from 'react-native';
import { Colors, Spacing, Typography } from '../styles';
import Icon from './icons/Icon';
import { mapAPI } from '../services/api';

export default function SafetyIndicator({ userLocation, onPress }) {
  const [safetyLevel, setSafetyLevel] = useState(null); // null, 'safe', 'caution', 'danger'
  const [safetyScore, setSafetyScore] = useState(0);
  const [nearbyHazards, setNearbyHazards] = useState([]);
  const [loading, setLoading] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false); // 확장/축소 상태
  const pulseAnim = useState(new Animated.Value(1))[0];

  useEffect(() => {
    if (userLocation) {
      checkSafety();
    }
  }, [userLocation?.latitude, userLocation?.longitude]);

  // 위험 상태일 때 펄스 애니메이션
  useEffect(() => {
    if (safetyLevel === 'danger') {
      Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, {
            toValue: 1.1,
            duration: 1000,
            useNativeDriver: true,
          }),
          Animated.timing(pulseAnim, {
            toValue: 1,
            duration: 1000,
            useNativeDriver: true,
          }),
        ])
      ).start();
    } else {
      pulseAnim.setValue(1);
    }
  }, [safetyLevel]);

  const checkSafety = async () => {
    if (!userLocation) return;

    setLoading(true);
    try {
      // 현재 위치 주변 1km 반경의 위험 정보 조회
      const radius = 0.01; // 약 1km
      const response = await mapAPI.getBounds(
        userLocation.latitude - radius,
        userLocation.longitude - radius,
        userLocation.latitude + radius,
        userLocation.longitude + radius
      );

      const hazards = response.data.hazards || [];
      setNearbyHazards(hazards);

      // 안전도 계산
      const score = calculateSafety(hazards, userLocation);
      setSafetyScore(score);

      // 안전도 레벨 결정
      if (score <= 30) {
        setSafetyLevel('safe');
      } else if (score <= 60) {
        setSafetyLevel('caution');
      } else {
        setSafetyLevel('danger');
      }
    } catch (error) {
      console.error('[SafetyIndicator] 안전도 확인 오류:', error);
      // 에러 시 중립 상태
      setSafetyLevel('safe');
      setSafetyScore(0);
    } finally {
      setLoading(false);
    }
  };

  const calculateSafety = (hazards, location) => {
    if (!hazards || hazards.length === 0) return 0;

    // 거리 기반 가중치 계산
    let totalWeightedRisk = 0;
    let totalWeight = 0;

    hazards.forEach((hazard) => {
      const distance = getDistance(
        location.latitude,
        location.longitude,
        hazard.latitude,
        hazard.longitude
      );

      // 거리에 반비례하는 가중치 (가까울수록 높음)
      const weight = Math.max(0, 1 - distance / 1); // 1km까지만 고려

      if (weight > 0) {
        totalWeightedRisk += hazard.risk_score * weight;
        totalWeight += weight;
      }
    });

    return totalWeight > 0 ? totalWeightedRisk / totalWeight : 0;
  };

  // 두 좌표 간 거리 계산 (km)
  const getDistance = (lat1, lon1, lat2, lon2) => {
    const R = 6371; // 지구 반경 (km)
    const dLat = ((lat2 - lat1) * Math.PI) / 180;
    const dLon = ((lon2 - lon1) * Math.PI) / 180;
    const a =
      Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos((lat1 * Math.PI) / 180) *
        Math.cos((lat2 * Math.PI) / 180) *
        Math.sin(dLon / 2) *
        Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
  };

  const getSafetyConfig = () => {
    switch (safetyLevel) {
      case 'safe':
        return {
          color: Colors.success,
          bgColor: `${Colors.success}15`,
          icon: 'shield',
          title: '현재 안전합니다',
          subtitle: '주변에 위험 요소가 없습니다',
        };
      case 'caution':
        return {
          color: Colors.warning,
          bgColor: `${Colors.warning}15`,
          icon: 'warning',
          title: '주의 - 위험 지역 근처',
          subtitle: `반경 1km 이내 위험 ${nearbyHazards.length}개`,
        };
      case 'danger':
        return {
          color: Colors.error,
          bgColor: `${Colors.error}15`,
          icon: 'warning',
          title: '경고 - 위험 지역 내',
          subtitle: `즉시 안전한 곳으로 이동하세요`,
        };
      default:
        return {
          color: Colors.textSecondary,
          bgColor: Colors.surface,
          icon: 'info',
          title: '위치 확인 중...',
          subtitle: '',
        };
    }
  };

  if (!userLocation || loading || !safetyLevel) {
    return null;
  }

  const config = getSafetyConfig();

  const handlePress = () => {
    setIsExpanded(!isExpanded);
    if (onPress && isExpanded) {
      onPress();
    }
  };

  // 축소된 상태 (작은 칩)
  if (!isExpanded) {
    return (
      <Animated.View style={{ transform: [{ scale: pulseAnim }] }}>
        <TouchableOpacity
          style={[styles.compactContainer, { backgroundColor: config.bgColor, borderColor: config.color }]}
          onPress={handlePress}
          activeOpacity={0.8}
          accessible={true}
          accessibilityRole="button"
          accessibilityLabel={`안전도: ${config.title}`}
          accessibilityHint="두 번 탭하여 상세 정보를 확인하세요"
          accessibilityState={{ expanded: false }}
        >
          <Icon name={config.icon} size={20} color={config.color} />
          <Text style={[styles.compactTitle, { color: config.color }]}>{config.title}</Text>
          <Icon name="chevron-down" size={16} color={config.color} />
        </TouchableOpacity>
      </Animated.View>
    );
  }

  // 확장된 상태 (전체 정보)
  return (
    <Animated.View style={{ transform: [{ scale: pulseAnim }] }}>
      <TouchableOpacity
        style={[styles.container, { backgroundColor: config.bgColor }]}
        onPress={handlePress}
        activeOpacity={0.8}
        accessible={true}
        accessibilityRole="button"
        accessibilityLabel={`안전도: ${config.title}. ${config.subtitle}`}
        accessibilityHint="두 번 탭하여 축소하세요"
        accessibilityState={{ expanded: true }}
      >
        <View style={styles.iconContainer}>
          <Icon name={config.icon} size={24} color={config.color} />
        </View>

        <View style={styles.textContainer}>
          <Text style={[styles.title, { color: config.color }]}>{config.title}</Text>
          <Text style={styles.subtitle}>{config.subtitle}</Text>
        </View>

        <Icon name="chevron-up" size={20} color={Colors.textSecondary} />
      </TouchableOpacity>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  // 확장된 상태
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: Spacing.md,
    borderRadius: 12,
    marginHorizontal: Spacing.md,
    marginTop: Spacing.md,
    shadowColor: Colors.shadowDark,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  iconContainer: {
    marginRight: Spacing.md,
  },
  textContainer: {
    flex: 1,
  },
  title: {
    ...Typography.labelMedium,
    fontWeight: '600',
    marginBottom: Spacing.xs,
  },
  subtitle: {
    ...Typography.bodySmall,
    color: Colors.textSecondary,
  },
  // 축소된 상태 (컴팩트 칩)
  compactContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.sm,
    borderRadius: 20,
    marginHorizontal: Spacing.md,
    marginTop: Spacing.md,
    borderWidth: 1.5,
    shadowColor: Colors.shadowDark,
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
    gap: Spacing.xs,
  },
  compactTitle: {
    ...Typography.labelSmall,
    fontWeight: '600',
    flex: 1,
  },
});
