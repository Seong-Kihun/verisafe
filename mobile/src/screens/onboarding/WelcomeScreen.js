/**
 * 온보딩 - 환영 화면
 */

import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Image,
  ScrollView,
} from 'react-native';
import { Colors, Typography, Spacing } from '../../styles';
import Icon from '../../components/icons/Icon';

export default function WelcomeScreen({ navigation }) {
  const handleGetStarted = () => {
    navigation.navigate('LanguageSelect');
  };

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.content}
      showsVerticalScrollIndicator={false}
    >
      {/* 로고/아이콘 영역 */}
      <View style={styles.logoContainer}>
        <View style={styles.logoCircle}>
          <Icon name="shield" size={80} color={Colors.primary} />
        </View>
        <Text style={styles.appName}>VeriSafe</Text>
        <Text style={styles.tagline}>Your Safety Companion</Text>
      </View>

      {/* 기능 소개 */}
      <View style={styles.featuresContainer}>
        <Feature
          icon="map"
          title="안전한 경로 안내"
          description="실시간 위험 정보를 반영한 최적 경로 제공"
        />
        <Feature
          icon="warning"
          title="위험 정보 제보"
          description="현장의 위험 상황을 신속하게 공유"
        />
        <Feature
          icon="article"
          title="실시간 뉴스"
          description="지역별 안전 정보와 뉴스 확인"
        />
        <Feature
          icon="emergency"
          title="긴급 SOS"
          description="위급 상황 시 긴급 연락처로 즉시 알림"
        />
      </View>

      {/* 시작 버튼 */}
      <TouchableOpacity
        style={styles.startButton}
        onPress={handleGetStarted}
        activeOpacity={0.8}
      >
        <Text style={styles.startButtonText}>시작하기</Text>
        <Icon name="arrowForward" size={24} color={Colors.textInverse} />
      </TouchableOpacity>

      {/* 하단 텍스트 */}
      <Text style={styles.footerText}>
        KOICA와 함께하는 안전한 여정
      </Text>
    </ScrollView>
  );
}

// 기능 소개 컴포넌트
const Feature = ({ icon, title, description }) => (
  <View style={styles.featureItem}>
    <View style={styles.featureIconContainer}>
      <Icon name={icon} size={28} color={Colors.primary} />
    </View>
    <View style={styles.featureTextContainer}>
      <Text style={styles.featureTitle}>{title}</Text>
      <Text style={styles.featureDescription}>{description}</Text>
    </View>
  </View>
);

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  content: {
    flexGrow: 1,
    paddingHorizontal: Spacing.xl,
    paddingTop: Spacing.xxxl,
    paddingBottom: Spacing.xl,
  },
  logoContainer: {
    alignItems: 'center',
    marginBottom: Spacing.xxxl,
  },
  logoCircle: {
    width: 140,
    height: 140,
    borderRadius: 70,
    backgroundColor: Colors.primary + '10',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: Spacing.lg,
  },
  appName: {
    ...Typography.display,
    color: Colors.primary,
    marginBottom: Spacing.xs,
  },
  tagline: {
    ...Typography.body,
    color: Colors.textSecondary,
  },
  featuresContainer: {
    marginBottom: Spacing.xxxl,
  },
  featureItem: {
    flexDirection: 'row',
    marginBottom: Spacing.xl,
    alignItems: 'flex-start',
  },
  featureIconContainer: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: Colors.primary + '10',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: Spacing.lg,
  },
  featureTextContainer: {
    flex: 1,
    paddingTop: Spacing.xs,
  },
  featureTitle: {
    ...Typography.h3,
    color: Colors.textPrimary,
    marginBottom: Spacing.xs,
  },
  featureDescription: {
    ...Typography.body,
    color: Colors.textSecondary,
    lineHeight: 22,
  },
  startButton: {
    flexDirection: 'row',
    backgroundColor: Colors.primary,
    paddingVertical: Spacing.lg,
    paddingHorizontal: Spacing.xl,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    gap: Spacing.sm,
    shadowColor: Colors.shadowMedium,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 4,
    marginBottom: Spacing.xl,
  },
  startButtonText: {
    ...Typography.button,
    color: Colors.textInverse,
  },
  footerText: {
    ...Typography.caption,
    color: Colors.textTertiary,
    textAlign: 'center',
  },
});
