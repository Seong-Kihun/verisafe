/**
 * 온보딩 - 완료 화면
 */

import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
} from 'react-native';
import { Colors, Typography, Spacing } from '../../styles';
import { useOnboarding } from '../../contexts/OnboardingContext';
import Icon from '../../components/icons/Icon';

export default function CompleteScreen({ navigation }) {
  const { onboardingData, completeOnboarding } = useOnboarding();
  const [loading, setLoading] = useState(false);

  const handleComplete = async () => {
    setLoading(true);
    try {
      const success = await completeOnboarding();
      if (success) {
        // 온보딩 완료 후 자동으로 메인 앱으로 이동
        // (AppNavigator에서 처리됨)
      }
    } catch (error) {
      console.error('Failed to complete onboarding:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      {/* 메인 콘텐츠 */}
      <View style={styles.content}>
        {/* 성공 아이콘 */}
        <View style={styles.iconContainer}>
          <View style={styles.successCircle}>
            <Icon name="check" size={80} color={Colors.success} />
          </View>
          <View style={styles.celebrationContainer}>
            <Text style={styles.celebrationEmoji}>🎉</Text>
            <Text style={styles.celebrationEmoji}>🎊</Text>
          </View>
        </View>

        {/* 제목 */}
        <Text style={styles.title}>설정 완료!</Text>
        <Text style={styles.subtitle}>
          이제 VeriSafe를 사용할 준비가 되었습니다
        </Text>

        {/* 설정 요약 */}
        <View style={styles.summaryContainer}>
          <SummaryItem
            icon="language"
            label="언어"
            value={getLanguageName(onboardingData.language)}
          />
          {onboardingData.country && (
            <SummaryItem
              icon="map"
              label="활동 국가"
              value={`${onboardingData.country.flag} ${onboardingData.country.name.split('(')[0].trim()}`}
            />
          )}
          {onboardingData.profile?.name && (
            <SummaryItem
              icon="person"
              label="이름"
              value={onboardingData.profile.name}
            />
          )}
        </View>

        {/* 안내 문구 */}
        <View style={styles.tipsContainer}>
          <Text style={styles.tipsTitle}>💡 시작 팁</Text>
          <Text style={styles.tipItem}>
            • 지도 탭에서 주변 위험 정보를 확인하세요
          </Text>
          <Text style={styles.tipItem}>
            • 경로 계획 시 안전 경로를 우선 선택하세요
          </Text>
          <Text style={styles.tipItem}>
            • 위험 상황을 발견하면 즉시 제보해주세요
          </Text>
          <Text style={styles.tipItem}>
            • 긴급 상황 시 SOS 버튼을 길게 눌러주세요
          </Text>
        </View>
      </View>

      {/* 시작 버튼 */}
      <View style={styles.footer}>
        <TouchableOpacity
          style={styles.startButton}
          onPress={handleComplete}
          activeOpacity={0.8}
          disabled={loading}
        >
          {loading ? (
            <ActivityIndicator size="small" color={Colors.textInverse} />
          ) : (
            <>
              <Text style={styles.startButtonText}>VeriSafe 시작하기</Text>
              <Icon name="arrowForward" size={24} color={Colors.textInverse} />
            </>
          )}
        </TouchableOpacity>
      </View>
    </View>
  );
}

// 요약 항목 컴포넌트
const SummaryItem = ({ icon, label, value }) => (
  <View style={styles.summaryItem}>
    <View style={styles.summaryIconContainer}>
      <Icon name={icon} size={20} color={Colors.primary} />
    </View>
    <View style={styles.summaryTextContainer}>
      <Text style={styles.summaryLabel}>{label}</Text>
      <Text style={styles.summaryValue}>{value}</Text>
    </View>
  </View>
);

// 언어 이름 변환
const getLanguageName = (code) => {
  const languageMap = {
    ko: '한국어',
    en: 'English',
    es: 'Español',
    fr: 'Français',
    pt: 'Português',
    sw: 'Kiswahili',
  };
  return languageMap[code] || code;
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  content: {
    flex: 1,
    paddingHorizontal: Spacing.xl,
    paddingTop: Spacing.xxxl,
    alignItems: 'center',
  },
  iconContainer: {
    position: 'relative',
    marginBottom: Spacing.xl,
  },
  successCircle: {
    width: 160,
    height: 160,
    borderRadius: 80,
    backgroundColor: Colors.success + '20',
    borderWidth: 4,
    borderColor: Colors.success,
    alignItems: 'center',
    justifyContent: 'center',
  },
  celebrationContainer: {
    position: 'absolute',
    top: -20,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: Spacing.md,
  },
  celebrationEmoji: {
    fontSize: 32,
  },
  title: {
    ...Typography.display,
    color: Colors.textPrimary,
    marginBottom: Spacing.sm,
  },
  subtitle: {
    ...Typography.bodyLarge,
    color: Colors.textSecondary,
    textAlign: 'center',
    marginBottom: Spacing.xxxl,
  },
  summaryContainer: {
    width: '100%',
    backgroundColor: Colors.surface,
    borderRadius: 16,
    padding: Spacing.lg,
    marginBottom: Spacing.xl,
  },
  summaryItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: Spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: Colors.borderLight,
  },
  summaryIconContainer: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: Colors.primary + '10',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: Spacing.md,
  },
  summaryTextContainer: {
    flex: 1,
  },
  summaryLabel: {
    ...Typography.caption,
    color: Colors.textSecondary,
    marginBottom: Spacing.xs / 2,
  },
  summaryValue: {
    ...Typography.bodyLarge,
    color: Colors.textPrimary,
    fontWeight: '600',
  },
  tipsContainer: {
    width: '100%',
    backgroundColor: Colors.info + '10',
    borderRadius: 16,
    padding: Spacing.lg,
  },
  tipsTitle: {
    ...Typography.h3,
    color: Colors.info,
    marginBottom: Spacing.md,
  },
  tipItem: {
    ...Typography.body,
    color: Colors.info,
    lineHeight: 24,
    marginBottom: Spacing.sm,
  },
  footer: {
    paddingHorizontal: Spacing.xl,
    paddingVertical: Spacing.lg,
    borderTopWidth: 1,
    borderTopColor: Colors.borderLight,
  },
  startButton: {
    flexDirection: 'row',
    backgroundColor: Colors.primary,
    paddingVertical: Spacing.lg,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    gap: Spacing.sm,
    shadowColor: Colors.shadowMedium,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 4,
  },
  startButtonText: {
    ...Typography.button,
    color: Colors.textInverse,
    fontSize: 18,
  },
});
