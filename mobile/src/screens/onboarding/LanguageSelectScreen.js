/**
 * 온보딩 - 언어 선택 화면
 */

import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import i18n from '../../i18n';
import { Colors, Typography, Spacing } from '../../styles';
import { useOnboarding } from '../../contexts/OnboardingContext';
import Icon from '../../components/icons/Icon';

const LANGUAGES = [
  { code: 'ko', name: '한국어 (Korean)', flag: '🇰🇷' },
  { code: 'en', name: 'English', flag: '🇺🇸' },
  { code: 'es', name: 'Español (Spanish)', flag: '🇪🇸' },
  { code: 'fr', name: 'Français (French)', flag: '🇫🇷' },
  { code: 'pt', name: 'Português (Portuguese)', flag: '🇵🇹' },
  { code: 'sw', name: 'Kiswahili (Swahili)', flag: '🇹🇿' },
];

export default function LanguageSelectScreen({ navigation }) {
  const { onboardingData, updateOnboardingData } = useOnboarding();
  const [selectedLanguage, setSelectedLanguage] = useState(onboardingData.language || 'ko');

  const handleSelectLanguage = async (languageCode) => {
    setSelectedLanguage(languageCode);
    updateOnboardingData('language', languageCode);
    // 즉시 언어 변경 (미리보기)
    await i18n.changeLanguage(languageCode);
  };

  const handleNext = () => {
    navigation.navigate('CountrySelect');
  };

  const handleSkip = () => {
    navigation.navigate('CountrySelect');
  };

  return (
    <SafeAreaView style={styles.safeArea} edges={['top', 'bottom']}>
      <View style={styles.container}>
        {/* 뒤로가기 버튼 */}
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => navigation.goBack()}
          activeOpacity={0.7}
        >
          <Icon name="arrowBack" size={24} color={Colors.textPrimary} />
        </TouchableOpacity>

        {/* 헤더 */}
        <View style={styles.header}>
          <Text style={styles.title}>언어를 선택해주세요</Text>
          <Text style={styles.subtitle}>
            VeriSafe를 사용할 언어를 선택하세요
          </Text>
        </View>

        {/* 언어 목록 */}
        <ScrollView
          style={styles.scrollView}
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          {LANGUAGES.map((lang) => (
            <TouchableOpacity
              key={lang.code}
              style={[
                styles.languageOption,
                selectedLanguage === lang.code && styles.languageOptionSelected,
              ]}
              onPress={() => handleSelectLanguage(lang.code)}
              activeOpacity={0.7}
            >
              <Text style={styles.languageFlag}>{lang.flag}</Text>
              <Text style={[
                styles.languageName,
                selectedLanguage === lang.code && styles.languageNameSelected,
              ]}>
                {lang.name}
              </Text>
              {selectedLanguage === lang.code && (
                <View style={styles.checkIcon}>
                  <Icon name="check" size={24} color={Colors.primary} />
                </View>
              )}
            </TouchableOpacity>
          ))}
        </ScrollView>

        {/* 하단 버튼 */}
        <View style={styles.footer}>
          <TouchableOpacity
            style={styles.skipButton}
            onPress={handleSkip}
            activeOpacity={0.7}
          >
            <Text style={styles.skipButtonText}>건너뛰기</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.nextButton}
            onPress={handleNext}
            activeOpacity={0.8}
          >
            <Text style={styles.nextButtonText}>다음</Text>
            <Icon name="arrowForward" size={20} color={Colors.textInverse} />
          </TouchableOpacity>
        </View>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  container: {
    flex: 1,
  },
  backButton: {
    position: 'absolute',
    top: Spacing.md,
    left: Spacing.lg,
    zIndex: 10,
    padding: Spacing.sm,
  },
  header: {
    paddingHorizontal: Spacing.xl,
    paddingTop: Spacing.xxxl + Spacing.xl,
    paddingBottom: Spacing.xl,
  },
  title: {
    ...Typography.h1,
    color: Colors.textPrimary,
    marginBottom: Spacing.sm,
  },
  subtitle: {
    ...Typography.body,
    color: Colors.textSecondary,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    paddingHorizontal: Spacing.xl,
    paddingBottom: Spacing.xl,
  },
  languageOption: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: Spacing.lg,
    borderRadius: 12,
    backgroundColor: Colors.surface,
    borderWidth: 2,
    borderColor: 'transparent',
    marginBottom: Spacing.md,
  },
  languageOptionSelected: {
    backgroundColor: Colors.primary + '10',
    borderColor: Colors.primary,
  },
  languageFlag: {
    fontSize: 32,
    marginRight: Spacing.md,
  },
  languageName: {
    ...Typography.bodyLarge,
    color: Colors.textPrimary,
    flex: 1,
  },
  languageNameSelected: {
    color: Colors.primary,
    fontWeight: '600',
  },
  checkIcon: {
    marginLeft: Spacing.sm,
  },
  footer: {
    flexDirection: 'row',
    paddingHorizontal: Spacing.xl,
    paddingVertical: Spacing.lg,
    gap: Spacing.md,
    borderTopWidth: 1,
    borderTopColor: Colors.borderLight,
  },
  skipButton: {
    flex: 1,
    paddingVertical: Spacing.lg,
    borderRadius: 12,
    backgroundColor: Colors.surface,
    alignItems: 'center',
    justifyContent: 'center',
  },
  skipButtonText: {
    ...Typography.button,
    color: Colors.textSecondary,
  },
  nextButton: {
    flex: 2,
    flexDirection: 'row',
    paddingVertical: Spacing.lg,
    borderRadius: 12,
    backgroundColor: Colors.primary,
    alignItems: 'center',
    justifyContent: 'center',
    gap: Spacing.sm,
    shadowColor: Colors.shadowMedium,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 4,
    elevation: 2,
  },
  nextButtonText: {
    ...Typography.button,
    color: Colors.textInverse,
  },
});
