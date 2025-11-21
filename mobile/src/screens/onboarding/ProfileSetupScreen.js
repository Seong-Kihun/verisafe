/**
 * 온보딩 - 프로필 설정 화면 (선택)
 */

import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  TextInput,
  ScrollView,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Colors, Typography, Spacing } from '../../styles';
import { useOnboarding } from '../../contexts/OnboardingContext';
import Icon from '../../components/icons/Icon';

export default function ProfileSetupScreen({ navigation }) {
  const { onboardingData, updateOnboardingData } = useOnboarding();
  const [name, setName] = useState(onboardingData.profile?.name || '');
  const [organization, setOrganization] = useState(onboardingData.profile?.organization || '');

  const handleNext = () => {
    updateOnboardingData('profile', { name, organization });
    navigation.navigate('Permission');
  };

  const handleSkip = () => {
    navigation.navigate('Permission');
  };

  return (
    <SafeAreaView style={styles.safeArea} edges={['top', 'bottom']}>
      <KeyboardAvoidingView
        style={styles.container}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        keyboardVerticalOffset={0}
      >
        {/* 뒤로가기 버튼 */}
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => navigation.goBack()}
          activeOpacity={0.7}
        >
          <Icon name="arrowBack" size={24} color={Colors.textPrimary} />
        </TouchableOpacity>

      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
        keyboardShouldPersistTaps="handled"
      >
        {/* 헤더 */}
        <View style={styles.header}>
          <View style={styles.iconContainer}>
            <Icon name="person" size={48} color={Colors.primary} />
          </View>
          <Text style={styles.title}>프로필을 설정해주세요</Text>
          <Text style={styles.subtitle}>
            나중에 언제든지 변경할 수 있습니다 (선택 사항)
          </Text>
        </View>

        {/* 입력 필드 */}
        <View style={styles.formContainer}>
          <View style={styles.inputGroup}>
            <Text style={styles.label}>이름</Text>
            <TextInput
              style={styles.input}
              placeholder="예: 홍길동"
              placeholderTextColor={Colors.textTertiary}
              value={name}
              onChangeText={setName}
              autoCapitalize="words"
            />
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>소속 조직 (선택)</Text>
            <TextInput
              style={styles.input}
              placeholder="예: KOICA, NGO 단체명 등"
              placeholderTextColor={Colors.textTertiary}
              value={organization}
              onChangeText={setOrganization}
              autoCapitalize="words"
            />
          </View>

          {/* 안내 문구 */}
          <View style={styles.infoBox}>
            <Icon name="info" size={20} color={Colors.info} />
            <Text style={styles.infoText}>
              입력하신 정보는 앱 내에서만 사용되며, 외부로 공유되지 않습니다.
            </Text>
          </View>
        </View>
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
      </KeyboardAvoidingView>
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
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
  },
  header: {
    paddingHorizontal: Spacing.xl,
    paddingTop: Spacing.xxxl + Spacing.xl,
    paddingBottom: Spacing.xl,
    alignItems: 'center',
  },
  iconContainer: {
    width: 96,
    height: 96,
    borderRadius: 48,
    backgroundColor: Colors.primary + '10',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: Spacing.lg,
  },
  title: {
    ...Typography.h1,
    color: Colors.textPrimary,
    marginBottom: Spacing.sm,
    textAlign: 'center',
  },
  subtitle: {
    ...Typography.body,
    color: Colors.textSecondary,
    textAlign: 'center',
  },
  formContainer: {
    paddingHorizontal: Spacing.xl,
  },
  inputGroup: {
    marginBottom: Spacing.xl,
  },
  label: {
    ...Typography.label,
    color: Colors.textPrimary,
    marginBottom: Spacing.sm,
  },
  input: {
    ...Typography.input,
    color: Colors.textPrimary,
    backgroundColor: Colors.surface,
    borderWidth: 1,
    borderColor: Colors.border,
    borderRadius: 12,
    paddingHorizontal: Spacing.lg,
    paddingVertical: Spacing.md,
  },
  infoBox: {
    flexDirection: 'row',
    backgroundColor: Colors.info + '10',
    borderRadius: 12,
    padding: Spacing.lg,
    gap: Spacing.md,
    marginTop: Spacing.lg,
  },
  infoText: {
    ...Typography.bodySmall,
    color: Colors.info,
    flex: 1,
    lineHeight: 20,
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
