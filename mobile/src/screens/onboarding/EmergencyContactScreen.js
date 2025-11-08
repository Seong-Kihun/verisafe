/**
 * 온보딩 - 긴급 연락처 설정 화면 (선택)
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
  Alert,
} from 'react-native';
import { Colors, Typography, Spacing } from '../../styles';
import { emergencyContactsStorage } from '../../services/storage';
import Icon from '../../components/icons/Icon';

export default function EmergencyContactScreen({ navigation }) {
  const [name, setName] = useState('');
  const [phone, setPhone] = useState('');
  const [adding, setAdding] = useState(false);

  const validatePhone = (phoneNumber) => {
    // 간단한 전화번호 검증 (숫자, +, -, 공백 허용)
    return /^[\d\s\-+()]+$/.test(phoneNumber) && phoneNumber.replace(/\D/g, '').length >= 8;
  };

  const handleAddContact = async () => {
    if (!name.trim()) {
      Alert.alert('알림', '이름을 입력해주세요.');
      return;
    }

    if (!phone.trim()) {
      Alert.alert('알림', '전화번호를 입력해주세요.');
      return;
    }

    if (!validatePhone(phone.trim())) {
      Alert.alert('알림', '올바른 전화번호를 입력해주세요.\n예: +211-XXX-XXXX');
      return;
    }

    setAdding(true);
    try {
      const newContact = await emergencyContactsStorage.add({
        name: name.trim(),
        phone: phone.trim(),
        relationship: 'other',
        shareLocation: true,
      });

      if (newContact) {
        Alert.alert(
          '등록 완료',
          `${name.trim()}님이 긴급 연락처로 등록되었습니다.`,
          [
            {
              text: '확인',
              onPress: () => navigation.navigate('Complete'),
            },
          ]
        );
      } else {
        Alert.alert('오류', '긴급 연락처 등록에 실패했습니다.');
      }
    } catch (error) {
      console.error('Failed to add emergency contact:', error);
      if (error.message.includes('Maximum')) {
        Alert.alert('알림', '긴급 연락처는 최대 5명까지 등록 가능합니다.');
      } else {
        Alert.alert('오류', '긴급 연락처 등록에 실패했습니다.');
      }
    } finally {
      setAdding(false);
    }
  };

  const handleSkip = () => {
    navigation.navigate('Complete');
  };

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
    >
      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
        keyboardShouldPersistTaps="handled"
      >
        {/* 헤더 */}
        <View style={styles.header}>
          <View style={styles.iconContainer}>
            <Icon name="emergency" size={48} color={Colors.danger} />
          </View>
          <Text style={styles.title}>긴급 연락처 설정</Text>
          <Text style={styles.subtitle}>
            위급 상황 시 SOS 버튼을 누르면 등록된 연락처로 알림이 전송됩니다
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
            <Text style={styles.label}>전화번호</Text>
            <TextInput
              style={styles.input}
              placeholder="예: +211-XXX-XXXX"
              placeholderTextColor={Colors.textTertiary}
              value={phone}
              onChangeText={setPhone}
              keyboardType="phone-pad"
            />
          </View>

          {/* 안내 문구 */}
          <View style={styles.infoBox}>
            <Icon name="info" size={20} color={Colors.warning} />
            <View style={styles.infoTextContainer}>
              <Text style={styles.infoText}>
                SOS 버튼을 누르면:
              </Text>
              <Text style={styles.infoText}>
                • 현재 위치가 SMS로 전송됩니다
              </Text>
              <Text style={styles.infoText}>
                • 추가로 최대 4명까지 등록 가능합니다
              </Text>
            </View>
          </View>

          {/* 추가 버튼 */}
          <TouchableOpacity
            style={styles.addButton}
            onPress={handleAddContact}
            activeOpacity={0.8}
            disabled={adding}
          >
            <Icon name="add" size={24} color={Colors.textInverse} />
            <Text style={styles.addButtonText}>
              {adding ? '등록 중...' : '긴급 연락처 등록'}
            </Text>
          </TouchableOpacity>
        </View>
      </ScrollView>

      {/* 하단 버튼 */}
      <View style={styles.footer}>
        <TouchableOpacity
          style={styles.skipButton}
          onPress={handleSkip}
          activeOpacity={0.7}
        >
          <Text style={styles.skipButtonText}>나중에 설정</Text>
        </TouchableOpacity>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
  },
  header: {
    paddingHorizontal: Spacing.xl,
    paddingTop: Spacing.xxxl,
    paddingBottom: Spacing.xl,
    alignItems: 'center',
  },
  iconContainer: {
    width: 96,
    height: 96,
    borderRadius: 48,
    backgroundColor: Colors.danger + '10',
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
    lineHeight: 22,
  },
  formContainer: {
    paddingHorizontal: Spacing.xl,
  },
  inputGroup: {
    marginBottom: Spacing.lg,
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
    backgroundColor: Colors.warning + '10',
    borderRadius: 12,
    padding: Spacing.lg,
    gap: Spacing.md,
    marginTop: Spacing.md,
    marginBottom: Spacing.xl,
    alignItems: 'flex-start',
  },
  infoTextContainer: {
    flex: 1,
  },
  infoText: {
    ...Typography.bodySmall,
    color: Colors.warning,
    lineHeight: 20,
    marginBottom: Spacing.xs / 2,
  },
  addButton: {
    flexDirection: 'row',
    backgroundColor: Colors.danger,
    paddingVertical: Spacing.lg,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    gap: Spacing.sm,
    shadowColor: Colors.shadowMedium,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 4,
    elevation: 2,
  },
  addButtonText: {
    ...Typography.button,
    color: Colors.textInverse,
  },
  footer: {
    paddingHorizontal: Spacing.xl,
    paddingVertical: Spacing.lg,
    borderTopWidth: 1,
    borderTopColor: Colors.borderLight,
  },
  skipButton: {
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
});
