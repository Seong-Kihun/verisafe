/**
 * 긴급 연락처 편집 화면
 */

import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  ScrollView,
  Alert,
  ActivityIndicator,
  Switch,
} from 'react-native';
import { Colors, Typography, Spacing } from '../styles';
import { emergencyContactsStorage } from '../services/storage';
import Icon from '../components/icons/Icon';

export default function EmergencyContactEditScreen({ navigation, route }) {
  // Type guard for navigation parameters
  const validateContactParam = (contact) => {
    if (!contact) return null;

    // Ensure required fields exist
    if (typeof contact.id !== 'string' && typeof contact.id !== 'number') {
      console.warn('[EmergencyContactEdit] Invalid contact parameter: missing id');
      return null;
    }

    return contact;
  };

  const editingContact = validateContactParam(route?.params?.contact);
  const isEditing = !!editingContact;

  const [saving, setSaving] = useState(false);
  const [name, setName] = useState('');
  const [phone, setPhone] = useState('');
  const [email, setEmail] = useState('');
  const [relationship, setRelationship] = useState('other');
  const [shareLocation, setShareLocation] = useState(true);

  useEffect(() => {
    if (editingContact) {
      setName(editingContact.name || '');
      setPhone(editingContact.phone || '');
      setEmail(editingContact.email || '');
      setRelationship(editingContact.relationship || 'other');
      setShareLocation(editingContact.shareLocation !== undefined ? editingContact.shareLocation : true);
    }
  }, [editingContact]);

  const validatePhone = (phone) => {
    // Remove all spaces, dashes, parentheses for validation
    const cleaned = phone.replace(/[\s\-\(\)]/g, '');

    // International format with country code: +XXX followed by 6-15 digits
    // Supports: +211XXXXXXXXX (South Sudan), +1XXXXXXXXXX (US), +44XXXXXXXXXX (UK), etc.
    const internationalRegex = /^\+[1-9]\d{6,14}$/;

    // Local format without country code: 6-15 digits, optionally starting with 0
    const localRegex = /^0?\d{6,14}$/;

    return internationalRegex.test(cleaned) || localRegex.test(cleaned);
  };

  const validateEmail = (email) => {
    if (!email) return true; // Email is optional
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  };

  const handleSave = async () => {
    if (!name.trim()) {
      Alert.alert('알림', '이름을 입력해주세요.');
      return;
    }

    if (!phone.trim()) {
      Alert.alert('알림', '전화번호를 입력해주세요.');
      return;
    }

    if (!validatePhone(phone.trim())) {
      Alert.alert('알림', '올바른 전화번호 형식을 입력해주세요.\n예시:\n• +211 XXX XXX XXX (South Sudan)\n• +1 XXX XXX XXXX (USA)\n• 0912345678 (로컬 번호)');
      return;
    }

    if (email.trim() && !validateEmail(email.trim())) {
      Alert.alert('알림', '올바른 이메일 형식을 입력해주세요.\n예: example@email.com');
      return;
    }

    setSaving(true);
    try {
      const contactData = {
        name: name.trim(),
        phone: phone.trim(),
        email: email.trim(),
        relationship,
        shareLocation,
      };

      let success;
      if (isEditing) {
        success = await emergencyContactsStorage.update(editingContact.id, contactData);
      } else {
        const result = await emergencyContactsStorage.add(contactData);
        success = !!result;
      }

      if (success) {
        Alert.alert('성공', isEditing ? '연락처가 수정되었습니다.' : '연락처가 추가되었습니다.', [
          { text: '확인', onPress: () => navigation.goBack() },
        ]);
      } else {
        Alert.alert('오류', '저장에 실패했습니다.');
      }
    } catch (error) {
      console.error('[EmergencyContactEdit] Failed to save contact:', error);
      if (error.message === 'Maximum 5 emergency contacts allowed') {
        Alert.alert('제한 초과', '긴급 연락처는 최대 5명까지만 등록할 수 있습니다.');
      } else {
        Alert.alert('오류', '저장에 실패했습니다.');
      }
    } finally {
      setSaving(false);
    }
  };

  const relationshipOptions = [
    { value: 'family', label: '가족', icon: 'people', color: Colors.primary },
    { value: 'friend', label: '친구', icon: 'person', color: Colors.success },
    { value: 'colleague', label: '동료', icon: 'work', color: Colors.info },
    { value: 'other', label: '기타', icon: 'contactPhone', color: Colors.textSecondary },
  ];

  return (
    <View style={styles.container}>
      <ScrollView style={styles.scrollView}>
        <View style={styles.section}>
          <Text style={styles.label}>이름 *</Text>
          <TextInput
            style={styles.input}
            value={name}
            onChangeText={setName}
            placeholder="이름을 입력하세요"
            placeholderTextColor={Colors.textTertiary}
          />
        </View>

        <View style={styles.section}>
          <Text style={styles.label}>전화번호 *</Text>
          <TextInput
            style={styles.input}
            value={phone}
            onChangeText={setPhone}
            placeholder="+211 XXX XXX XXX 또는 로컬 번호"
            placeholderTextColor={Colors.textTertiary}
            keyboardType="phone-pad"
          />
        </View>

        <View style={styles.section}>
          <Text style={styles.label}>이메일 (선택)</Text>
          <TextInput
            style={styles.input}
            value={email}
            onChangeText={setEmail}
            placeholder="email@example.com"
            placeholderTextColor={Colors.textTertiary}
            keyboardType="email-address"
            autoCapitalize="none"
          />
        </View>

        <View style={styles.section}>
          <Text style={styles.label}>관계</Text>
          <View style={styles.relationshipGrid}>
            {relationshipOptions.map((option) => (
              <TouchableOpacity
                key={option.value}
                style={[
                  styles.relationshipOption,
                  relationship === option.value && styles.relationshipOptionActive,
                ]}
                onPress={() => setRelationship(option.value)}
              >
                <Icon
                  name={option.icon}
                  size={24}
                  color={relationship === option.value ? option.color : Colors.textTertiary}
                />
                <Text style={[
                  styles.relationshipLabel,
                  relationship === option.value && { color: option.color, fontWeight: '600' },
                ]}>
                  {option.label}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>

        <View style={styles.section}>
          <View style={styles.switchRow}>
            <View style={styles.switchLeft}>
              <Icon name="locationOn" size={24} color={shareLocation ? Colors.success : Colors.textTertiary} />
              <View>
                <Text style={styles.switchLabel}>위치 공유</Text>
                <Text style={styles.switchDescription}>
                  SOS 발동 시 실시간 위치 공유
                </Text>
              </View>
            </View>
            <Switch
              value={shareLocation}
              onValueChange={setShareLocation}
              trackColor={{ false: Colors.border, true: Colors.success + '60' }}
              thumbColor={shareLocation ? Colors.success : Colors.textTertiary}
            />
          </View>
        </View>

        <View style={styles.infoSection}>
          <Icon name="info" size={20} color={Colors.info} />
          <Text style={styles.infoText}>
            긴급 상황 발생 시 이 연락처로 자동으로 알림이 전송됩니다.
            {'\n'}우선순위는 등록 순서대로 자동 지정됩니다.
          </Text>
        </View>
      </ScrollView>

      <View style={styles.footer}>
        <TouchableOpacity
          style={styles.cancelButton}
          onPress={() => navigation.goBack()}
        >
          <Text style={styles.cancelButtonText}>취소</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.saveButton}
          onPress={handleSave}
          disabled={saving}
        >
          {saving ? (
            <ActivityIndicator color={Colors.textInverse} />
          ) : (
            <Text style={styles.saveButtonText}>
              {isEditing ? '수정' : '추가'}
            </Text>
          )}
        </TouchableOpacity>
      </View>
    </View>
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
  section: {
    padding: Spacing.lg,
    borderBottomWidth: 1,
    borderBottomColor: Colors.borderLight,
  },
  label: {
    ...Typography.label,
    color: Colors.textPrimary,
    marginBottom: Spacing.sm,
  },
  input: {
    backgroundColor: Colors.surface,
    borderRadius: 12,
    padding: Spacing.md,
    borderWidth: 1,
    borderColor: Colors.border,
    ...Typography.input,
    color: Colors.textPrimary,
    minHeight: Spacing.inputHeight,
  },
  relationshipGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: Spacing.sm,
  },
  relationshipOption: {
    flex: 1,
    minWidth: '45%',
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
    backgroundColor: Colors.surface,
    padding: Spacing.md,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: Colors.border,
  },
  relationshipOptionActive: {
    borderColor: Colors.primary,
    backgroundColor: Colors.primaryLight + '10',
  },
  relationshipLabel: {
    ...Typography.body,
    color: Colors.textSecondary,
  },
  switchRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  switchLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.md,
    flex: 1,
  },
  switchLabel: {
    ...Typography.body,
    color: Colors.textPrimary,
    marginBottom: 2,
  },
  switchDescription: {
    ...Typography.captionSmall,
    color: Colors.textSecondary,
  },
  infoSection: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: Spacing.sm,
    backgroundColor: Colors.info + '10',
    padding: Spacing.md,
    margin: Spacing.lg,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: Colors.info + '40',
  },
  infoText: {
    ...Typography.captionSmall,
    color: Colors.textSecondary,
    flex: 1,
    lineHeight: 18,
  },
  footer: {
    flexDirection: 'row',
    padding: Spacing.lg,
    gap: Spacing.md,
    borderTopWidth: 1,
    borderTopColor: Colors.border,
    backgroundColor: Colors.surface,
  },
  cancelButton: {
    flex: 1,
    backgroundColor: Colors.surface,
    borderRadius: 12,
    paddingVertical: Spacing.md,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: Colors.border,
    minHeight: Spacing.buttonHeight,
    justifyContent: 'center',
  },
  cancelButtonText: {
    ...Typography.button,
    color: Colors.textSecondary,
  },
  saveButton: {
    flex: 1,
    backgroundColor: Colors.primary,
    borderRadius: 12,
    paddingVertical: Spacing.md,
    alignItems: 'center',
    minHeight: Spacing.buttonHeight,
    justifyContent: 'center',
  },
  saveButtonText: {
    ...Typography.button,
    color: Colors.textInverse,
    fontWeight: '600',
  },
});
