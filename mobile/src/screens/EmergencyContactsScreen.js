/**
 * 긴급 연락망 관리 화면
 */

import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  Linking,
  Switch,
} from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { Colors, Typography, Spacing } from '../styles';
import { emergencyContactsStorage } from '../services/storage';
import Icon from '../components/icons/Icon';
import { GestureSettings } from '../services/emergencyGesture';

export default function EmergencyContactsScreen({ navigation }) {
  const [loading, setLoading] = useState(true);
  const [contacts, setContacts] = useState([]);
  const [gestureSettings, setGestureSettings] = useState(null);

  useFocusEffect(
    useCallback(() => {
      loadContacts();
      loadGestureSettings();
    }, [])
  );

  const loadContacts = async () => {
    try {
      const data = await emergencyContactsStorage.getAll();
      // 우선순위 순으로 정렬
      const sorted = data.sort((a, b) => a.priority - b.priority);
      setContacts(sorted);
    } catch (error) {
      console.error('[EmergencyContacts] Failed to load emergency contacts:', error);
      Alert.alert('오류', '긴급 연락망을 불러올 수 없습니다.');
    } finally {
      setLoading(false);
    }
  };

  const loadGestureSettings = async () => {
    try {
      const data = await GestureSettings.load();
      setGestureSettings(data);
    } catch (error) {
      console.error('[EmergencyContacts] Failed to load gesture settings:', error);
    }
  };

  const handleToggleTapGesture = async (value) => {
    const newSettings = { ...gestureSettings, tapGestureEnabled: value };
    setGestureSettings(newSettings);
    await GestureSettings.save(newSettings);
  };

  const handleToggleVolumeGesture = async (value) => {
    const newSettings = { ...gestureSettings, volumeGestureEnabled: value };
    setGestureSettings(newSettings);
    await GestureSettings.save(newSettings);
  };

  const handleEdit = (contact) => {
    // Type guard: validate contact before navigation
    if (!contact || !contact.id) {
      console.error('[EmergencyContacts] Invalid contact for edit:', contact);
      Alert.alert('오류', '연락처 정보를 불러올 수 없습니다.');
      return;
    }
    navigation.navigate('EmergencyContactEdit', { contact });
  };

  const handleDelete = (contactId, contactName) => {
    Alert.alert(
      '삭제 확인',
      `${contactName}을(를) 긴급 연락망에서 삭제하시겠습니까?`,
      [
        { text: '취소', style: 'cancel' },
        {
          text: '삭제',
          style: 'destructive',
          onPress: async () => {
            const success = await emergencyContactsStorage.remove(contactId);
            if (success) {
              setContacts(prev => prev.filter(c => c.id !== contactId));
            } else {
              Alert.alert('오류', '삭제에 실패했습니다.');
            }
          },
        },
      ]
    );
  };

  const handleCall = (phone) => {
    Linking.openURL(`tel:${phone}`);
  };

  const getRelationshipLabel = (relationship) => {
    const labels = {
      family: '가족',
      friend: '친구',
      colleague: '동료',
      other: '기타',
    };
    return labels[relationship] || '기타';
  };

  const getRelationshipIcon = (relationship) => {
    const icons = {
      family: 'people',
      friend: 'person',
      colleague: 'work',
      other: 'contactPhone',
    };
    return icons[relationship] || 'contactPhone';
  };

  const renderItem = ({ item }) => (
    <View style={styles.contactCard}>
      <View style={styles.contactHeader}>
        <View style={styles.priorityBadge}>
          <Text style={styles.priorityText}>{item.priority}</Text>
        </View>
        <View style={styles.contactInfo}>
          <Text style={styles.contactName}>{item.name}</Text>
          <Text style={styles.contactPhone}>{item.phone}</Text>
          {item.email && (
            <Text style={styles.contactEmail}>{item.email}</Text>
          )}
        </View>
        <View style={styles.relationshipBadge}>
          <Icon
            name={getRelationshipIcon(item.relationship)}
            size={16}
            color={Colors.primary}
          />
          <Text style={styles.relationshipText}>
            {getRelationshipLabel(item.relationship)}
          </Text>
        </View>
      </View>

      <View style={styles.contactMeta}>
        <View style={styles.metaItem}>
          <Icon
            name={item.shareLocation ? 'locationOn' : 'locationOff'}
            size={16}
            color={item.shareLocation ? Colors.success : Colors.textTertiary}
          />
          <Text style={[
            styles.metaText,
            item.shareLocation ? styles.metaActive : styles.metaInactive
          ]}>
            {item.shareLocation ? '위치 공유 ON' : '위치 공유 OFF'}
          </Text>
        </View>
      </View>

      <View style={styles.contactActions}>
        <TouchableOpacity
          style={styles.actionButton}
          onPress={() => handleCall(item.phone)}
        >
          <Icon name="phone" size={20} color={Colors.success} />
          <Text style={[styles.actionButtonText, { color: Colors.success }]}>
            전화
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.actionButton}
          onPress={() => handleEdit(item)}
        >
          <Icon name="edit" size={20} color={Colors.primary} />
          <Text style={styles.actionButtonText}>수정</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.actionButton, styles.deleteButton]}
          onPress={() => handleDelete(item.id, item.name)}
        >
          <Icon name="delete" size={20} color={Colors.danger} />
          <Text style={[styles.actionButtonText, styles.deleteText]}>삭제</Text>
        </TouchableOpacity>
      </View>
    </View>
  );

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color={Colors.primary} />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {contacts.length === 0 ? (
        <View style={styles.emptyContainer}>
          <Icon name="contactPhone" size={64} color={Colors.textTertiary} />
          <Text style={styles.emptyTitle}>긴급 연락망이 비어있습니다</Text>
          <Text style={styles.emptyText}>
            위급한 상황에서 자동으로{'\n'}
            연락을 받을 사람을 추가하세요{'\n'}
            (최대 5명)
          </Text>
          <TouchableOpacity
            style={styles.addFirstButton}
            onPress={() => navigation.navigate('EmergencyContactEdit')}
          >
            <Icon name="add" size={20} color={Colors.textInverse} />
            <Text style={styles.addFirstButtonText}>첫 연락처 추가</Text>
          </TouchableOpacity>
        </View>
      ) : (
        <>
          <View style={styles.header}>
            <View style={styles.headerLeft}>
              <Text style={styles.headerTitle}>긴급 연락망 {contacts.length}/5</Text>
              <Text style={styles.headerSubtitle}>
                우선순위 순으로 알림이 전송됩니다
              </Text>
            </View>
            {contacts.length < 5 && (
              <TouchableOpacity
                style={styles.addButton}
                onPress={() => navigation.navigate('EmergencyContactEdit')}
              >
                <Icon name="add" size={24} color={Colors.primary} />
              </TouchableOpacity>
            )}
          </View>

          <FlatList
            data={contacts}
            renderItem={renderItem}
            keyExtractor={item => item.id}
            contentContainerStyle={styles.listContainer}
          />

          <View style={styles.infoBox}>
            <Icon name="info" size={20} color={Colors.info} />
            <Text style={styles.infoText}>
              안전 체크인을 놓치면 등록된 연락처로 자동 알림이 전송됩니다.{'\n'}
              아래에서 긴급 제스처 기능을 켜면 빠르게 SOS를 요청할 수 있습니다.
            </Text>
          </View>

          {/* 긴급 제스처 설정 */}
          {gestureSettings && (
            <View style={styles.gestureSection}>
              <View style={styles.gestureSectionHeader}>
                <Icon name="touch" size={24} color={Colors.primary} />
                <View style={{ flex: 1 }}>
                  <Text style={styles.gestureSectionTitle}>🆘 긴급 제스처 설정</Text>
                  <Text style={styles.gestureSectionSubtitle}>
                    특정 제스처로 빠르게 SOS 메시지 발송 (기본적으로 꺼져있음)
                  </Text>
                </View>
              </View>

              {!gestureSettings.tapGestureEnabled && !gestureSettings.volumeGestureEnabled && (
                <View style={styles.enableHintBox}>
                  <Icon name="info" size={16} color={Colors.primary} />
                  <Text style={styles.enableHintText}>
                    아래 스위치를 켜서 긴급 제스처 기능을 활성화하세요
                  </Text>
                </View>
              )}

              <View style={styles.gestureCard}>
                <View style={styles.gestureItemHeader}>
                  <View style={styles.gestureItemLeft}>
                    <Icon name="touch" size={20} color={Colors.textPrimary} />
                    <View style={{ flex: 1 }}>
                      <Text style={styles.gestureItemTitle}>
                        화면 모서리 {gestureSettings.tapCount}번 탭
                      </Text>
                      <Text style={styles.gestureItemDescription}>
                        {gestureSettings.tapTimeout/1000}초 안에 화면 모서리를 {gestureSettings.tapCount}번 탭하면 SOS 발송
                      </Text>
                    </View>
                  </View>
                  <Switch
                    value={gestureSettings.tapGestureEnabled}
                    onValueChange={handleToggleTapGesture}
                    trackColor={{ false: Colors.border, true: Colors.primary + '60' }}
                    thumbColor={gestureSettings.tapGestureEnabled ? Colors.primary : Colors.textTertiary}
                  />
                </View>
              </View>

              <View style={styles.gestureCard}>
                <View style={styles.gestureItemHeader}>
                  <View style={styles.gestureItemLeft}>
                    <Icon name="volume" size={20} color={Colors.textPrimary} />
                    <View style={{ flex: 1 }}>
                      <Text style={styles.gestureItemTitle}>
                        볼륨 버튼 {gestureSettings.volumePressCount}번 누르기
                      </Text>
                      <Text style={styles.gestureItemDescription}>
                        {gestureSettings.volumeTimeout/1000}초 안에 볼륨 버튼을 {gestureSettings.volumePressCount}번 누르면 SOS 발송{'\n'}
                        <Text style={{ color: Colors.textTertiary, fontSize: 11 }}>
                          (커스텀 빌드 필요)
                        </Text>
                      </Text>
                    </View>
                  </View>
                  <Switch
                    value={gestureSettings.volumeGestureEnabled}
                    onValueChange={handleToggleVolumeGesture}
                    trackColor={{ false: Colors.border, true: Colors.primary + '60' }}
                    thumbColor={gestureSettings.volumeGestureEnabled ? Colors.primary : Colors.textTertiary}
                  />
                </View>
              </View>
            </View>
          )}
        </>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: Colors.background,
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: Spacing.xl,
    backgroundColor: Colors.background,
  },
  emptyTitle: {
    ...Typography.h2,
    color: Colors.textPrimary,
    marginTop: Spacing.lg,
    marginBottom: Spacing.sm,
  },
  emptyText: {
    ...Typography.body,
    color: Colors.textSecondary,
    textAlign: 'center',
    marginBottom: Spacing.xl,
    lineHeight: 24,
  },
  addFirstButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.xs,
    backgroundColor: Colors.primary,
    paddingHorizontal: Spacing.lg,
    paddingVertical: Spacing.md,
    borderRadius: 12,
  },
  addFirstButtonText: {
    ...Typography.button,
    color: Colors.textInverse,
    fontWeight: '600',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: Spacing.lg,
    backgroundColor: Colors.surface,
    borderBottomWidth: 1,
    borderBottomColor: Colors.border,
  },
  headerLeft: {
    flex: 1,
  },
  headerTitle: {
    ...Typography.h3,
    color: Colors.textPrimary,
    marginBottom: 4,
  },
  headerSubtitle: {
    ...Typography.caption,
    color: Colors.textSecondary,
  },
  addButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: Colors.primaryLight + '20',
    justifyContent: 'center',
    alignItems: 'center',
  },
  listContainer: {
    padding: Spacing.md,
  },
  contactCard: {
    backgroundColor: Colors.surface,
    borderRadius: 16,
    padding: Spacing.lg,
    marginBottom: Spacing.md,
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 4,
    elevation: 2,
  },
  contactHeader: {
    flexDirection: 'row',
    marginBottom: Spacing.sm,
    alignItems: 'flex-start',
  },
  priorityBadge: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: Colors.primary,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: Spacing.md,
  },
  priorityText: {
    ...Typography.labelSmall,
    color: Colors.textInverse,
    fontWeight: '700',
  },
  contactInfo: {
    flex: 1,
  },
  contactName: {
    ...Typography.h4,
    color: Colors.textPrimary,
    marginBottom: 4,
  },
  contactPhone: {
    ...Typography.body,
    color: Colors.textSecondary,
    marginBottom: 2,
  },
  contactEmail: {
    ...Typography.bodySmall,
    color: Colors.textTertiary,
  },
  relationshipBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: Colors.primaryLight + '20',
    paddingHorizontal: Spacing.sm,
    paddingVertical: 4,
    borderRadius: 8,
  },
  relationshipText: {
    ...Typography.captionSmall,
    color: Colors.primary,
    fontWeight: '600',
  },
  contactMeta: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: Spacing.md,
    paddingLeft: 32 + Spacing.md,
  },
  metaItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  metaText: {
    ...Typography.captionSmall,
  },
  metaActive: {
    color: Colors.success,
    fontWeight: '600',
  },
  metaInactive: {
    color: Colors.textTertiary,
  },
  contactActions: {
    flexDirection: 'row',
    gap: Spacing.sm,
  },
  actionButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: Spacing.xs,
    backgroundColor: Colors.primaryLight + '20',
    paddingVertical: Spacing.sm,
    borderRadius: 12,
  },
  actionButtonText: {
    ...Typography.labelSmall,
    color: Colors.primary,
    fontWeight: '600',
  },
  deleteButton: {
    backgroundColor: Colors.danger + '10',
  },
  deleteText: {
    color: Colors.danger,
  },
  infoBox: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
    backgroundColor: Colors.warning + '10',
    padding: Spacing.md,
    margin: Spacing.lg,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: Colors.warning + '40',
  },
  infoText: {
    ...Typography.captionSmall,
    color: Colors.textSecondary,
    flex: 1,
    lineHeight: 18,
  },
  gestureSection: {
    margin: Spacing.lg,
    backgroundColor: Colors.surface,
    borderRadius: 16,
    padding: Spacing.lg,
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 4,
    elevation: 2,
  },
  gestureSectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.md,
    marginBottom: Spacing.lg,
  },
  gestureSectionTitle: {
    ...Typography.h3,
    color: Colors.textPrimary,
    marginBottom: 4,
  },
  gestureSectionSubtitle: {
    ...Typography.caption,
    color: Colors.textSecondary,
  },
  enableHintBox: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.xs,
    backgroundColor: Colors.primary + '10',
    padding: Spacing.md,
    borderRadius: 12,
    marginBottom: Spacing.md,
    borderWidth: 1,
    borderColor: Colors.primary + '30',
  },
  enableHintText: {
    ...Typography.captionSmall,
    color: Colors.primary,
    flex: 1,
    fontWeight: '600',
  },
  gestureCard: {
    backgroundColor: Colors.background,
    borderRadius: 12,
    padding: Spacing.md,
    marginBottom: Spacing.sm,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  gestureItemHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  gestureItemLeft: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: Spacing.sm,
    marginRight: Spacing.sm,
  },
  gestureItemTitle: {
    ...Typography.body,
    color: Colors.textPrimary,
    fontWeight: '600',
    marginBottom: 4,
  },
  gestureItemDescription: {
    ...Typography.captionSmall,
    color: Colors.textSecondary,
    lineHeight: 16,
  },
});
