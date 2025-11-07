/**
 * 설정 화면
 */

import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Switch,
  Alert,
  ActivityIndicator,
  Share,
  Modal,
} from 'react-native';
import { useTranslation } from 'react-i18next';
import { Colors, Typography, Spacing } from '../styles';
import { settingsStorage, exportAllData, clearAllData, statsStorage } from '../services/storage';
import Icon from '../components/icons/Icon';
import { setLanguage } from '../i18n';

export default function SettingsScreen({ navigation }) {
  const { t, i18n } = useTranslation();
  const [loading, setLoading] = useState(true);
  const [languageModalVisible, setLanguageModalVisible] = useState(false);
  const [settings, setSettings] = useState({
    notifications: true,
    language: i18n.language || 'ko',
    autoSync: true,
  });

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      const data = await settingsStorage.get();
      if (data) {
        setSettings(data);
      }
    } catch (error) {
      console.error('Failed to load settings:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleToggleNotifications = async (value) => {
    const newSettings = { ...settings, notifications: value };
    setSettings(newSettings);
    await settingsStorage.save(newSettings);
  };

  const handleToggleAutoSync = async (value) => {
    const newSettings = { ...settings, autoSync: value };
    setSettings(newSettings);
    await settingsStorage.save(newSettings);
  };

  const handleLanguageChange = () => {
    setLanguageModalVisible(true);
  };

  const handleSelectLanguage = async (language) => {
    const newSettings = { ...settings, language };
    setSettings(newSettings);
    await settingsStorage.save(newSettings);
    await setLanguage(language);
    setLanguageModalVisible(false);
  };

  const handleExportData = async () => {
    Alert.alert(
      t('settings.exportDataConfirm.title'),
      t('settings.exportDataConfirm.message'),
      [
        { text: t('common.cancel'), style: 'cancel' },
        {
          text: t('settings.exportData'),
          onPress: async () => {
            try {
              const data = await exportAllData();
              if (data) {
                const jsonString = JSON.stringify(data, null, 2);
                await Share.share({
                  message: jsonString,
                  title: 'VeriSafe Data Backup',
                });
              } else {
                Alert.alert(t('common.error'), t('settings.alerts.exportError'));
              }
            } catch (error) {
              console.error('Failed to export data:', error);
              Alert.alert(t('common.error'), t('settings.alerts.exportError'));
            }
          },
        },
      ]
    );
  };

  const handleClearCache = async () => {
    Alert.alert(
      t('settings.clearCacheConfirm.title'),
      t('settings.clearCacheConfirm.message'),
      [
        { text: t('common.cancel'), style: 'cancel' },
        {
          text: t('common.delete'),
          style: 'destructive',
          onPress: () => {
            Alert.alert(t('common.success'), t('settings.alerts.cacheCleared'));
          },
        },
      ]
    );
  };

  const handleResetStats = async () => {
    Alert.alert(
      t('settings.resetStatsConfirm.title'),
      t('settings.resetStatsConfirm.message'),
      [
        { text: t('common.cancel'), style: 'cancel' },
        {
          text: t('settings.resetStats'),
          style: 'destructive',
          onPress: async () => {
            const success = await statsStorage.reset();
            if (success) {
              Alert.alert(t('common.success'), t('settings.alerts.statsReset'));
            } else {
              Alert.alert(t('common.error'), t('settings.alerts.statsResetError'));
            }
          },
        },
      ]
    );
  };

  const handleClearAllData = async () => {
    Alert.alert(
      t('settings.deleteAllConfirm.title'),
      t('settings.deleteAllConfirm.message'),
      [
        { text: t('common.cancel'), style: 'cancel' },
        {
          text: t('common.delete'),
          style: 'destructive',
          onPress: () => {
            // 한 번 더 확인
            Alert.alert(
              t('settings.deleteAllConfirm.confirmTitle'),
              t('settings.deleteAllConfirm.confirmMessage'),
              [
                { text: t('common.cancel'), style: 'cancel' },
                {
                  text: t('common.delete'),
                  style: 'destructive',
                  onPress: async () => {
                    const success = await clearAllData();
                    if (success) {
                      Alert.alert(t('common.success'), t('settings.alerts.dataDeleted'));
                    } else {
                      Alert.alert(t('common.error'), t('settings.alerts.dataDeleteError'));
                    }
                  },
                },
              ]
            );
          },
        },
      ]
    );
  };

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color={Colors.primary} />
      </View>
    );
  }

  return (
    <>
      <ScrollView style={styles.container}>
      {/* 일반 설정 */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>{t('settings.general')}</Text>

        <View style={styles.settingItem}>
          <View style={styles.settingLeft}>
            <Icon name="notifications" size={24} color={Colors.textPrimary} />
            <Text style={styles.settingLabel}>{t('settings.notifications')}</Text>
          </View>
          <Switch
            value={settings.notifications}
            onValueChange={handleToggleNotifications}
            trackColor={{ false: Colors.border, true: Colors.primary + '60' }}
            thumbColor={settings.notifications ? Colors.primary : Colors.textTertiary}
          />
        </View>

        <TouchableOpacity style={styles.settingItem} onPress={handleLanguageChange}>
          <View style={styles.settingLeft}>
            <Icon name="language" size={24} color={Colors.textPrimary} />
            <Text style={styles.settingLabel}>{t('settings.language')}</Text>
          </View>
          <View style={styles.settingRight}>
            <Text style={styles.settingValue}>
              {t(`settings.languages.${settings.language}`)}
            </Text>
            <Icon name="chevronRight" size={20} color={Colors.textTertiary} />
          </View>
        </TouchableOpacity>

        <View style={styles.settingItem}>
          <View style={styles.settingLeft}>
            <Icon name="sync" size={24} color={Colors.textPrimary} />
            <Text style={styles.settingLabel}>{t('settings.autoSync')}</Text>
          </View>
          <Switch
            value={settings.autoSync}
            onValueChange={handleToggleAutoSync}
            trackColor={{ false: Colors.border, true: Colors.primary + '60' }}
            thumbColor={settings.autoSync ? Colors.primary : Colors.textTertiary}
          />
        </View>
      </View>

      {/* 데이터 관리 */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>{t('settings.data')}</Text>

        <TouchableOpacity style={styles.settingItem} onPress={handleExportData}>
          <View style={styles.settingLeft}>
            <Icon name="download" size={24} color={Colors.primary} />
            <Text style={styles.settingLabel}>{t('settings.exportData')}</Text>
          </View>
          <Icon name="chevronRight" size={20} color={Colors.textTertiary} />
        </TouchableOpacity>

        <TouchableOpacity style={styles.settingItem} onPress={handleClearCache}>
          <View style={styles.settingLeft}>
            <Icon name="delete" size={24} color={Colors.textSecondary} />
            <Text style={styles.settingLabel}>{t('settings.clearCache')}</Text>
          </View>
          <Icon name="chevronRight" size={20} color={Colors.textTertiary} />
        </TouchableOpacity>

        <TouchableOpacity style={styles.settingItem} onPress={handleResetStats}>
          <View style={styles.settingLeft}>
            <Icon name="refresh" size={24} color={Colors.warning} />
            <Text style={styles.settingLabel}>{t('settings.resetStats')}</Text>
          </View>
          <Icon name="chevronRight" size={20} color={Colors.textTertiary} />
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.settingItem, styles.dangerItem]}
          onPress={handleClearAllData}
        >
          <View style={styles.settingLeft}>
            <Icon name="deleteForever" size={24} color={Colors.danger} />
            <Text style={[styles.settingLabel, styles.dangerText]}>{t('settings.deleteAllData')}</Text>
          </View>
          <Icon name="chevronRight" size={20} color={Colors.danger} />
        </TouchableOpacity>
      </View>

      {/* 정보 */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>{t('settings.info')}</Text>

        <View style={styles.infoItem}>
          <Text style={styles.infoLabel}>{t('settings.version')}</Text>
          <Text style={styles.infoValue}>1.0.0</Text>
        </View>

        <View style={styles.infoItem}>
          <Text style={styles.infoLabel}>{t('settings.developer')}</Text>
          <Text style={styles.infoValue}>KOICA</Text>
        </View>

        <View style={styles.infoItem}>
          <Text style={styles.infoLabel}>{t('settings.contact')}</Text>
          <Text style={styles.infoValue}>support@verisafe.com</Text>
        </View>
      </View>

      <View style={styles.footer}>
        <Text style={styles.copyright}>{t('copyright')}</Text>
      </View>
    </ScrollView>

      {/* 언어 선택 모달 */}
      <Modal
        visible={languageModalVisible}
        transparent={true}
        animationType="fade"
        onRequestClose={() => setLanguageModalVisible(false)}
      >
        <TouchableOpacity
          style={styles.modalOverlay}
          activeOpacity={1}
          onPress={() => setLanguageModalVisible(false)}
        >
          <View style={styles.modalContent} onStartShouldSetResponder={() => true}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>{t('settings.languageSelect.title')}</Text>
              <Text style={styles.modalSubtitle}>{t('settings.languageSelect.message')}</Text>
            </View>

            <View style={styles.languageList}>
              {[
                { code: 'ko', name: t('settings.languages.ko'), icon: '🇰🇷' },
                { code: 'en', name: t('settings.languages.en'), icon: '🇺🇸' },
                { code: 'es', name: t('settings.languages.es'), icon: '🇪🇸' },
                { code: 'fr', name: t('settings.languages.fr'), icon: '🇫🇷' },
                { code: 'pt', name: t('settings.languages.pt'), icon: '🇵🇹' },
                { code: 'sw', name: t('settings.languages.sw'), icon: '🇹🇿' },
              ].map((lang) => (
                <TouchableOpacity
                  key={lang.code}
                  style={[
                    styles.languageOption,
                    settings.language === lang.code && styles.languageOptionSelected,
                  ]}
                  onPress={() => handleSelectLanguage(lang.code)}
                >
                  <Text style={styles.languageIcon}>{lang.icon}</Text>
                  <Text style={[
                    styles.languageName,
                    settings.language === lang.code && styles.languageNameSelected,
                  ]}>
                    {lang.name}
                  </Text>
                  {settings.language === lang.code && (
                    <Icon name="check" size={24} color={Colors.primary} />
                  )}
                </TouchableOpacity>
              ))}
            </View>

            <TouchableOpacity
              style={styles.modalCancelButton}
              onPress={() => setLanguageModalVisible(false)}
            >
              <Text style={styles.modalCancelText}>{t('common.cancel')}</Text>
            </TouchableOpacity>
          </View>
        </TouchableOpacity>
      </Modal>
    </>
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
  section: {
    backgroundColor: Colors.surface,
    marginTop: Spacing.lg,
    paddingVertical: Spacing.md,
  },
  sectionTitle: {
    ...Typography.h3,
    color: Colors.textPrimary,
    paddingHorizontal: Spacing.lg,
    paddingVertical: Spacing.sm,
  },
  settingItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: Spacing.md,
    paddingHorizontal: Spacing.lg,
    borderBottomWidth: 1,
    borderBottomColor: Colors.borderLight,
  },
  settingLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.md,
    flex: 1,
  },
  settingLabel: {
    ...Typography.body,
    color: Colors.textPrimary,
  },
  settingRight: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
  },
  settingValue: {
    ...Typography.body,
    color: Colors.textSecondary,
  },
  dangerItem: {
    borderBottomWidth: 0,
  },
  dangerText: {
    color: Colors.danger,
  },
  infoItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: Spacing.md,
    paddingHorizontal: Spacing.lg,
    borderBottomWidth: 1,
    borderBottomColor: Colors.borderLight,
  },
  infoLabel: {
    ...Typography.body,
    color: Colors.textSecondary,
  },
  infoValue: {
    ...Typography.body,
    color: Colors.textPrimary,
  },
  footer: {
    padding: Spacing.xl,
    alignItems: 'center',
  },
  copyright: {
    ...Typography.captionSmall,
    color: Colors.textTertiary,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: Spacing.lg,
  },
  modalContent: {
    backgroundColor: Colors.surface,
    borderRadius: 16,
    width: '100%',
    maxWidth: 400,
    padding: Spacing.xl,
    shadowColor: Colors.shadowDark,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 12,
    elevation: 8,
  },
  modalHeader: {
    marginBottom: Spacing.lg,
  },
  modalTitle: {
    ...Typography.h2,
    color: Colors.textPrimary,
    marginBottom: Spacing.xs,
  },
  modalSubtitle: {
    ...Typography.body,
    color: Colors.textSecondary,
  },
  languageList: {
    gap: Spacing.sm,
  },
  languageOption: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: Spacing.md,
    borderRadius: 12,
    backgroundColor: Colors.background,
    borderWidth: 2,
    borderColor: 'transparent',
  },
  languageOptionSelected: {
    backgroundColor: Colors.primary + '10',
    borderColor: Colors.primary,
  },
  languageIcon: {
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
  modalCancelButton: {
    marginTop: Spacing.lg,
    padding: Spacing.md,
    borderRadius: 12,
    backgroundColor: Colors.background,
    alignItems: 'center',
  },
  modalCancelText: {
    ...Typography.bodyLarge,
    color: Colors.textSecondary,
  },
});
