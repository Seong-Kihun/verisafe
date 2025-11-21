/**
 * 온보딩 - 권한 요청 화면
 */

import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  Alert,
  Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import * as Location from 'expo-location';
import * as Notifications from 'expo-notifications';
import { Colors, Typography, Spacing } from '../../styles';
import { useOnboarding } from '../../contexts/OnboardingContext';
import Icon from '../../components/icons/Icon';

export default function PermissionScreen({ navigation }) {
  const { updateOnboardingData } = useOnboarding();
  const [permissions, setPermissions] = useState({
    location: false,
    notifications: false,
    camera: false,
  });

  const requestLocationPermission = async () => {
    try {
      const { status } = await Location.requestForegroundPermissionsAsync();
      const granted = status === 'granted';
      setPermissions(prev => ({ ...prev, location: granted }));
      updateOnboardingData('permissionsGranted', { ...permissions, location: granted });

      if (!granted) {
        Alert.alert(
          '위치 권한',
          '위치 권한이 거부되었습니다. 경로 안내 기능을 사용하려면 설정에서 권한을 허용해주세요.'
        );
      }
    } catch (error) {
      console.error('Location permission error:', error);
    }
  };

  const requestNotificationPermission = async () => {
    try {
      const { status } = await Notifications.requestPermissionsAsync();
      const granted = status === 'granted';
      setPermissions(prev => ({ ...prev, notifications: granted }));
      updateOnboardingData('permissionsGranted', { ...permissions, notifications: granted });

      if (!granted) {
        Alert.alert(
          '알림 권한',
          '알림 권한이 거부되었습니다. 위험 정보 알림을 받으려면 설정에서 권한을 허용해주세요.'
        );
      }
    } catch (error) {
      console.error('Notification permission error:', error);
    }
  };

  const requestCameraPermission = () => {
    // 카메라 권한은 실제 사용 시 요청하도록 설정
    setPermissions(prev => ({ ...prev, camera: true }));
    updateOnboardingData('permissionsGranted', { ...permissions, camera: true });
    Alert.alert(
      '카메라 권한',
      '카메라는 위험 제보 시 사진 촬영에 사용됩니다. 실제 사용 시 권한을 요청합니다.'
    );
  };

  const handleNext = () => {
    navigation.navigate('EmergencyContact');
  };

  const handleSkip = () => {
    navigation.navigate('EmergencyContact');
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

      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* 헤더 */}
        <View style={styles.header}>
          <View style={styles.iconContainer}>
            <Icon name="lock" size={48} color={Colors.primary} />
          </View>
          <Text style={styles.title}>권한 설정</Text>
          <Text style={styles.subtitle}>
            VeriSafe가 최상의 경험을 제공하기 위해 필요한 권한입니다
          </Text>
        </View>

        {/* 권한 목록 */}
        <View style={styles.permissionsContainer}>
          <PermissionItem
            icon="location"
            title="위치"
            description="안전한 경로 안내와 주변 위험 정보 제공"
            required={true}
            granted={permissions.location}
            onRequest={requestLocationPermission}
          />
          <PermissionItem
            icon="notifications"
            title="알림"
            description="실시간 위험 정보와 중요한 알림 받기"
            required={false}
            granted={permissions.notifications}
            onRequest={requestNotificationPermission}
          />
          <PermissionItem
            icon="camera"
            title="카메라"
            description="위험 상황 제보 시 사진 첨부"
            required={false}
            granted={permissions.camera}
            onRequest={requestCameraPermission}
          />
        </View>

        {/* 안내 문구 */}
        <View style={styles.infoBox}>
          <Icon name="shield" size={20} color={Colors.success} />
          <Text style={styles.infoText}>
            VeriSafe는 사용자의 개인정보를 보호하며, 수집된 정보는 안전 서비스 제공 목적으로만 사용됩니다.
          </Text>
        </View>
      </ScrollView>

      {/* 하단 버튼 */}
      <View style={styles.footer}>
        <TouchableOpacity
          style={styles.skipButton}
          onPress={handleSkip}
          activeOpacity={0.7}
        >
          <Text style={styles.skipButtonText}>나중에</Text>
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

// 권한 항목 컴포넌트
const PermissionItem = ({ icon, title, description, required, granted, onRequest }) => (
  <View style={styles.permissionItem}>
    <View style={styles.permissionLeft}>
      <View style={[styles.permissionIcon, granted && styles.permissionIconGranted]}>
        <Icon name={icon} size={28} color={granted ? Colors.success : Colors.textSecondary} />
      </View>
      <View style={styles.permissionTextContainer}>
        <View style={styles.permissionTitleRow}>
          <Text style={styles.permissionTitle}>{title}</Text>
          {required && (
            <View style={styles.requiredBadge}>
              <Text style={styles.requiredText}>필수</Text>
            </View>
          )}
        </View>
        <Text style={styles.permissionDescription}>{description}</Text>
      </View>
    </View>
    <TouchableOpacity
      style={[styles.permissionButton, granted && styles.permissionButtonGranted]}
      onPress={onRequest}
      disabled={granted}
      activeOpacity={0.7}
    >
      <Text style={[styles.permissionButtonText, granted && styles.permissionButtonTextGranted]}>
        {granted ? '허용됨' : '허용'}
      </Text>
    </TouchableOpacity>
  </View>
);

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
    paddingBottom: Spacing.xl,
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
  },
  subtitle: {
    ...Typography.body,
    color: Colors.textSecondary,
    textAlign: 'center',
  },
  permissionsContainer: {
    paddingHorizontal: Spacing.xl,
    gap: Spacing.md,
  },
  permissionItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: Colors.surface,
    borderRadius: 12,
    padding: Spacing.lg,
  },
  permissionLeft: {
    flexDirection: 'row',
    flex: 1,
    alignItems: 'center',
    marginRight: Spacing.md,
  },
  permissionIcon: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: Colors.border,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: Spacing.md,
  },
  permissionIconGranted: {
    backgroundColor: Colors.success + '20',
  },
  permissionTextContainer: {
    flex: 1,
  },
  permissionTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: Spacing.xs / 2,
  },
  permissionTitle: {
    ...Typography.bodyLarge,
    fontWeight: '600',
    color: Colors.textPrimary,
    marginRight: Spacing.sm,
  },
  requiredBadge: {
    backgroundColor: Colors.danger + '20',
    paddingHorizontal: Spacing.sm,
    paddingVertical: 2,
    borderRadius: 4,
  },
  requiredText: {
    ...Typography.captionSmall,
    color: Colors.danger,
    fontWeight: '600',
  },
  permissionDescription: {
    ...Typography.bodySmall,
    color: Colors.textSecondary,
    lineHeight: 18,
  },
  permissionButton: {
    paddingHorizontal: Spacing.lg,
    paddingVertical: Spacing.sm,
    borderRadius: 8,
    backgroundColor: Colors.primary,
  },
  permissionButtonGranted: {
    backgroundColor: Colors.success + '20',
  },
  permissionButtonText: {
    ...Typography.buttonSmall,
    color: Colors.textInverse,
  },
  permissionButtonTextGranted: {
    color: Colors.success,
  },
  infoBox: {
    flexDirection: 'row',
    backgroundColor: Colors.success + '10',
    borderRadius: 12,
    padding: Spacing.lg,
    gap: Spacing.md,
    marginHorizontal: Spacing.xl,
    marginTop: Spacing.xl,
  },
  infoText: {
    ...Typography.bodySmall,
    color: Colors.success,
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
