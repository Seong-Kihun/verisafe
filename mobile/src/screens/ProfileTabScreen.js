/**
 * 내페이지탭 화면
 * 프로필, 즐겨찾기, 설정
 */

import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Image,
  ActivityIndicator,
  Linking,
  Platform,
} from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { useTranslation } from 'react-i18next';
import { Colors, Typography, Spacing, CommonStyles } from '../styles';
import {
  userProfileStorage,
  statsStorage,
  savedPlacesStorage,
  recentRoutesStorage,
  myReportsStorage,
} from '../services/storage';
import Icon from '../components/icons/Icon';

export default function ProfileTabScreen({ navigation }) {
  const { t } = useTranslation();
  const [loading, setLoading] = useState(true);
  const [user, setUser] = useState(null);
  const [stats, setStats] = useState({
    reportsSubmitted: 0,
    reportsVerified: 0,
    routesCalculated: 0,
  });
  const [savedPlaces, setSavedPlaces] = useState([]);
  const [recentRoutes, setRecentRoutes] = useState([]);
  const [myReports, setMyReports] = useState([]);

  // 화면이 포커스될 때마다 데이터 다시 로드
  useFocusEffect(
    useCallback(() => {
      loadAllData();
    }, [])
  );

  const loadAllData = async () => {
    setLoading(true);
    try {
      const [userData, statsData, placesData, routesData, reportsData] = await Promise.all([
        userProfileStorage.get(),
        statsStorage.get(),
        savedPlacesStorage.getAll(),
        recentRoutesStorage.getAll(),
        myReportsStorage.getAll(),
      ]);

      setUser(userData);
      setStats(statsData);
      setSavedPlaces(placesData.slice(0, 3)); // 최대 3개만 표시
      setRecentRoutes(routesData.slice(0, 3)); // 최대 3개만 표시
      setMyReports(reportsData.slice(0, 3)); // 최대 3개만 표시
    } catch (error) {
      console.error('Failed to load profile data:', error);
    } finally {
      setLoading(false);
    }
  };

  const calculateProfileCompletion = (user) => {
    if (!user) return 0;
    const fields = ['name', 'email', 'phone', 'organization'];
    const filled = fields.filter(field => user[field] && user[field].trim().length > 0).length;
    return Math.round((filled / fields.length) * 100);
  };

  const getBadge = (stats) => {
    if (!stats) return null;

    const { reportsSubmitted, reportsVerified } = stats;
    const verificationRate = reportsSubmitted > 0
      ? Math.round((reportsVerified / reportsSubmitted) * 100)
      : 0;

    // 배지 조건
    if (reportsVerified >= 50 && verificationRate >= 80) {
      return { icon: '🏆', label: '전설의 제보자', color: '#FFD700' };
    }
    if (reportsVerified >= 20 && verificationRate >= 70) {
      return { icon: '⭐', label: '믿을 수 있는 제보자', color: '#4CAF50' };
    }
    if (reportsSubmitted >= 10) {
      return { icon: '🎖️', label: '활발한 제보자', color: '#2196F3' };
    }
    if (reportsSubmitted >= 1) {
      return { icon: '🌟', label: '첫 제보 완료', color: '#FF9800' };
    }
    return null;
  };

  const getCategoryIcon = (category) => {
    const icons = {
      airport: '✈️',
      government: '🏛️',
      hospital: '🏥',
      hotel: '🏨',
      restaurant: '🍽️',
      shop: '🏪',
      other: '📍',
    };
    return icons[category] || '📍';
  };

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color={Colors.primary} />
      </View>
    );
  }

  if (!user) {
    return (
      <View style={styles.loadingContainer}>
        <Text style={styles.errorText}>{t('profile.loadError')}</Text>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container}>
      {/* 프로필 섹션 */}
      <View style={styles.profileSection}>
        <TouchableOpacity onPress={() => navigation.navigate('ProfileEdit')}>
          <View style={styles.avatarContainer}>
            <View style={styles.avatar}>
              <Text style={styles.avatarText}>
                {user.name.charAt(0).toUpperCase()}
              </Text>
            </View>
            <View style={styles.editBadge}>
              <Icon name="edit" size={16} color={Colors.textInverse} />
            </View>
          </View>
        </TouchableOpacity>
        <Text style={styles.userName}>{user.name}</Text>
        <Text style={styles.userEmail}>{user.email}</Text>
        {user.organization && (
          <Text style={styles.userOrganization}>{user.organization}</Text>
        )}

        {/* 배지 표시 */}
        {(() => {
          const badge = getBadge(stats);
          if (badge) {
            return (
              <View style={[styles.badgeContainer, { backgroundColor: badge.color + '20' }]}>
                <Text style={styles.badgeIcon}>{badge.icon}</Text>
                <Text style={[styles.badgeLabel, { color: badge.color }]}>
                  {badge.label}
                </Text>
              </View>
            );
          }
          return null;
        })()}

        {/* 프로필 완성도 */}
        {(() => {
          const completion = calculateProfileCompletion(user);
          if (completion < 100) {
            return (
              <View style={styles.completionContainer}>
                <View style={styles.completionHeader}>
                  <Text style={styles.completionLabel}>프로필 완성도</Text>
                  <Text style={styles.completionPercent}>{completion}%</Text>
                </View>
                <View style={styles.completionBarContainer}>
                  <View style={[styles.completionBar, { width: `${completion}%` }]} />
                </View>
                {completion < 100 && (
                  <TouchableOpacity
                    style={styles.completeProfileButton}
                    onPress={() => navigation.navigate('ProfileEdit')}
                  >
                    <Text style={styles.completeProfileText}>프로필 완성하기</Text>
                    <Icon name="chevronRight" size={16} color={Colors.primary} />
                  </TouchableOpacity>
                )}
              </View>
            );
          }
          return null;
        })()}
      </View>

      {/* 통계 섹션 */}
      <View style={styles.statsSection}>
        <TouchableOpacity style={styles.statCard} onPress={() => navigation.navigate('MyReports')}>
          <Text style={styles.statValue}>{stats.reportsSubmitted}</Text>
          <Text style={styles.statLabel}>{t('profile.stats.reportsSubmitted')}</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.statCard} onPress={() => navigation.navigate('MyReports')}>
          <Text style={styles.statValue}>{stats.reportsVerified}</Text>
          <Text style={styles.statLabel}>{t('profile.stats.reportsVerified')}</Text>
          {stats.reportsSubmitted > 0 && (
            <Text style={styles.statSubLabel}>
              {Math.round((stats.reportsVerified / stats.reportsSubmitted) * 100)}% 검증률
            </Text>
          )}
        </TouchableOpacity>
        <TouchableOpacity style={styles.statCard} onPress={() => navigation.navigate('RecentRoutes')}>
          <Text style={styles.statValue}>{stats.routesCalculated}</Text>
          <Text style={styles.statLabel}>{t('profile.stats.routesCalculated')}</Text>
        </TouchableOpacity>
      </View>

      {/* 나의 활동 섹션 */}
      <View style={styles.section}>
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>{t('profile.myActivity')}</Text>
        </View>

        {/* 매퍼/관리자 전용: 웹 포털 링크 */}
        {user && (user.role === 'mapper' || user.role === 'admin') && (
          <TouchableOpacity
            style={styles.menuItem}
            onPress={() => {
              const portalUrl = Platform.OS === 'web'
                ? 'http://localhost:3000'
                : 'http://192.168.45.177:3000';
              Linking.openURL(portalUrl).catch(err =>
                console.error('Failed to open URL:', err)
              );
            }}
          >
            <View style={styles.menuLeft}>
              <Icon name="computer" size={24} color={Colors.primary} />
              <View>
                <Text style={styles.menuLabel}>매핑 포털</Text>
                <Text style={styles.menuDescription}>
                  {user.role === 'admin' ? '지도 편집 & 검수하기' : '지도 편집하기'}
                </Text>
              </View>
            </View>
            <Icon name="chevronRight" size={20} color={Colors.textTertiary} />
          </TouchableOpacity>
        )}

        <TouchableOpacity
          style={styles.menuItem}
          onPress={() => navigation.navigate('MyReports')}
        >
          <View style={styles.menuLeft}>
            <Icon name="report" size={24} color={Colors.primary} />
            <View>
              <Text style={styles.menuLabel}>{t('profile.myReports')}</Text>
              <Text style={styles.menuDescription}>{t('profile.reportsCount', { count: myReports.length })}</Text>
            </View>
          </View>
          <Icon name="chevronRight" size={20} color={Colors.textTertiary} />
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.menuItem}
          onPress={() => navigation.navigate('SavedPlaces')}
        >
          <View style={styles.menuLeft}>
            <Icon name="bookmarkBorder" size={24} color={Colors.primary} />
            <View>
              <Text style={styles.menuLabel}>{t('profile.savedPlaces')}</Text>
              <Text style={styles.menuDescription}>{t('profile.placesCount', { count: savedPlaces.length })}</Text>
            </View>
          </View>
          <Icon name="chevronRight" size={20} color={Colors.textTertiary} />
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.menuItem}
          onPress={() => navigation.navigate('RecentRoutes')}
        >
          <View style={styles.menuLeft}>
            <Icon name="history" size={24} color={Colors.primary} />
            <View>
              <Text style={styles.menuLabel}>{t('profile.recentRoutes')}</Text>
              <Text style={styles.menuDescription}>{t('profile.routesCount', { count: recentRoutes.length })}</Text>
            </View>
          </View>
          <Icon name="chevronRight" size={20} color={Colors.textTertiary} />
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.menuItem}
          onPress={() => navigation.navigate('EmergencyContacts')}
        >
          <View style={styles.menuLeft}>
            <Icon name="contactPhone" size={24} color={Colors.danger} />
            <View>
              <Text style={styles.menuLabel}>긴급 연락망</Text>
              <Text style={styles.menuDescription}>위급 상황 자동 알림</Text>
            </View>
          </View>
          <Icon name="chevronRight" size={20} color={Colors.textTertiary} />
        </TouchableOpacity>
      </View>

      {/* 설정 섹션 */}
      <View style={styles.section}>
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>{t('profile.settings')}</Text>
        </View>

        <TouchableOpacity
          style={styles.menuItem}
          onPress={() => navigation.navigate('Settings')}
        >
          <View style={styles.menuLeft}>
            <Icon name="settings" size={24} color={Colors.textPrimary} />
            <Text style={styles.menuLabel}>{t('profile.environmentSettings')}</Text>
          </View>
          <Icon name="chevronRight" size={20} color={Colors.textTertiary} />
        </TouchableOpacity>
      </View>

      {/* 정보 섹션 */}
      <View style={styles.section}>
        <Text style={styles.version}>VeriSafe v1.0.0</Text>
        <Text style={styles.copyright}>{t('copyright')}</Text>
      </View>
    </ScrollView>
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
  errorText: {
    ...Typography.body,
    color: Colors.danger,
  },
  profileSection: {
    backgroundColor: Colors.surface,
    padding: Spacing.xl,
    alignItems: 'center',
    ...CommonStyles.shadowSmall,
  },
  avatarContainer: {
    marginBottom: Spacing.md,
    position: 'relative',
  },
  avatar: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: Colors.primary,
    justifyContent: 'center',
    alignItems: 'center',
  },
  avatarText: {
    ...Typography.h1,
    color: Colors.textInverse,
    fontSize: 36,
    fontWeight: '700',
  },
  editBadge: {
    position: 'absolute',
    bottom: 0,
    right: 0,
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: Colors.primary,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: Colors.surface,
  },
  userName: {
    ...Typography.h2,
    color: Colors.textPrimary,
    marginBottom: Spacing.xs,
  },
  userEmail: {
    ...Typography.bodyMedium,
    color: Colors.textSecondary,
  },
  userOrganization: {
    ...Typography.bodySmall,
    color: Colors.textTertiary,
    marginTop: Spacing.xs,
  },
  badgeContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.xs,
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.sm,
    borderRadius: 20,
    marginTop: Spacing.md,
  },
  badgeIcon: {
    fontSize: 20,
  },
  badgeLabel: {
    ...Typography.labelSmall,
    fontWeight: '700',
  },
  completionContainer: {
    width: '100%',
    marginTop: Spacing.lg,
    paddingTop: Spacing.md,
    borderTopWidth: 1,
    borderTopColor: Colors.borderLight,
  },
  completionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: Spacing.sm,
  },
  completionLabel: {
    ...Typography.bodySmall,
    color: Colors.textSecondary,
  },
  completionPercent: {
    ...Typography.labelSmall,
    color: Colors.primary,
    fontWeight: '700',
  },
  completionBarContainer: {
    width: '100%',
    height: 8,
    backgroundColor: Colors.borderLight,
    borderRadius: 4,
    overflow: 'hidden',
  },
  completionBar: {
    height: '100%',
    backgroundColor: Colors.primary,
    borderRadius: 4,
  },
  completeProfileButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: Spacing.xs,
    marginTop: Spacing.sm,
  },
  completeProfileText: {
    ...Typography.labelSmall,
    color: Colors.primary,
    fontWeight: '600',
  },
  statsSection: {
    flexDirection: 'row',
    backgroundColor: Colors.surface,
    marginTop: Spacing.lg,
    paddingVertical: Spacing.lg,
    ...CommonStyles.shadowSmall,
  },
  statCard: {
    flex: 1,
    alignItems: 'center',
  },
  statValue: {
    ...Typography.h1,
    color: Colors.primary,
    fontSize: 28,
    marginBottom: Spacing.xs,
  },
  statLabel: {
    ...Typography.caption,
    color: Colors.textSecondary,
  },
  statSubLabel: {
    ...Typography.captionSmall,
    color: Colors.success,
    fontWeight: '600',
    marginTop: Spacing.xs,
  },
  section: {
    backgroundColor: Colors.surface,
    marginTop: Spacing.lg,
    paddingTop: Spacing.md,
    ...CommonStyles.shadowSmall,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: Spacing.sm,
    paddingHorizontal: Spacing.lg,
  },
  sectionTitle: {
    ...Typography.h3,
    color: Colors.textPrimary,
  },
  menuItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: Spacing.md,
    paddingHorizontal: Spacing.lg,
    borderBottomWidth: 1,
    borderBottomColor: Colors.borderLight,
  },
  menuLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.md,
    flex: 1,
  },
  menuLabel: {
    ...Typography.body,
    color: Colors.textPrimary,
    marginBottom: 2,
  },
  menuDescription: {
    ...Typography.captionSmall,
    color: Colors.textSecondary,
  },
  version: {
    ...Typography.bodySmall,
    color: Colors.textSecondary,
    textAlign: 'center',
    marginTop: Spacing.lg,
    marginBottom: Spacing.xs,
    paddingHorizontal: Spacing.lg,
  },
  copyright: {
    ...Typography.captionSmall,
    color: Colors.textTertiary,
    textAlign: 'center',
    marginBottom: Spacing.lg,
    paddingHorizontal: Spacing.lg,
  },
});
