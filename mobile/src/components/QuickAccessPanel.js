/**
 * QuickAccessPanel - 빠른 액세스 패널 (Phase 3)
 *
 * 기능:
 * 1. "집", "회사" 빠른 설정
 * 2. 최근 목적지 3개 표시
 * 3. 즐겨찾기 접근
 *
 * 벤치마킹: Kakao Navi의 빠른 설정
 */

import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  Alert,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Colors, Spacing, Typography } from '../styles';
import Icon from './icons/Icon';

const STORAGE_KEYS = {
  HOME: '@verisafe:home_location',
  WORK: '@verisafe:work_location',
  RECENT_DESTINATIONS: '@verisafe:recent_destinations',
  FAVORITES: '@verisafe:favorites',
};

export default function QuickAccessPanel({ onSelectLocation, userLocation }) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [homeLocation, setHomeLocation] = useState(null);
  const [workLocation, setWorkLocation] = useState(null);
  const [recentDestinations, setRecentDestinations] = useState([]);
  const [favorites, setFavorites] = useState([]);

  useEffect(() => {
    loadQuickAccessData();
  }, []);

  const loadQuickAccessData = async () => {
    try {
      const [home, work, recent, favs] = await Promise.all([
        AsyncStorage.getItem(STORAGE_KEYS.HOME),
        AsyncStorage.getItem(STORAGE_KEYS.WORK),
        AsyncStorage.getItem(STORAGE_KEYS.RECENT_DESTINATIONS),
        AsyncStorage.getItem(STORAGE_KEYS.FAVORITES),
      ]);

      if (home) setHomeLocation(JSON.parse(home));
      if (work) setWorkLocation(JSON.parse(work));
      if (recent) setRecentDestinations(JSON.parse(recent));
      if (favs) setFavorites(JSON.parse(favs));
    } catch (error) {
      console.error('[QuickAccessPanel] 데이터 로드 오류:', error);
    }
  };

  const handleSetHome = async () => {
    if (!userLocation) {
      Alert.alert('위치 오류', '현재 위치를 확인할 수 없습니다.');
      return;
    }

    const location = {
      lat: userLocation.latitude,
      lng: userLocation.longitude,
      address: '현재 위치',
      name: '집',
    };

    try {
      await AsyncStorage.setItem(STORAGE_KEYS.HOME, JSON.stringify(location));
      setHomeLocation(location);
      Alert.alert('설정 완료', '집 위치가 설정되었습니다.');
    } catch (error) {
      console.error('[QuickAccessPanel] 집 설정 오류:', error);
      Alert.alert('오류', '집 위치를 설정할 수 없습니다.');
    }
  };

  const handleSetWork = async () => {
    if (!userLocation) {
      Alert.alert('위치 오류', '현재 위치를 확인할 수 없습니다.');
      return;
    }

    const location = {
      lat: userLocation.latitude,
      lng: userLocation.longitude,
      address: '현재 위치',
      name: '회사',
    };

    try {
      await AsyncStorage.setItem(STORAGE_KEYS.WORK, JSON.stringify(location));
      setWorkLocation(location);
      Alert.alert('설정 완료', '회사 위치가 설정되었습니다.');
    } catch (error) {
      console.error('[QuickAccessPanel] 회사 설정 오류:', error);
      Alert.alert('오류', '회사 위치를 설정할 수 없습니다.');
    }
  };

  const handleQuickLocationPress = (location) => {
    if (onSelectLocation) {
      onSelectLocation(location);
    }
    setIsExpanded(false);
  };

  const handleLongPress = (type) => {
    Alert.alert(
      `${type === 'home' ? '집' : '회사'} 삭제`,
      `저장된 ${type === 'home' ? '집' : '회사'} 위치를 삭제하시겠습니까?`,
      [
        { text: '취소', style: 'cancel' },
        {
          text: '삭제',
          style: 'destructive',
          onPress: async () => {
            try {
              await AsyncStorage.removeItem(
                type === 'home' ? STORAGE_KEYS.HOME : STORAGE_KEYS.WORK
              );
              if (type === 'home') {
                setHomeLocation(null);
              } else {
                setWorkLocation(null);
              }
              Alert.alert('삭제 완료', `${type === 'home' ? '집' : '회사'} 위치가 삭제되었습니다.`);
            } catch (error) {
              console.error('[QuickAccessPanel] 삭제 오류:', error);
            }
          },
        },
      ]
    );
  };

  return (
    <View style={styles.container}>
      {/* 접기/펼치기 버튼 */}
      <TouchableOpacity
        style={styles.toggleButton}
        onPress={() => setIsExpanded(!isExpanded)}
        activeOpacity={0.8}
      >
        <Icon name="save" size={20} color={Colors.primary} />
        <Text style={styles.toggleText}>빠른 이동</Text>
        <Icon
          name={isExpanded ? 'chevron-left' : 'chevron-right'}
          size={20}
          color={Colors.textSecondary}
        />
      </TouchableOpacity>

      {/* 펼쳐진 패널 */}
      {isExpanded && (
        <View style={styles.expandedPanel}>
          <ScrollView
            horizontal
            showsHorizontalScrollIndicator={false}
            style={styles.scrollView}
            contentContainerStyle={styles.scrollContent}
          >
            {/* 집 버튼 */}
            <TouchableOpacity
              style={styles.quickButton}
              onPress={() => {
                if (homeLocation) {
                  handleQuickLocationPress(homeLocation);
                } else {
                  handleSetHome();
                }
              }}
              onLongPress={() => homeLocation && handleLongPress('home')}
              activeOpacity={0.7}
            >
              <View style={[styles.iconCircle, homeLocation && styles.iconCircleActive]}>
                <Icon
                  name="local-hospital"
                  size={24}
                  color={homeLocation ? Colors.primary : Colors.textSecondary}
                />
              </View>
              <Text style={styles.quickButtonText}>
                {homeLocation ? '집' : '집 설정'}
              </Text>
            </TouchableOpacity>

            {/* 회사 버튼 */}
            <TouchableOpacity
              style={styles.quickButton}
              onPress={() => {
                if (workLocation) {
                  handleQuickLocationPress(workLocation);
                } else {
                  handleSetWork();
                }
              }}
              onLongPress={() => workLocation && handleLongPress('work')}
              activeOpacity={0.7}
            >
              <View style={[styles.iconCircle, workLocation && styles.iconCircleActive]}>
                <Icon
                  name="account-balance"
                  size={24}
                  color={workLocation ? Colors.primary : Colors.textSecondary}
                />
              </View>
              <Text style={styles.quickButtonText}>
                {workLocation ? '회사' : '회사 설정'}
              </Text>
            </TouchableOpacity>

            {/* 최근 목적지 */}
            {recentDestinations.slice(0, 3).map((destination, index) => (
              <TouchableOpacity
                key={`recent-${index}`}
                style={styles.quickButton}
                onPress={() => handleQuickLocationPress(destination)}
                activeOpacity={0.7}
              >
                <View style={styles.iconCircle}>
                  <Icon name="time" size={24} color={Colors.textSecondary} />
                </View>
                <Text style={styles.quickButtonText} numberOfLines={1}>
                  {destination.name || destination.address}
                </Text>
              </TouchableOpacity>
            ))}

            {/* 즐겨찾기 */}
            {favorites.slice(0, 2).map((favorite, index) => (
              <TouchableOpacity
                key={`fav-${index}`}
                style={styles.quickButton}
                onPress={() => handleQuickLocationPress(favorite)}
                activeOpacity={0.7}
              >
                <View style={styles.iconCircle}>
                  <Icon name="save" size={24} color={Colors.accent} />
                </View>
                <Text style={styles.quickButtonText} numberOfLines={1}>
                  {favorite.name || favorite.address}
                </Text>
              </TouchableOpacity>
            ))}
          </ScrollView>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: Colors.surface,
    borderRadius: 16,
    overflow: 'hidden',
    marginBottom: Spacing.md,
  },
  toggleButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: Spacing.sm,
    paddingHorizontal: Spacing.md,
    gap: Spacing.xs,
  },
  toggleText: {
    ...Typography.label,
    color: Colors.textPrimary,
    fontWeight: '600',
    flex: 1,
  },
  expandedPanel: {
    borderTopWidth: 1,
    borderTopColor: Colors.border,
    paddingVertical: Spacing.sm,
  },
  scrollView: {
    paddingHorizontal: Spacing.sm,
  },
  scrollContent: {
    gap: Spacing.sm,
  },
  quickButton: {
    alignItems: 'center',
    paddingHorizontal: Spacing.sm,
    minWidth: 80,
  },
  iconCircle: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: Colors.borderLight,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: Spacing.xs,
  },
  iconCircleActive: {
    backgroundColor: Colors.primary + '20',
  },
  quickButtonText: {
    ...Typography.labelSmall,
    color: Colors.textSecondary,
    textAlign: 'center',
  },
});

// AsyncStorage 유틸리티 함수들을 export
export const saveRecentDestination = async (destination) => {
  try {
    const existing = await AsyncStorage.getItem(STORAGE_KEYS.RECENT_DESTINATIONS);
    let destinations = existing ? JSON.parse(existing) : [];

    // 중복 제거 (같은 좌표)
    destinations = destinations.filter(
      (d) => !(d.lat === destination.lat && d.lng === destination.lng)
    );

    // 새 목적지를 맨 앞에 추가
    destinations.unshift(destination);

    // 최대 10개까지만 저장
    destinations = destinations.slice(0, 10);

    await AsyncStorage.setItem(
      STORAGE_KEYS.RECENT_DESTINATIONS,
      JSON.stringify(destinations)
    );
  } catch (error) {
    console.error('[QuickAccessPanel] 최근 목적지 저장 오류:', error);
  }
};

export const saveFavorite = async (location) => {
  try {
    const existing = await AsyncStorage.getItem(STORAGE_KEYS.FAVORITES);
    let favorites = existing ? JSON.parse(existing) : [];

    // 중복 확인
    const isDuplicate = favorites.some(
      (f) => f.lat === location.lat && f.lng === location.lng
    );

    if (isDuplicate) {
      Alert.alert('알림', '이미 즐겨찾기에 추가된 위치입니다.');
      return false;
    }

    favorites.unshift(location);
    await AsyncStorage.setItem(STORAGE_KEYS.FAVORITES, JSON.stringify(favorites));
    return true;
  } catch (error) {
    console.error('[QuickAccessPanel] 즐겨찾기 저장 오류:', error);
    return false;
  }
};

export const removeFavorite = async (location) => {
  try {
    const existing = await AsyncStorage.getItem(STORAGE_KEYS.FAVORITES);
    let favorites = existing ? JSON.parse(existing) : [];

    favorites = favorites.filter(
      (f) => !(f.lat === location.lat && f.lng === location.lng)
    );

    await AsyncStorage.setItem(STORAGE_KEYS.FAVORITES, JSON.stringify(favorites));
    return true;
  } catch (error) {
    console.error('[QuickAccessPanel] 즐겨찾기 삭제 오류:', error);
    return false;
  }
};
