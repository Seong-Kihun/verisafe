/**
 * 뉴스탭 화면
 * 위험 뉴스 및 정보 (실제 DB 데이터 연동)
 */

import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  FlatList,
  ActivityIndicator,
  RefreshControl,
} from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { useTranslation } from 'react-i18next';
import { Colors, Typography, Spacing, CommonStyles, getRiskColor } from '../styles';
import { mapAPI } from '../services/api';
import { useMapContext } from '../contexts/MapContext';

const HAZARD_TYPE_IDS = [
  'all',
  'armed_conflict',
  'protest_riot',
  'checkpoint',
  'road_damage',
  'natural_disaster',
  'flood',
  'landslide',
  'other',
];

// 위험 타입별 아이콘
const getHazardIcon = (hazardType) => {
  const iconMap = {
    'armed_conflict': '🔫',
    'conflict': '⚔️',
    'protest_riot': '👥',
    'protest': '📢',
    'checkpoint': '⚠️',
    'road_damage': '🚧',
    'natural_disaster': '💥',
    'flood': '🌊',
    'landslide': '⛰️',
    'other': '⚠️',
  };
  return iconMap[hazardType] || '⚠️';
};

export default function NewsTabScreen() {
  const { t } = useTranslation();
  const navigation = useNavigation();
  const { openPlaceSheet, userCountry } = useMapContext();

  const [selectedFilter, setSelectedFilter] = useState('all');
  const [hazards, setHazards] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  // 초기 로딩 시 위험 정보 로딩
  useEffect(() => {
    loadHazards();
  }, [userCountry]); // userCountry가 변경되면 다시 로드

  const loadHazards = async () => {
    try {
      setLoading(true);
      console.log('[NewsTab DEBUG] 위험 정보 로딩 시작...');

      // 사용자가 선택한 국가의 중심 좌표 사용
      const center = userCountry?.center || { latitude: 4.8594, longitude: 31.5713 }; // 기본값: 주바

      console.log('[NewsTab DEBUG] 국가 중심:', userCountry?.name || '기본(남수단)', center);

      // 선택한 국가 중심에서 반경 50km 내 위험 정보 조회 (국가 필터링 포함)
      const response = await mapAPI.getHazards(
        center.latitude,
        center.longitude,
        50,
        userCountry?.code // 국가 코드로 필터링
      );

      console.log('[NewsTab DEBUG] API 응답 상태:', response.status);
      console.log('[NewsTab DEBUG] API 응답 전체:', JSON.stringify(response.data, null, 2));
      console.log('[NewsTab DEBUG] 데이터 타입:', typeof response.data);
      console.log('[NewsTab DEBUG] 배열 여부:', Array.isArray(response.data));
      console.log('[NewsTab DEBUG] 데이터 개수:', response.data?.length || 0);

      if (response.data && Array.isArray(response.data)) {
        console.log('[NewsTab DEBUG] 첫 번째 데이터 샘플:', response.data[0] ? JSON.stringify(response.data[0], null, 2) : '없음');

        // 날짜순 정렬 (최신순) - start_date 또는 created_at 사용
        const sortedHazards = response.data.sort((a, b) => {
          const dateA = new Date(a.start_date || a.created_at || Date.now());
          const dateB = new Date(b.start_date || b.created_at || Date.now());
          return dateB - dateA;
        });
        setHazards(sortedHazards);
        console.log('[NewsTab DEBUG] ✅ 위험정보 로드 완료:', sortedHazards.length, '개');
      } else {
        console.warn('[NewsTab DEBUG] ⚠️ 응답 데이터가 배열이 아님:', response.data);
        setHazards([]);
      }
    } catch (error) {
      console.error('[NewsTab DEBUG] ❌ 위험 정보 로딩 실패');
      console.error('[NewsTab DEBUG] 에러 메시지:', error.message);
      console.error('[NewsTab DEBUG] 에러 코드:', error.code);
      console.error('[NewsTab DEBUG] 에러 응답:', error.response?.data);
      console.error('[NewsTab DEBUG] 에러 상태:', error.response?.status);
      console.error('[NewsTab DEBUG] 전체 에러:', error);
      setHazards([]);
    } finally {
      setLoading(false);
      console.log('[NewsTab DEBUG] 로딩 완료');
    }
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await loadHazards();
    setRefreshing(false);
  };

  const filteredNews = selectedFilter === 'all'
    ? hazards
    : hazards.filter(item => item.hazard_type === selectedFilter);

  const formatDate = (date) => {
    const now = new Date();
    const diff = now - date;
    const hours = Math.floor(diff / (1000 * 60 * 60));
    const days = Math.floor(hours / 24);

    if (hours < 1) return t('common.timeAgo.justNow');
    if (hours < 24) return t('common.timeAgo.hoursAgo', { count: hours });
    if (days < 7) return t('common.timeAgo.daysAgo', { count: days });
    return date.toLocaleDateString();
  };

  const handleHazardPress = (hazard) => {
    // 지도 탭으로 이동하고 해당 위험 정보 표시
    openPlaceSheet({
      id: hazard.id,
      name: t(`common.hazardTypes.${hazard.hazard_type}`),
      address: hazard.description || '',
      latitude: hazard.latitude,
      longitude: hazard.longitude,
      category: 'danger',
      description: hazard.description,
      risk_score: hazard.risk_score,
      hazard_type: hazard.hazard_type,
      type: 'hazard',
    });

    // 지도 탭으로 이동
    navigation.navigate('MapStack', {
      screen: 'MapMain',
    });
  };

  const renderNewsItem = ({ item }) => {
    // start_date 우선, 없으면 created_at, 둘 다 없으면 현재 시각
    const createdDate = new Date(item.start_date || item.created_at || Date.now());

    return (
      <TouchableOpacity
        style={styles.newsCard}
        onPress={() => handleHazardPress(item)}
        activeOpacity={0.7}
      >
        <View style={styles.newsHeader}>
          <View style={[
            styles.riskBadge,
            { backgroundColor: getRiskColor(item.risk_score) }
          ]}>
            <Text style={styles.riskText}>
              {getHazardIcon(item.hazard_type)}
            </Text>
          </View>
          <View style={styles.newsMeta}>
            <Text style={styles.newsTitle}>
              {t(`common.hazardTypes.${item.hazard_type}`)} {t('news.occurred')}
            </Text>
            <Text style={styles.newsDate}>{formatDate(createdDate)}</Text>
          </View>
        </View>
        <Text style={styles.newsDescription}>
          {item.description || t('news.warningMessage', { type: t(`common.hazardTypes.${item.hazard_type}`) })}
        </Text>
        <View style={styles.newsFooter}>
          <View style={styles.locationBadge}>
            <Text style={styles.locationText}>
              📍 {item.latitude.toFixed(4)}, {item.longitude.toFixed(4)}
            </Text>
          </View>
          <View style={styles.riskScoreBadge}>
            <Text style={styles.riskScoreText}>
              {t('news.riskScoreLabel', { score: item.risk_score })}
            </Text>
          </View>
        </View>
      </TouchableOpacity>
    );
  };

  return (
    <View style={styles.container}>
      {/* 위험 유형 필터 바 */}
      <View style={styles.filterContainer}>
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.filterScroll}
        >
          {HAZARD_TYPE_IDS.map((filterId) => (
            <TouchableOpacity
              key={filterId}
              style={[
                styles.filterChip,
                selectedFilter === filterId && styles.filterChipActive,
              ]}
              onPress={() => setSelectedFilter(filterId)}
            >
              <Text
                style={[
                  styles.filterText,
                  selectedFilter === filterId && styles.filterTextActive,
                ]}
              >
                {t(`common.hazardTypes.${filterId}`)}
              </Text>
            </TouchableOpacity>
          ))}
        </ScrollView>
      </View>

      {/* 뉴스 리스트 */}
      {loading ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={Colors.primary} />
          <Text style={styles.loadingText}>{t('news.loading')}</Text>
        </View>
      ) : (
        <FlatList
          data={filteredNews}
          renderItem={renderNewsItem}
          keyExtractor={(item) => item.id.toString()}
          contentContainerStyle={styles.listContent}
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={onRefresh}
              colors={[Colors.primary]}
              tintColor={Colors.primary}
            />
          }
          ListEmptyComponent={
            <View style={styles.emptyContainer}>
              <Text style={styles.emptyText}>
                {selectedFilter === 'all'
                  ? t('news.emptyAll')
                  : t('news.emptyFiltered', { type: t(`common.hazardTypes.${selectedFilter}`) })}
              </Text>
            </View>
          }
        />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  filterContainer: {
    backgroundColor: Colors.surface,
    paddingVertical: Spacing.md,
    paddingHorizontal: Spacing.md,
    flexDirection: 'row',
    alignItems: 'center',
    ...CommonStyles.shadowSmall,
  },
  filterLabel: {
    ...Typography.labelSmall,
    color: Colors.textSecondary,
    marginRight: Spacing.sm,
    fontWeight: '600',
  },
  filterScroll: {
    paddingHorizontal: Spacing.lg,
  },
  filterChip: {
    paddingHorizontal: Spacing.lg,
    paddingVertical: Spacing.sm,
    borderRadius: 20,
    backgroundColor: Colors.borderLight,
    marginRight: Spacing.sm,
  },
  filterChipActive: {
    backgroundColor: Colors.primary,
  },
  filterText: {
    ...Typography.buttonSmall,
    color: Colors.textSecondary,
  },
  filterTextActive: {
    color: Colors.textInverse,
  },
  listContent: {
    padding: Spacing.lg,
  },
  newsCard: {
    ...CommonStyles.card,
    marginBottom: Spacing.md,
  },
  newsHeader: {
    flexDirection: 'row',
    marginBottom: Spacing.sm,
  },
  riskBadge: {
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: Spacing.md,
  },
  riskText: {
    fontSize: 20,
  },
  newsMeta: {
    flex: 1,
  },
  newsTitle: {
    ...Typography.h3,
    color: Colors.textPrimary,
    marginBottom: Spacing.xs,
  },
  newsDate: {
    ...Typography.caption,
    color: Colors.textTertiary,
  },
  newsDescription: {
    ...Typography.bodyMedium,
    color: Colors.textSecondary,
    marginBottom: Spacing.md,
  },
  newsFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  locationBadge: {
    paddingHorizontal: Spacing.sm,
    paddingVertical: Spacing.xs,
    borderRadius: 12,
    backgroundColor: Colors.borderLight,
  },
  locationText: {
    ...Typography.caption,
    color: Colors.textSecondary,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: Spacing.xxl,
  },
  loadingText: {
    ...Typography.body,
    color: Colors.textSecondary,
    marginTop: Spacing.md,
  },
  riskScoreBadge: {
    paddingHorizontal: Spacing.sm,
    paddingVertical: Spacing.xs,
    borderRadius: 12,
    backgroundColor: Colors.primaryLight + '20',
  },
  riskScoreText: {
    ...Typography.labelSmall,
    color: Colors.primary,
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: Spacing.xxl,
  },
  emptyText: {
    ...Typography.bodyMedium,
    color: Colors.textSecondary,
  },
});
