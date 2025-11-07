/**
 * DataDashboardScreen - 외부 데이터 수집 대시보드
 *
 * 기능:
 * - 데이터 수집 통계 확인
 * - 소스별 데이터 현황
 * - 수동 데이터 수집 트리거
 */

import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  ActivityIndicator,
  RefreshControl,
  Alert,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Colors, Spacing, Typography } from '../styles';
import api from '../services/api';
import Icon from '../components/icons/Icon';

export default function DataDashboardScreen() {
  const insets = useSafeAreaInsets();
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [collecting, setCollecting] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  const loadStats = async () => {
    try {
      const response = await api.get('/api/data/dashboard/stats');
      setStats(response.data);
    } catch (error) {
      console.error('[Dashboard] 통계 로드 오류:', error);
      Alert.alert('오류', '통계를 불러올 수 없습니다.');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const triggerCollection = async () => {
    setCollecting(true);
    try {
      const response = await api.post('/api/data/dashboard/trigger-collection');
      const stats = response.data.statistics;
      Alert.alert(
        '수집 완료',
        `${stats.total}개 항목이 수집되었습니다.\n\n` +
        `ACLED: ${stats.acled}개\n` +
        `GDACS: ${stats.gdacs}개\n` +
        `ReliefWeb: ${stats.reliefweb}개\n` +
        `Twitter: ${stats.twitter}개\n` +
        `News: ${stats.news}개\n` +
        `Sentinel: ${stats.sentinel}개`,
        [{ text: '확인', onPress: () => loadStats() }]
      );
    } catch (error) {
      console.error('[Dashboard] 수집 오류:', error);
      Alert.alert('오류', '데이터 수집에 실패했습니다.');
    } finally {
      setCollecting(false);
    }
  };

  const onRefresh = () => {
    setRefreshing(true);
    loadStats();
  };

  useEffect(() => {
    loadStats();
  }, []);

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color={Colors.primary} />
        <Text style={styles.loadingText}>통계를 불러오는 중...</Text>
      </View>
    );
  }

  const getSourceColor = (source) => {
    const colors = {
      acled: '#EF4444',
      gdacs: '#F59E0B',
      reliefweb: '#10B981',
      twitter: '#1DA1F2',
      news: '#8B5CF6',
      sentinel: '#0EA5E9',
    };
    return colors[source] || Colors.textSecondary;
  };

  const getSourceDescription = (source) => {
    const descriptions = {
      acled: '분쟁 및 폭력 사건',
      gdacs: '재난 및 자연재해',
      reliefweb: '인도적 지원 보고서',
      twitter: '소셜 미디어 모니터링',
      news: '뉴스 기사 분석',
      sentinel: '위성 이미지 분석',
    };
    return descriptions[source] || '';
  };

  return (
    <ScrollView
      style={[styles.container, { paddingTop: insets.top }]}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
      }
    >
      <Text style={styles.title}>데이터 수집 현황</Text>

      {/* 전체 통계 카드 */}
      <View style={styles.mainCard}>
        <Icon name="warning" size={48} color={Colors.primary + '40'} />
        <Text style={styles.bigNumber}>{stats?.total_hazards || 0}</Text>
        <Text style={styles.subtitle}>총 수집된 위험 정보</Text>
        <View style={styles.recentBadge}>
          <Text style={styles.recentText}>최근 24시간: {stats?.recent_24h || 0}개</Text>
        </View>
      </View>

      {/* 소스별 통계 */}
      <Text style={styles.sectionTitle}>📡 데이터 소스</Text>

      {Object.entries(stats?.sources || {}).map(([source, data]) => (
        <View key={source} style={styles.sourceCard}>
          <View style={styles.sourceHeader}>
            <View style={styles.sourceLeft}>
              <View style={[styles.sourceIcon, { backgroundColor: getSourceColor(source) + '20' }]}>
                <Icon name="database" size={24} color={getSourceColor(source)} />
              </View>
              <View>
                <Text style={styles.sourceName}>{source.toUpperCase()}</Text>
                <Text style={styles.sourceDesc}>{getSourceDescription(source)}</Text>
              </View>
            </View>
            <View style={[
              styles.statusBadge,
              { backgroundColor: data.status === 'active' ? '#10B981' : '#EF4444' }
            ]}>
              <Text style={styles.statusText}>{data.status === 'active' ? '활성' : '비활성'}</Text>
            </View>
          </View>

          <View style={styles.sourceStats}>
            <View style={styles.statItem}>
              <Text style={styles.statLabel}>수집 항목</Text>
              <Text style={styles.statValue}>{data.count}개</Text>
            </View>
            <View style={styles.statDivider} />
            <View style={styles.statItem}>
              <Text style={styles.statLabel}>최근 업데이트</Text>
              <Text style={styles.statValue}>
                {data.last_updated ? new Date(data.last_updated).toLocaleDateString('ko-KR') : '없음'}
              </Text>
            </View>
          </View>
        </View>
      ))}

      {/* 위험도 분포 */}
      <Text style={styles.sectionTitle}>⚠️ 위험도 분포</Text>
      <View style={styles.riskCard}>
        <View style={styles.riskRow}>
          <View style={styles.riskLeft}>
            <View style={[styles.riskDot, { backgroundColor: '#10B981' }]} />
            <Text style={styles.riskLabel}>낮음 (0-39)</Text>
          </View>
          <Text style={styles.riskValue}>{stats?.risk_distribution?.low || 0}개</Text>
        </View>
        <View style={styles.riskRow}>
          <View style={styles.riskLeft}>
            <View style={[styles.riskDot, { backgroundColor: '#F59E0B' }]} />
            <Text style={styles.riskLabel}>보통 (40-69)</Text>
          </View>
          <Text style={styles.riskValue}>{stats?.risk_distribution?.medium || 0}개</Text>
        </View>
        <View style={styles.riskRow}>
          <View style={styles.riskLeft}>
            <View style={[styles.riskDot, { backgroundColor: '#EF4444' }]} />
            <Text style={styles.riskLabel}>높음 (70-100)</Text>
          </View>
          <Text style={styles.riskValue}>{stats?.risk_distribution?.high || 0}개</Text>
        </View>
      </View>

      {/* 위험 유형 분포 */}
      {Object.keys(stats?.type_distribution || {}).length > 0 && (
        <>
          <Text style={styles.sectionTitle}>📊 위험 유형</Text>
          <View style={styles.typeCard}>
            {Object.entries(stats.type_distribution).map(([type, count]) => (
              <View key={type} style={styles.typeRow}>
                <Text style={styles.typeLabel}>{type}</Text>
                <Text style={styles.typeValue}>{count}개</Text>
              </View>
            ))}
          </View>
        </>
      )}

      {/* 수동 수집 버튼 */}
      <TouchableOpacity
        style={[styles.collectButton, collecting && styles.collectButtonDisabled]}
        onPress={triggerCollection}
        disabled={collecting}
        activeOpacity={0.8}
      >
        {collecting ? (
          <ActivityIndicator color={Colors.textInverse} />
        ) : (
          <>
            <Icon name="refresh" size={24} color={Colors.textInverse} />
            <Text style={styles.collectButtonText}>지금 데이터 수집하기</Text>
          </>
        )}
      </TouchableOpacity>

      <Text style={styles.lastCheck}>
        마지막 확인: {stats?.last_check ? new Date(stats.last_check).toLocaleString('ko-KR') : ''}
      </Text>

      <View style={styles.infoBox}>
        <Icon name="info" size={20} color={Colors.info} />
        <Text style={styles.infoText}>
          데이터는 24시간마다 자동으로 수집됩니다.{'\n'}
          필요시 수동으로 즉시 수집할 수 있습니다.
        </Text>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
    padding: Spacing.lg,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: Colors.background,
  },
  loadingText: {
    ...Typography.body,
    color: Colors.textSecondary,
    marginTop: Spacing.md,
  },
  title: {
    ...Typography.h1,
    color: Colors.textPrimary,
    marginBottom: Spacing.xl,
  },
  mainCard: {
    backgroundColor: Colors.surfaceElevated,
    borderRadius: 16,
    padding: Spacing.xl,
    marginBottom: Spacing.xl,
    alignItems: 'center',
    shadowColor: Colors.shadowDark,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  bigNumber: {
    fontSize: 56,
    fontWeight: 'bold',
    color: Colors.primary,
    marginTop: Spacing.sm,
    marginBottom: Spacing.xs,
  },
  subtitle: {
    ...Typography.body,
    color: Colors.textSecondary,
    marginBottom: Spacing.md,
  },
  recentBadge: {
    backgroundColor: Colors.accent + '20',
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.sm,
    borderRadius: 20,
  },
  recentText: {
    ...Typography.labelSmall,
    color: Colors.accentDark,
    fontWeight: '600',
  },
  sectionTitle: {
    ...Typography.h2,
    color: Colors.textPrimary,
    marginTop: Spacing.xl,
    marginBottom: Spacing.md,
  },
  sourceCard: {
    backgroundColor: Colors.surfaceElevated,
    borderRadius: 12,
    padding: Spacing.lg,
    marginBottom: Spacing.md,
  },
  sourceHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: Spacing.md,
  },
  sourceLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  sourceIcon: {
    width: 48,
    height: 48,
    borderRadius: 24,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: Spacing.md,
  },
  sourceName: {
    ...Typography.h3,
    color: Colors.textPrimary,
    fontWeight: 'bold',
    marginBottom: Spacing.xs,
  },
  sourceDesc: {
    ...Typography.bodySmall,
    color: Colors.textSecondary,
  },
  statusBadge: {
    paddingHorizontal: Spacing.sm,
    paddingVertical: Spacing.xs,
    borderRadius: 12,
  },
  statusText: {
    ...Typography.labelSmall,
    color: Colors.textInverse,
    fontWeight: '600',
  },
  sourceStats: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: Spacing.sm,
  },
  statItem: {
    flex: 1,
  },
  statDivider: {
    width: 1,
    height: 30,
    backgroundColor: Colors.border,
    marginHorizontal: Spacing.md,
  },
  statLabel: {
    ...Typography.labelSmall,
    color: Colors.textSecondary,
    marginBottom: Spacing.xs,
  },
  statValue: {
    ...Typography.body,
    color: Colors.textPrimary,
    fontWeight: '600',
  },
  riskCard: {
    backgroundColor: Colors.surfaceElevated,
    borderRadius: 12,
    padding: Spacing.lg,
    marginBottom: Spacing.md,
  },
  riskRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: Spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: Colors.borderLight,
  },
  riskLeft: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  riskDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: Spacing.sm,
  },
  riskLabel: {
    ...Typography.body,
    color: Colors.textPrimary,
  },
  riskValue: {
    ...Typography.body,
    color: Colors.primary,
    fontWeight: '700',
  },
  typeCard: {
    backgroundColor: Colors.surfaceElevated,
    borderRadius: 12,
    padding: Spacing.lg,
    marginBottom: Spacing.md,
  },
  typeRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: Spacing.sm,
  },
  typeLabel: {
    ...Typography.body,
    color: Colors.textPrimary,
  },
  typeValue: {
    ...Typography.body,
    color: Colors.textSecondary,
    fontWeight: '600',
  },
  collectButton: {
    flexDirection: 'row',
    backgroundColor: Colors.primary,
    borderRadius: 12,
    padding: Spacing.lg,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: Spacing.xl,
    marginBottom: Spacing.md,
    gap: Spacing.sm,
  },
  collectButtonDisabled: {
    opacity: 0.5,
  },
  collectButtonText: {
    ...Typography.button,
    color: Colors.textInverse,
  },
  lastCheck: {
    ...Typography.bodySmall,
    color: Colors.textSecondary,
    textAlign: 'center',
    marginBottom: Spacing.lg,
  },
  infoBox: {
    flexDirection: 'row',
    backgroundColor: Colors.info + '10',
    borderRadius: 12,
    padding: Spacing.md,
    marginBottom: Spacing.xxxl,
    gap: Spacing.sm,
  },
  infoText: {
    ...Typography.bodySmall,
    color: Colors.textSecondary,
    flex: 1,
  },
});
