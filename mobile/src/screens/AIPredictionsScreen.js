/**
 * AIPredictionsScreen - AI 기반 위험 예측 및 분석
 *
 * 기능:
 * - 향후 위험 예측 (1-7일)
 * - 이상 징후 감지
 * - 위험 핫스팟 예측
 * - NLP 분석 결과
 */

import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
  RefreshControl,
  Alert,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Colors, Spacing, Typography } from '../styles';
import api from '../services/api';
import Icon from '../components/icons/Icon';

export default function AIPredictionsScreen() {
  const insets = useSafeAreaInsets();
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [activeTab, setActiveTab] = useState('predictions'); // predictions, anomalies, hotspots

  const [predictions, setPredictions] = useState([]);
  const [anomalies, setAnomalies] = useState([]);
  const [hotspots, setHotspots] = useState([]);

  const loadData = async () => {
    try {
      const response = await api.get('/api/ai/analytics/overview');
      setPredictions(response.data.predictions || []);
      setAnomalies(response.data.anomalies || []);
      setHotspots(response.data.hotspots || []);
    } catch (error) {
      console.error('[AIPredictions] 데이터 로드 오류:', error);
      Alert.alert('오류', 'AI 예측 데이터를 불러올 수 없습니다.');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const onRefresh = () => {
    setRefreshing(true);
    loadData();
  };

  useEffect(() => {
    loadData();
  }, []);

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color={Colors.primary} />
        <Text style={styles.loadingText}>AI 분석 중...</Text>
      </View>
    );
  }

  const renderPredictions = () => (
    <View style={styles.content}>
      <Text style={styles.sectionTitle}>🔮 향후 7일 위험 예측</Text>
      <Text style={styles.sectionDescription}>
        과거 데이터 패턴 분석을 통한 예측입니다.
      </Text>

      {predictions.length === 0 ? (
        <View style={styles.emptyState}>
          <Icon name="info" size={48} color={Colors.textSecondary} />
          <Text style={styles.emptyText}>예측된 위험이 없습니다.</Text>
        </View>
      ) : (
        predictions.map((prediction, index) => (
          <View key={index} style={styles.predictionCard}>
            <View style={styles.predictionHeader}>
              <View style={styles.predictionLeft}>
                <Text style={styles.predictionDays}>+{prediction.days_ahead}일</Text>
                <Text style={styles.predictionDate}>
                  {new Date(prediction.predicted_date).toLocaleDateString('ko-KR', {
                    month: 'short',
                    day: 'numeric'
                  })}
                </Text>
              </View>
              <View style={[
                styles.probabilityBadge,
                {
                  backgroundColor:
                    prediction.probability >= 0.7
                      ? '#EF4444'
                      : prediction.probability >= 0.5
                      ? '#F59E0B'
                      : '#10B981'
                }
              ]}>
                <Text style={styles.probabilityText}>
                  {(prediction.probability * 100).toFixed(0)}%
                </Text>
              </View>
            </View>

            <View style={styles.predictionBody}>
              <View style={styles.predictionRow}>
                <Icon name="warning" size={20} color={Colors.error} />
                <Text style={styles.predictionType}>{prediction.hazard_type}</Text>
              </View>

              <View style={styles.predictionRow}>
                <Icon name="location" size={20} color={Colors.textSecondary} />
                <Text style={styles.predictionLocation}>
                  {prediction.latitude.toFixed(4)}, {prediction.longitude.toFixed(4)}
                </Text>
              </View>

              <Text style={styles.predictionRisk}>
                예상 위험도: {prediction.predicted_risk_score}점
              </Text>

              <Text style={styles.predictionReason}>{prediction.reasoning}</Text>
            </View>
          </View>
        ))
      )}
    </View>
  );

  const renderAnomalies = () => (
    <View style={styles.content}>
      <Text style={styles.sectionTitle}>⚠️ 이상 징후 감지</Text>
      <Text style={styles.sectionDescription}>
        최근 24시간 데이터에서 비정상적인 패턴이 감지되었습니다.
      </Text>

      {anomalies.length === 0 ? (
        <View style={styles.emptyState}>
          <Icon name="check-box" size={48} color={Colors.success} />
          <Text style={styles.emptyText}>이상 징후가 감지되지 않았습니다.</Text>
        </View>
      ) : (
        anomalies.map((anomaly, index) => (
          <View key={index} style={styles.anomalyCard}>
            <View style={styles.anomalyHeader}>
              <View style={[
                styles.severityBadge,
                {
                  backgroundColor:
                    anomaly.severity === 'high'
                      ? '#EF4444'
                      : anomaly.severity === 'medium'
                      ? '#F59E0B'
                      : '#10B981'
                }
              ]}>
                <Text style={styles.severityText}>
                  {anomaly.severity === 'high'
                    ? '높음'
                    : anomaly.severity === 'medium'
                    ? '보통'
                    : '낮음'}
                </Text>
              </View>
              <Text style={styles.anomalyTime}>
                {new Date(anomaly.detected_at).toLocaleTimeString('ko-KR', {
                  hour: '2-digit',
                  minute: '2-digit'
                })}
              </Text>
            </View>

            <Text style={styles.anomalyType}>{anomaly.type.replace('_', ' ')}</Text>
            <Text style={styles.anomalyDescription}>{anomaly.description}</Text>

            {anomaly.current_value !== undefined && (
              <View style={styles.anomalyStats}>
                <View style={styles.anomalyStat}>
                  <Text style={styles.anomalyStatLabel}>현재</Text>
                  <Text style={styles.anomalyStatValue}>{anomaly.current_value}</Text>
                </View>
                <Icon name="arrow-forward" size={20} color={Colors.textSecondary} />
                <View style={styles.anomalyStat}>
                  <Text style={styles.anomalyStatLabel}>평균</Text>
                  <Text style={styles.anomalyStatValue}>{anomaly.baseline_value}</Text>
                </View>
              </View>
            )}
          </View>
        ))
      )}
    </View>
  );

  const renderHotspots = () => (
    <View style={styles.content}>
      <Text style={styles.sectionTitle}>🔥 위험 핫스팟</Text>
      <Text style={styles.sectionDescription}>
        위험이 집중적으로 발생하는 지역입니다.
      </Text>

      {hotspots.length === 0 ? (
        <View style={styles.emptyState}>
          <Icon name="info" size={48} color={Colors.textSecondary} />
          <Text style={styles.emptyText}>핫스팟이 감지되지 않았습니다.</Text>
        </View>
      ) : (
        hotspots.map((hotspot, index) => (
          <View key={index} style={styles.hotspotCard}>
            <View style={styles.hotspotHeader}>
              <View style={styles.hotspotRank}>
                <Text style={styles.hotspotRankText}>#{index + 1}</Text>
              </View>
              <View style={[
                styles.confidenceBadge,
                {
                  backgroundColor:
                    hotspot.confidence === 'high' ? '#10B981' : '#F59E0B'
                }
              ]}>
                <Text style={styles.confidenceText}>
                  {hotspot.confidence === 'high' ? '신뢰도 높음' : '신뢰도 보통'}
                </Text>
              </View>
            </View>

            <View style={styles.hotspotBody}>
              <View style={styles.hotspotRow}>
                <Icon name="location" size={20} color={Colors.primary} />
                <Text style={styles.hotspotLocation}>
                  {hotspot.latitude.toFixed(4)}, {hotspot.longitude.toFixed(4)}
                </Text>
              </View>

              <View style={styles.hotspotStats}>
                <View style={styles.hotspotStat}>
                  <Text style={styles.hotspotStatLabel}>발생 건수</Text>
                  <Text style={styles.hotspotStatValue}>{hotspot.hazard_count}건</Text>
                </View>
                <View style={styles.hotspotStat}>
                  <Text style={styles.hotspotStatLabel}>평균 위험도</Text>
                  <Text style={styles.hotspotStatValue}>{hotspot.avg_risk_score}점</Text>
                </View>
                <View style={styles.hotspotStat}>
                  <Text style={styles.hotspotStatLabel}>주요 유형</Text>
                  <Text style={styles.hotspotStatValue}>{hotspot.most_common_type}</Text>
                </View>
              </View>
            </View>
          </View>
        ))
      )}
    </View>
  );

  return (
    <View style={[styles.container, { paddingTop: insets.top }]}>
      <Text style={styles.title}>AI 예측 및 분석</Text>

      {/* 탭 네비게이션 */}
      <View style={styles.tabs}>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'predictions' && styles.tabActive]}
          onPress={() => setActiveTab('predictions')}
        >
          <Text style={[styles.tabText, activeTab === 'predictions' && styles.tabTextActive]}>
            예측
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.tab, activeTab === 'anomalies' && styles.tabActive]}
          onPress={() => setActiveTab('anomalies')}
        >
          <Text style={[styles.tabText, activeTab === 'anomalies' && styles.tabTextActive]}>
            이상징후
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.tab, activeTab === 'hotspots' && styles.tabActive]}
          onPress={() => setActiveTab('hotspots')}
        >
          <Text style={[styles.tabText, activeTab === 'hotspots' && styles.tabTextActive]}>
            핫스팟
          </Text>
        </TouchableOpacity>
      </View>

      <ScrollView
        style={styles.scrollView}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
      >
        {activeTab === 'predictions' && renderPredictions()}
        {activeTab === 'anomalies' && renderAnomalies()}
        {activeTab === 'hotspots' && renderHotspots()}

        <View style={styles.infoBox}>
          <Icon name="info" size={20} color={Colors.info} />
          <Text style={styles.infoText}>
            AI 예측은 과거 데이터 패턴을 기반으로 합니다.{'\n'}
            실제 상황은 다를 수 있으니 참고용으로만 활용하세요.
          </Text>
        </View>
      </ScrollView>
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
  loadingText: {
    ...Typography.body,
    color: Colors.textSecondary,
    marginTop: Spacing.md,
  },
  title: {
    ...Typography.h1,
    color: Colors.textPrimary,
    marginHorizontal: Spacing.lg,
    marginBottom: Spacing.md,
  },
  tabs: {
    flexDirection: 'row',
    paddingHorizontal: Spacing.lg,
    marginBottom: Spacing.md,
    gap: Spacing.sm,
  },
  tab: {
    flex: 1,
    paddingVertical: Spacing.md,
    borderRadius: 12,
    backgroundColor: Colors.surface,
    alignItems: 'center',
  },
  tabActive: {
    backgroundColor: Colors.primary,
  },
  tabText: {
    ...Typography.button,
    color: Colors.textSecondary,
  },
  tabTextActive: {
    color: Colors.textInverse,
  },
  scrollView: {
    flex: 1,
  },
  content: {
    padding: Spacing.lg,
  },
  sectionTitle: {
    ...Typography.h2,
    color: Colors.textPrimary,
    marginBottom: Spacing.xs,
  },
  sectionDescription: {
    ...Typography.bodySmall,
    color: Colors.textSecondary,
    marginBottom: Spacing.lg,
  },
  emptyState: {
    alignItems: 'center',
    padding: Spacing.xxxl,
  },
  emptyText: {
    ...Typography.body,
    color: Colors.textSecondary,
    marginTop: Spacing.md,
  },
  predictionCard: {
    backgroundColor: Colors.surfaceElevated,
    borderRadius: 12,
    padding: Spacing.lg,
    marginBottom: Spacing.md,
  },
  predictionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: Spacing.md,
  },
  predictionLeft: {
    flexDirection: 'row',
    alignItems: 'baseline',
    gap: Spacing.sm,
  },
  predictionDays: {
    fontSize: 24,
    fontWeight: 'bold',
    color: Colors.primary,
  },
  predictionDate: {
    ...Typography.bodySmall,
    color: Colors.textSecondary,
  },
  probabilityBadge: {
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.sm,
    borderRadius: 12,
  },
  probabilityText: {
    ...Typography.labelSmall,
    color: Colors.textInverse,
    fontWeight: 'bold',
  },
  predictionBody: {
    gap: Spacing.sm,
  },
  predictionRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
  },
  predictionType: {
    ...Typography.body,
    color: Colors.textPrimary,
    fontWeight: '600',
  },
  predictionLocation: {
    ...Typography.bodySmall,
    color: Colors.textSecondary,
  },
  predictionRisk: {
    ...Typography.body,
    color: Colors.textPrimary,
  },
  predictionReason: {
    ...Typography.bodySmall,
    color: Colors.textSecondary,
    marginTop: Spacing.sm,
    fontStyle: 'italic',
  },
  anomalyCard: {
    backgroundColor: Colors.surfaceElevated,
    borderRadius: 12,
    padding: Spacing.lg,
    marginBottom: Spacing.md,
    borderLeftWidth: 4,
    borderLeftColor: Colors.error,
  },
  anomalyHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: Spacing.md,
  },
  severityBadge: {
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.sm,
    borderRadius: 12,
  },
  severityText: {
    ...Typography.labelSmall,
    color: Colors.textInverse,
    fontWeight: 'bold',
  },
  anomalyTime: {
    ...Typography.bodySmall,
    color: Colors.textSecondary,
  },
  anomalyType: {
    ...Typography.h3,
    color: Colors.textPrimary,
    marginBottom: Spacing.sm,
    textTransform: 'capitalize',
  },
  anomalyDescription: {
    ...Typography.body,
    color: Colors.textSecondary,
    marginBottom: Spacing.md,
  },
  anomalyStats: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-around',
    paddingTop: Spacing.md,
    borderTopWidth: 1,
    borderTopColor: Colors.border,
  },
  anomalyStat: {
    alignItems: 'center',
  },
  anomalyStatLabel: {
    ...Typography.labelSmall,
    color: Colors.textSecondary,
    marginBottom: Spacing.xs,
  },
  anomalyStatValue: {
    ...Typography.body,
    color: Colors.textPrimary,
    fontWeight: 'bold',
  },
  hotspotCard: {
    backgroundColor: Colors.surfaceElevated,
    borderRadius: 12,
    padding: Spacing.lg,
    marginBottom: Spacing.md,
  },
  hotspotHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: Spacing.md,
  },
  hotspotRank: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: Colors.primary + '20',
    justifyContent: 'center',
    alignItems: 'center',
  },
  hotspotRankText: {
    ...Typography.body,
    color: Colors.primary,
    fontWeight: 'bold',
  },
  confidenceBadge: {
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.sm,
    borderRadius: 12,
  },
  confidenceText: {
    ...Typography.labelSmall,
    color: Colors.textInverse,
    fontWeight: '600',
  },
  hotspotBody: {
    gap: Spacing.md,
  },
  hotspotRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
  },
  hotspotLocation: {
    ...Typography.body,
    color: Colors.textPrimary,
  },
  hotspotStats: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingTop: Spacing.md,
    borderTopWidth: 1,
    borderTopColor: Colors.border,
  },
  hotspotStat: {
    flex: 1,
    alignItems: 'center',
  },
  hotspotStatLabel: {
    ...Typography.labelSmall,
    color: Colors.textSecondary,
    marginBottom: Spacing.xs,
  },
  hotspotStatValue: {
    ...Typography.body,
    color: Colors.textPrimary,
    fontWeight: 'bold',
  },
  infoBox: {
    flexDirection: 'row',
    backgroundColor: Colors.info + '10',
    borderRadius: 12,
    padding: Spacing.md,
    margin: Spacing.lg,
    gap: Spacing.sm,
  },
  infoText: {
    ...Typography.bodySmall,
    color: Colors.textSecondary,
    flex: 1,
  },
});
