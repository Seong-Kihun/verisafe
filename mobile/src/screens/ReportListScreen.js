/**
 * 제보 목록 화면
 * Google Maps 스타일의 깔끔한 리스트
 */

import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  RefreshControl,
  ActivityIndicator,
} from 'react-native';
import { Colors, Typography, Spacing, CommonStyles } from '../styles';
import { reportAPI } from '../services/api';
import { useNavigation } from '@react-navigation/native';
import Icon from '../components/icons/Icon';

const HAZARD_ICONS = {
  checkpoint: 'checkpoint',
  protest_riot: 'protest',
  road_damage: 'roadDamage',
  armed_conflict: 'conflict',
  natural_disaster: 'naturalDisaster',
  other: 'other',
};

const STATUS_COLORS = {
  pending: Colors.warning,
  verified: Colors.success,
  rejected: Colors.danger,
};

export default function ReportListScreen() {
  const navigation = useNavigation();
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    loadReports();
  }, []);

  const loadReports = async () => {
    try {
      const response = await reportAPI.list(4.8594, 31.5713, 15.0);
      setReports(response.data.reports || []);
    } catch (error) {
      console.error('Failed to load reports:', error);
    } finally {
      setLoading(false);
    }
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await loadReports();
    setRefreshing(false);
  };

  const renderItem = ({ item }) => (
    <TouchableOpacity style={styles.card}>
      <View style={styles.cardHeader}>
        <View style={styles.cardIconContainer}>
          <Icon 
            name={HAZARD_ICONS[item.hazard_type] || 'other'} 
            size={24} 
            color={Colors.primary} 
          />
        </View>
        <View style={styles.cardTitleContainer}>
          <Text style={styles.cardTitle}>
            {(() => {
              const hazardTypeLabels = {
                'armed_conflict': '무력충돌',
                'protest_riot': '시위/폭동',
                'checkpoint': '검문소',
                'road_damage': '도로 손상',
                'natural_disaster': '자연재해',
                'other': '기타 위험',
              };
              return hazardTypeLabels[item.hazard_type] || item.hazard_type;
            })()}
          </Text>
          <View style={[
            styles.statusBadge,
            { backgroundColor: STATUS_COLORS[item.status] || Colors.textSecondary }
          ]}>
            <Text style={styles.statusText}>
              {item.status === 'pending' ? '대기중' :
               item.status === 'verified' ? '검증됨' : '거부됨'}
            </Text>
          </View>
        </View>
      </View>

      {item.description && (
        <Text style={styles.cardDescription} numberOfLines={2}>
          {item.description}
        </Text>
      )}

      <View style={styles.cardFooter}>
        <View style={styles.cardLocationContainer}>
          <Icon name="location" size={14} color={Colors.textSecondary} />
          <Text style={styles.cardLocation}>
            {item.latitude.toFixed(4)}, {item.longitude.toFixed(4)}
          </Text>
        </View>
        <Text style={styles.cardDate}>
          {new Date(item.created_at).toLocaleDateString()}
        </Text>
      </View>
    </TouchableOpacity>
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
      <FlatList
        data={reports}
        renderItem={renderItem}
        keyExtractor={(item) => item.id}
        contentContainerStyle={styles.listContent}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={onRefresh}
            colors={[Colors.primary]}
          />
        }
        ListEmptyComponent={
          <View style={styles.emptyContainer}>
            <Icon name="report" size={48} color={Colors.textTertiary} />
            <Text style={styles.emptyTitle}>제보가 없습니다</Text>
            <Text style={styles.emptyDescription}>
              첫 번째 위험 제보를 등록해보세요.
            </Text>
          </View>
        }
      />

      {/* FAB 버튼 */}
      <TouchableOpacity
        style={styles.fab}
        onPress={() => navigation.navigate('ReportCreate')}
        activeOpacity={0.8}
      >
        <Icon name="report" size={28} color={Colors.textInverse} />
      </TouchableOpacity>
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
    backgroundColor: Colors.background,
    justifyContent: 'center',
    alignItems: 'center',
  },
  listContent: {
    padding: Spacing.lg,
  },
  card: {
    backgroundColor: Colors.surface,
    borderRadius: 16,
    padding: Spacing.lg,
    marginBottom: Spacing.md,
    shadowColor: Colors.shadow,
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: Spacing.sm,
  },
  cardIconContainer: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: Colors.primary + '20',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: Spacing.md,
  },
  cardTitleContainer: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  cardTitle: {
    ...Typography.h3,
    color: Colors.textPrimary,
  },
  statusBadge: {
    paddingHorizontal: Spacing.sm,
    paddingVertical: Spacing.xs,
    borderRadius: 12,
  },
  statusText: {
    ...Typography.labelSmall,
    color: Colors.textInverse,
  },
  cardDescription: {
    ...Typography.bodyMedium,
    color: Colors.textSecondary,
    marginBottom: Spacing.sm,
  },
  cardFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  cardLocationContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.xs,
  },
  cardLocation: {
    ...Typography.captionSmall,
    color: Colors.textTertiary,
  },
  cardDate: {
    ...Typography.captionSmall,
    color: Colors.textTertiary,
  },
  emptyContainer: {
    padding: Spacing.xxxl,
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 300,
  },
  emptyTitle: {
    ...Typography.h3,
    color: Colors.textPrimary,
    marginTop: Spacing.md,
    marginBottom: Spacing.xs,
  },
  emptyDescription: {
    ...Typography.body,
    color: Colors.textSecondary,
    textAlign: 'center',
    lineHeight: 24,
  },
  emptyText: {
    ...Typography.body,
    color: Colors.textSecondary,
  },
  fab: {
    position: 'absolute',
    bottom: Spacing.xl,
    right: Spacing.xl,
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: Colors.primary,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: Colors.shadowDark,
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.4,
    shadowRadius: 8,
    elevation: 6,
    zIndex: 1000,
  },
});
