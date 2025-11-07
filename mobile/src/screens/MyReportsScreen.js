/**
 * 나의 제보 목록 화면
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
} from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { Colors, Typography, Spacing } from '../styles';
import { myReportsStorage } from '../services/storage';
import Icon from '../components/icons/Icon';

const HAZARD_TYPES = {
  checkpoint: { label: '검문소', icon: 'checkpoint', color: '#FF6B6B' },
  protest_riot: { label: '시위/폭동', icon: 'protest', color: '#F59E0B' },
  road_damage: { label: '도로 파손', icon: 'roadDamage', color: '#F97316' },
  armed_conflict: { label: '무력충돌', icon: 'conflict', color: '#EF4444' },
  natural_disaster: { label: '자연재해', icon: 'naturalDisaster', color: '#DC2626' },
  other: { label: '기타', icon: 'other', color: '#6B7280' },
};

const STATUS_LABELS = {
  pending: { label: '대기중', color: Colors.warning },
  verified: { label: '검증됨', color: Colors.success },
  rejected: { label: '거부됨', color: Colors.danger },
};

const formatDate = (dateString) => {
  const date = new Date(dateString);
  return date.toLocaleDateString('ko-KR', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};

export default function MyReportsScreen({ navigation }) {
  const [loading, setLoading] = useState(true);
  const [reports, setReports] = useState([]);

  useFocusEffect(
    useCallback(() => {
      loadReports();
    }, [])
  );

  const loadReports = async () => {
    try {
      const data = await myReportsStorage.getAll();
      setReports(data);
    } catch (error) {
      console.error('Failed to load reports:', error);
      Alert.alert('오류', '제보 목록을 불러올 수 없습니다.');
    } finally {
      setLoading(false);
    }
  };

  const handleViewOnMap = (report) => {
    // MapStack의 Map 화면으로 이동하고 해당 위치를 표시
    navigation.navigate('MapStack', {
      screen: 'Map',
      params: {
        selectedLocation: {
          latitude: report.latitude,
          longitude: report.longitude,
        },
      },
    });
  };

  const renderItem = ({ item }) => {
    const hazardType = HAZARD_TYPES[item.hazard_type] || HAZARD_TYPES.other;
    const status = STATUS_LABELS[item.status] || STATUS_LABELS.pending;

    return (
      <View style={styles.reportCard}>
        <View style={styles.reportHeader}>
          <View style={[styles.iconContainer, { backgroundColor: hazardType.color + '20' }]}>
            <Icon name={hazardType.icon} size={24} color={hazardType.color} />
          </View>
          <View style={styles.reportInfo}>
            <Text style={styles.hazardType}>{hazardType.label}</Text>
            <Text style={styles.reportDate}>{formatDate(item.createdAt)}</Text>
          </View>
          <View style={[styles.statusBadge, { backgroundColor: status.color + '20' }]}>
            <Text style={[styles.statusText, { color: status.color }]}>
              {status.label}
            </Text>
          </View>
        </View>

        {item.description && (
          <Text style={styles.description} numberOfLines={2}>
            {item.description}
          </Text>
        )}

        <View style={styles.locationInfo}>
          <Icon name="location" size={16} color={Colors.textSecondary} />
          <Text style={styles.coordinates}>
            {item.latitude.toFixed(4)}, {item.longitude.toFixed(4)}
          </Text>
        </View>

        <TouchableOpacity
          style={styles.viewMapButton}
          onPress={() => handleViewOnMap(item)}
        >
          <Icon name="map" size={18} color={Colors.primary} />
          <Text style={styles.viewMapText}>지도에서 보기</Text>
        </TouchableOpacity>
      </View>
    );
  };

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color={Colors.primary} />
      </View>
    );
  }

  if (reports.length === 0) {
    return (
      <View style={styles.emptyContainer}>
        <Icon name="report" size={64} color={Colors.textTertiary} />
        <Text style={styles.emptyTitle}>제보 내역이 없습니다</Text>
        <Text style={styles.emptyText}>
          위험 상황을 발견하면{'\n'}제보해주세요
        </Text>
        <TouchableOpacity
          style={styles.goToReportButton}
          onPress={() => navigation.navigate('ReportStack')}
        >
          <Icon name="report" size={20} color={Colors.textInverse} />
          <Text style={styles.goToReportButtonText}>제보하기</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>총 {reports.length}개의 제보</Text>
        <View style={styles.stats}>
          <View style={styles.statItem}>
            <Text style={styles.statValue}>
              {reports.filter(r => r.status === 'verified').length}
            </Text>
            <Text style={styles.statLabel}>검증됨</Text>
          </View>
          <View style={styles.statItem}>
            <Text style={styles.statValue}>
              {reports.filter(r => r.status === 'pending').length}
            </Text>
            <Text style={styles.statLabel}>대기중</Text>
          </View>
        </View>
      </View>
      <FlatList
        data={reports}
        renderItem={renderItem}
        keyExtractor={item => item.id}
        contentContainerStyle={styles.listContainer}
      />
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
  },
  goToReportButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.xs,
    backgroundColor: Colors.primary,
    paddingHorizontal: Spacing.lg,
    paddingVertical: Spacing.md,
    borderRadius: 12,
  },
  goToReportButtonText: {
    ...Typography.button,
    color: Colors.textInverse,
    fontWeight: '600',
  },
  header: {
    padding: Spacing.lg,
    backgroundColor: Colors.surface,
    borderBottomWidth: 1,
    borderBottomColor: Colors.border,
  },
  headerTitle: {
    ...Typography.h3,
    color: Colors.textPrimary,
    marginBottom: Spacing.md,
  },
  stats: {
    flexDirection: 'row',
    gap: Spacing.xl,
  },
  statItem: {
    alignItems: 'center',
  },
  statValue: {
    ...Typography.h2,
    color: Colors.primary,
    fontSize: 24,
  },
  statLabel: {
    ...Typography.captionSmall,
    color: Colors.textSecondary,
    marginTop: Spacing.xs,
  },
  listContainer: {
    padding: Spacing.md,
  },
  reportCard: {
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
  reportHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: Spacing.md,
  },
  iconContainer: {
    width: 48,
    height: 48,
    borderRadius: 24,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: Spacing.md,
  },
  reportInfo: {
    flex: 1,
  },
  hazardType: {
    ...Typography.h3,
    color: Colors.textPrimary,
    marginBottom: Spacing.xs,
  },
  reportDate: {
    ...Typography.captionSmall,
    color: Colors.textSecondary,
  },
  statusBadge: {
    paddingHorizontal: Spacing.sm,
    paddingVertical: Spacing.xs,
    borderRadius: 8,
  },
  statusText: {
    ...Typography.labelSmall,
    fontWeight: '600',
  },
  description: {
    ...Typography.body,
    color: Colors.textSecondary,
    marginBottom: Spacing.md,
    lineHeight: 20,
  },
  locationInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.xs,
    marginBottom: Spacing.md,
  },
  coordinates: {
    ...Typography.bodySmall,
    color: Colors.textSecondary,
  },
  viewMapButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: Spacing.xs,
    backgroundColor: Colors.primaryLight + '20',
    paddingVertical: Spacing.sm,
    borderRadius: 12,
  },
  viewMapText: {
    ...Typography.labelSmall,
    color: Colors.primary,
    fontWeight: '600',
  },
});
