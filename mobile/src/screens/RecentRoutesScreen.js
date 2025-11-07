/**
 * 최근 경로 기록 화면
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
import { recentRoutesStorage } from '../services/storage';
import Icon from '../components/icons/Icon';

const formatDate = (dateString) => {
  const date = new Date(dateString);
  const now = new Date();
  const diff = now - date;
  const days = Math.floor(diff / (1000 * 60 * 60 * 24));

  if (days === 0) {
    const hours = Math.floor(diff / (1000 * 60 * 60));
    if (hours === 0) {
      const minutes = Math.floor(diff / (1000 * 60));
      return `${minutes}분 전`;
    }
    return `${hours}시간 전`;
  } else if (days === 1) {
    return '어제';
  } else if (days < 7) {
    return `${days}일 전`;
  } else {
    return date.toLocaleDateString('ko-KR', {
      month: 'short',
      day: 'numeric'
    });
  }
};

export default function RecentRoutesScreen({ navigation }) {
  const [loading, setLoading] = useState(true);
  const [routes, setRoutes] = useState([]);

  useFocusEffect(
    useCallback(() => {
      loadRoutes();
    }, [])
  );

  const loadRoutes = async () => {
    try {
      const data = await recentRoutesStorage.getAll();
      setRoutes(data);
    } catch (error) {
      console.error('Failed to load routes:', error);
      Alert.alert('오류', '최근 경로를 불러올 수 없습니다.');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = (routeId) => {
    Alert.alert(
      '삭제 확인',
      '이 경로를 삭제하시겠습니까?',
      [
        { text: '취소', style: 'cancel' },
        {
          text: '삭제',
          style: 'destructive',
          onPress: async () => {
            const success = await recentRoutesStorage.remove(routeId);
            if (success) {
              setRoutes(prev => prev.filter(r => r.id !== routeId));
            } else {
              Alert.alert('오류', '삭제에 실패했습니다.');
            }
          },
        },
      ]
    );
  };

  const handleClearAll = () => {
    if (routes.length === 0) return;

    Alert.alert(
      '전체 삭제',
      '모든 최근 경로를 삭제하시겠습니까?',
      [
        { text: '취소', style: 'cancel' },
        {
          text: '삭제',
          style: 'destructive',
          onPress: async () => {
            const success = await recentRoutesStorage.clear();
            if (success) {
              setRoutes([]);
            } else {
              Alert.alert('오류', '삭제에 실패했습니다.');
            }
          },
        },
      ]
    );
  };

  const handleUseRoute = (route) => {
    // MapStack의 RoutePlanning 화면으로 이동하고 출발지/도착지 설정
    navigation.navigate('MapStack', {
      screen: 'RoutePlanning',
      params: {
        origin: route.start,
        destination: route.end,
      },
    });
  };

  const renderItem = ({ item }) => (
    <View style={styles.routeCard}>
      <View style={styles.routeHeader}>
        <View style={styles.routeIconContainer}>
          <Icon name="directions" size={24} color={Colors.primary} />
        </View>
        <View style={styles.routeInfo}>
          <View style={styles.locationRow}>
            <View style={[styles.dot, { backgroundColor: Colors.primary }]} />
            <Text style={styles.locationText} numberOfLines={1}>
              {item.start.name || `${item.start.latitude.toFixed(4)}, ${item.start.longitude.toFixed(4)}`}
            </Text>
          </View>
          <View style={styles.locationRow}>
            <View style={[styles.dot, { backgroundColor: Colors.danger }]} />
            <Text style={styles.locationText} numberOfLines={1}>
              {item.end.name || `${item.end.latitude.toFixed(4)}, ${item.end.longitude.toFixed(4)}`}
            </Text>
          </View>
        </View>
      </View>

      <View style={styles.routeMeta}>
        <Text style={styles.routeTime}>{formatDate(item.searchedAt)}</Text>
        {item.distance && (
          <Text style={styles.routeDistance}>{(item.distance / 1000).toFixed(1)} km</Text>
        )}
      </View>

      <View style={styles.routeActions}>
        <TouchableOpacity
          style={styles.actionButton}
          onPress={() => handleUseRoute(item)}
        >
          <Icon name="directions" size={20} color={Colors.primary} />
          <Text style={styles.actionButtonText}>경로 찾기</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.actionButton, styles.deleteButton]}
          onPress={() => handleDelete(item.id)}
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
      {routes.length === 0 ? (
        <View style={styles.emptyContainer}>
          <Icon name="history" size={64} color={Colors.textTertiary} />
          <Text style={styles.emptyTitle}>최근 경로가 없습니다</Text>
          <Text style={styles.emptyText}>
            경로를 검색하면{'\n'}여기에 기록됩니다
          </Text>
          <TouchableOpacity
            style={styles.goToMapButton}
            onPress={() => navigation.navigate('MapStack')}
          >
            <Icon name="directions" size={20} color={Colors.textInverse} />
            <Text style={styles.goToMapButtonText}>경로 찾기</Text>
          </TouchableOpacity>
        </View>
      ) : (
        <>
          <View style={styles.header}>
            <Text style={styles.headerTitle}>최근 경로 {routes.length}개</Text>
            <TouchableOpacity onPress={handleClearAll}>
              <Text style={styles.clearAllText}>전체 삭제</Text>
            </TouchableOpacity>
          </View>
          <FlatList
            data={routes}
            renderItem={renderItem}
            keyExtractor={item => item.id}
            contentContainerStyle={styles.listContainer}
          />
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
  },
  goToMapButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.xs,
    backgroundColor: Colors.primary,
    paddingHorizontal: Spacing.lg,
    paddingVertical: Spacing.md,
    borderRadius: 12,
  },
  goToMapButtonText: {
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
  headerTitle: {
    ...Typography.h3,
    color: Colors.textPrimary,
  },
  clearAllText: {
    ...Typography.bodySmall,
    color: Colors.danger,
    fontWeight: '600',
  },
  listContainer: {
    padding: Spacing.md,
  },
  routeCard: {
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
  routeHeader: {
    flexDirection: 'row',
    marginBottom: Spacing.sm,
  },
  routeIconContainer: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: Colors.primaryLight + '20',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: Spacing.md,
  },
  routeInfo: {
    flex: 1,
    justifyContent: 'center',
    gap: Spacing.xs,
  },
  locationRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  locationText: {
    ...Typography.body,
    color: Colors.textPrimary,
    flex: 1,
  },
  routeMeta: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.md,
    marginBottom: Spacing.md,
    paddingLeft: 48 + Spacing.md,
  },
  routeTime: {
    ...Typography.captionSmall,
    color: Colors.textSecondary,
  },
  routeDistance: {
    ...Typography.captionSmall,
    color: Colors.textTertiary,
  },
  routeActions: {
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
});
