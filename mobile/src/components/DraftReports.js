/**
 * DraftReports.js - 임시 저장된 제보 관리
 * 미완료 제보를 불러와서 계속 작성하거나 삭제
 */

import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  Alert,
  RefreshControl,
} from 'react-native';
import Icon from './icons/Icon';
import { Colors, Typography, Spacing } from '../styles';
import { reportAPI } from '../services/api';

const HAZARD_TYPES = {
  armed_conflict: { label: '무력충돌', icon: 'conflict', color: '#DC2626' },
  protest_riot: { label: '시위/폭동', icon: 'protest', color: '#F59E0B' },
  checkpoint: { label: '검문소', icon: 'checkpoint', color: '#FF6B6B' },
  road_damage: { label: '도로 손상', icon: 'roadDamage', color: '#F97316' },
  natural_disaster: { label: '자연재해', icon: 'naturalDisaster', color: '#DC2626' },
  other: { label: '기타', icon: 'other', color: '#6B7280' },
};

export default function DraftReports({ navigation, onContinue }) {
  const [drafts, setDrafts] = useState([]);
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    loadDrafts();
  }, []);

  /**
   * 임시 저장 목록 불러오기
   */
  const loadDrafts = async () => {
    setLoading(true);
    try {
      const response = await reportAPI.getDrafts();
      setDrafts(response.data || []);
    } catch (error) {
      console.error('Failed to load drafts:', error);
      Alert.alert('오류', '임시 저장 목록을 불러올 수 없습니다.');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  /**
   * 새로고침
   */
  const handleRefresh = () => {
    setRefreshing(true);
    loadDrafts();
  };

  /**
   * 임시 저장 계속 작성
   */
  const handleContinue = (draft) => {
    if (onContinue) {
      onContinue(draft);
    } else {
      navigation.navigate('Report', { draft });
    }
  };

  /**
   * 임시 저장 삭제
   */
  const handleDelete = (draftId) => {
    Alert.alert(
      '삭제 확인',
      '이 임시 저장을 삭제하시겠습니까?',
      [
        { text: '취소', style: 'cancel' },
        {
          text: '삭제',
          style: 'destructive',
          onPress: async () => {
            try {
              await reportAPI.delete(draftId);
              setDrafts(prev => prev.filter(d => d.id !== draftId));
              Alert.alert('완료', '삭제되었습니다.');
            } catch (error) {
              console.error('Failed to delete draft:', error);
              Alert.alert('오류', '삭제에 실패했습니다.');
            }
          },
        },
      ]
    );
  };

  /**
   * 시간 포맷팅
   */
  const formatTime = (dateString) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMinutes = Math.floor((now - date) / 60000);

    if (diffMinutes < 60) {
      return `${diffMinutes}분 전`;
    } else if (diffMinutes < 1440) {
      const hours = Math.floor(diffMinutes / 60);
      return `${hours}시간 전`;
    } else {
      const days = Math.floor(diffMinutes / 1440);
      if (days === 1) return '어제';
      return `${days}일 전`;
    }
  };

  /**
   * 각 임시 저장 아이템 렌더링
   */
  const renderDraftItem = ({ item }) => {
    const hazardInfo = HAZARD_TYPES[item.hazard_type] || HAZARD_TYPES.other;

    return (
      <TouchableOpacity
        style={styles.draftCard}
        onPress={() => handleContinue(item)}
        activeOpacity={0.7}
      >
        <View style={styles.draftHeader}>
          <View style={[styles.iconCircle, { backgroundColor: `${hazardInfo.color}20` }]}>
            <Icon name={hazardInfo.icon} size={24} color={hazardInfo.color} />
          </View>

          <View style={styles.draftInfo}>
            <Text style={styles.draftTitle}>{hazardInfo.label}</Text>
            <Text style={styles.draftTime}>{formatTime(item.created_at)}</Text>
          </View>

          <TouchableOpacity
            style={styles.deleteButton}
            onPress={() => handleDelete(item.id)}
            activeOpacity={0.7}
          >
            <Icon name="delete" size={20} color={Colors.danger} />
          </TouchableOpacity>
        </View>

        {item.description && (
          <Text style={styles.draftDescription} numberOfLines={2}>
            {item.description}
          </Text>
        )}

        <View style={styles.draftFooter}>
          <View style={styles.draftLocation}>
            <Icon name="location" size={14} color={Colors.textTertiary} />
            <Text style={styles.draftLocationText}>
              {item.latitude.toFixed(4)}, {item.longitude.toFixed(4)}
            </Text>
          </View>

          {item.photos && JSON.parse(item.photos).length > 0 && (
            <View style={styles.draftPhotos}>
              <Icon name="camera" size={14} color={Colors.textTertiary} />
              <Text style={styles.draftPhotosText}>
                {JSON.parse(item.photos).length}장
              </Text>
            </View>
          )}
        </View>
      </TouchableOpacity>
    );
  };

  /**
   * 빈 상태 렌더링
   */
  const renderEmpty = () => (
    <View style={styles.emptyContainer}>
      <Icon name="article" size={64} color={Colors.textTertiary} />
      <Text style={styles.emptyTitle}>임시 저장된 제보가 없습니다</Text>
      <Text style={styles.emptySubtitle}>
        작성 중인 제보를 나중에 계속할 수 있습니다
      </Text>
    </View>
  );

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>임시 저장</Text>
        {drafts.length > 0 && (
          <Text style={styles.count}>{drafts.length}건</Text>
        )}
      </View>

      <FlatList
        data={drafts}
        renderItem={renderDraftItem}
        keyExtractor={item => item.id}
        contentContainerStyle={[
          styles.listContent,
          drafts.length === 0 && styles.listContentEmpty,
        ]}
        ListEmptyComponent={renderEmpty}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={handleRefresh}
            tintColor={Colors.primary}
          />
        }
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: Spacing.lg,
    paddingVertical: Spacing.md,
    backgroundColor: Colors.surface,
    borderBottomWidth: 1,
    borderBottomColor: Colors.border,
  },
  title: {
    ...Typography.h2,
    color: Colors.textPrimary,
  },
  count: {
    ...Typography.body,
    color: Colors.textSecondary,
    backgroundColor: Colors.surfaceElevated,
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.xs,
    borderRadius: 12,
  },
  listContent: {
    padding: Spacing.lg,
    gap: Spacing.md,
  },
  listContentEmpty: {
    flex: 1,
  },
  draftCard: {
    backgroundColor: Colors.surface,
    borderRadius: 12,
    padding: Spacing.md,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  draftHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: Spacing.sm,
  },
  iconCircle: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: Spacing.md,
  },
  draftInfo: {
    flex: 1,
  },
  draftTitle: {
    ...Typography.body,
    fontWeight: '600',
    color: Colors.textPrimary,
    marginBottom: 2,
  },
  draftTime: {
    ...Typography.caption,
    color: Colors.textSecondary,
  },
  deleteButton: {
    padding: Spacing.sm,
  },
  draftDescription: {
    ...Typography.body,
    color: Colors.textSecondary,
    marginBottom: Spacing.sm,
    lineHeight: 20,
  },
  draftFooter: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.md,
  },
  draftLocation: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.xs,
  },
  draftLocationText: {
    ...Typography.caption,
    color: Colors.textTertiary,
  },
  draftPhotos: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.xs,
  },
  draftPhotosText: {
    ...Typography.caption,
    color: Colors.textTertiary,
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: Spacing.xl,
  },
  emptyTitle: {
    ...Typography.h3,
    color: Colors.textPrimary,
    marginTop: Spacing.lg,
    marginBottom: Spacing.xs,
    textAlign: 'center',
  },
  emptySubtitle: {
    ...Typography.body,
    color: Colors.textSecondary,
    textAlign: 'center',
  },
});
