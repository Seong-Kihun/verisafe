/**
 * 즐겨찾기 장소 관리 화면
 */

import React, { useState, useEffect, useCallback } from 'react';
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
import { savedPlacesStorage } from '../services/storage';
import Icon from '../components/icons/Icon';

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

export default function SavedPlacesScreen({ navigation }) {
  const [loading, setLoading] = useState(true);
  const [places, setPlaces] = useState([]);

  // 화면이 포커스될 때마다 데이터 다시 로드
  useFocusEffect(
    useCallback(() => {
      loadPlaces();
    }, [])
  );

  const loadPlaces = async () => {
    try {
      const data = await savedPlacesStorage.getAll();
      setPlaces(data);
    } catch (error) {
      console.error('Failed to load places:', error);
      Alert.alert('오류', '즐겨찾기 장소를 불러올 수 없습니다.');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = (placeId, placeName) => {
    Alert.alert(
      '삭제 확인',
      `"${placeName}"을(를) 삭제하시겠습니까?`,
      [
        { text: '취소', style: 'cancel' },
        {
          text: '삭제',
          style: 'destructive',
          onPress: async () => {
            const success = await savedPlacesStorage.remove(placeId);
            if (success) {
              setPlaces(prev => prev.filter(p => p.id !== placeId));
            } else {
              Alert.alert('오류', '삭제에 실패했습니다.');
            }
          },
        },
      ]
    );
  };

  const handleViewOnMap = (place) => {
    // MapStack의 Map 화면으로 이동하고 해당 장소를 표시
    navigation.navigate('MapStack', {
      screen: 'Map',
      params: {
        selectedPlace: place,
      },
    });
  };

  const handleFindRoute = (place) => {
    // MapStack의 RoutePlanning 화면으로 이동하고 도착지로 설정
    navigation.navigate('MapStack', {
      screen: 'RoutePlanning',
      params: {
        destination: {
          latitude: place.latitude,
          longitude: place.longitude,
          name: place.name,
        },
      },
    });
  };

  const renderItem = ({ item }) => (
    <View style={styles.placeCard}>
      <View style={styles.placeHeader}>
        <View style={styles.placeIcon}>
          <Text style={styles.placeIconText}>{getCategoryIcon(item.category)}</Text>
        </View>
        <View style={styles.placeInfo}>
          <Text style={styles.placeName}>{item.name}</Text>
          <Text style={styles.placeCoords}>
            {item.latitude.toFixed(4)}, {item.longitude.toFixed(4)}
          </Text>
          {item.address && (
            <Text style={styles.placeAddress} numberOfLines={1}>
              {item.address}
            </Text>
          )}
        </View>
      </View>

      <View style={styles.placeActions}>
        <TouchableOpacity
          style={styles.actionButton}
          onPress={() => handleViewOnMap(item)}
        >
          <Icon name="map" size={20} color={Colors.primary} />
          <Text style={styles.actionButtonText}>지도</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.actionButton}
          onPress={() => handleFindRoute(item)}
        >
          <Icon name="directions" size={20} color={Colors.primary} />
          <Text style={styles.actionButtonText}>경로</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.actionButton, styles.deleteButton]}
          onPress={() => handleDelete(item.id, item.name)}
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

  if (places.length === 0) {
    return (
      <View style={styles.emptyContainer}>
        <Icon name="bookmarkBorder" size={64} color={Colors.textTertiary} />
        <Text style={styles.emptyTitle}>즐겨찾기가 비어있습니다</Text>
        <Text style={styles.emptyText}>
          지도에서 장소를 검색하고{'\n'}즐겨찾기에 추가해보세요
        </Text>
        <TouchableOpacity
          style={styles.goToMapButton}
          onPress={() => navigation.navigate('MapStack')}
        >
          <Icon name="map" size={20} color={Colors.textInverse} />
          <Text style={styles.goToMapButtonText}>지도로 이동</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <FlatList
        data={places}
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
  listContainer: {
    padding: Spacing.md,
  },
  placeCard: {
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
  placeHeader: {
    flexDirection: 'row',
    marginBottom: Spacing.md,
  },
  placeIcon: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: Colors.primaryLight + '20',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: Spacing.md,
  },
  placeIconText: {
    fontSize: 28,
  },
  placeInfo: {
    flex: 1,
    justifyContent: 'center',
  },
  placeName: {
    ...Typography.h3,
    color: Colors.textPrimary,
    marginBottom: Spacing.xs,
  },
  placeCoords: {
    ...Typography.captionSmall,
    color: Colors.textTertiary,
    marginBottom: Spacing.xs,
  },
  placeAddress: {
    ...Typography.bodySmall,
    color: Colors.textSecondary,
  },
  placeActions: {
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
