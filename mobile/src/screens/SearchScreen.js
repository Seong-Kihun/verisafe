/**
 * SearchScreen - 장소 검색 전체 화면 모달
 * 
 * 책임:
 * 1. 텍스트 입력 받기
 * 2. 자동완성 검색 (debounce 300ms)
 * 3. 최근 검색어 표시
 * 4. 장소 선택 시 MapContext 업데이트
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  FlatList,
  ActivityIndicator,
  Keyboard,
  Alert,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useNavigation, useRoute, CommonActions } from '@react-navigation/native';
import { Colors, Spacing, Typography } from '../styles';
import { useMapContext } from '../contexts/MapContext';
import { mapAPI } from '../services/api';
import Icon from '../components/icons/Icon';

const RECENT_SEARCHES = [
  { id: '1', name: '주바 국제공항', category: 'airport' },
  { id: '2', name: '주바 시청', category: 'government' },
];

// Phase 3: 카테고리 빠른 검색
const SEARCH_CATEGORIES = [
  { id: 'hospital', name: '병원', icon: 'local-hospital', query: 'hospital' },
  { id: 'embassy', name: '대사관', icon: 'account-balance', query: 'embassy' },
  { id: 'hotel', name: '호텔', icon: 'hotel', query: 'hotel' },
  { id: 'airport', name: '공항', icon: 'flight', query: 'airport' },
];

export default function SearchScreen() {
  const insets = useSafeAreaInsets();
  const navigation = useNavigation();
  const route = useRoute();
  const { closeSearch, openPlaceSheet } = useMapContext();
  
  // Route params에서 mode 가져오기
  const mode = route.params?.mode || 'general'; // 'start', 'end', 'general'
  
  const [query, setQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [showRecent, setShowRecent] = useState(true);

  const handleSearch = useCallback(async (searchQuery) => {
    try {
      console.log('[SearchScreen] 검색 요청:', searchQuery);
      const response = await mapAPI.autocomplete(searchQuery);
      console.log('[SearchScreen] 검색 응답:', response);
      console.log('[SearchScreen] response.data:', response.data);
      console.log('[SearchScreen] 결과 개수:', response.data?.length || 0);
      const results = response.data || [];
      console.log('[SearchScreen] 첫 번째 결과:', results[0]);
      setSearchResults(results);
      console.log('[SearchScreen] searchResults 상태 업데이트:', results.length);
    } catch (error) {
      console.error('[SearchScreen] 검색 실패:', error);
      console.error('[SearchScreen] 에러 상세:', error.response?.data || error.message);
      // 타임아웃 에러는 사용자에게 알리지 않고 빈 결과만 표시 (자동완성 특성상)
      if (error.code === 'ECONNABORTED' || error.message?.includes('timeout')) {
        console.warn('[SearchScreen] 검색 타임아웃 - 백엔드 서버 상태를 확인하세요');
      }
      setSearchResults([]);
    }
  }, []);

  // 검색어 변경 시 자동완성 (debounce 500ms, 최소 3자 이상)
  useEffect(() => {
    if (query.length === 0) {
      setSearchResults([]);
      setShowRecent(true);
      setIsSearching(false);
      return;
    }

    // 최소 검색어 길이 체크 (3자 이상)
    if (query.length < 3) {
      setSearchResults([]);
      setShowRecent(false);
      setIsSearching(false);
      return;
    }

    setShowRecent(false);
    setIsSearching(true);

    const timeoutId = setTimeout(async () => {
      await handleSearch(query);
      setIsSearching(false);
    }, 500); // debounce 시간을 300ms에서 500ms로 증가

    return () => clearTimeout(timeoutId);
  }, [query, handleSearch]);

  const handleSelectPlace = async (place) => {
    console.log('[SearchScreen] 장소 선택:', place);
    try {
      // 검색 결과에 이미 좌표가 있으면 바로 사용
      if (place.latitude && place.longitude) {
        const placeData = {
          id: place.id,
          name: place.name,
          description: place.address,
          address: place.address,
          latitude: place.latitude,
          longitude: place.longitude,
          category: null,
        };

        // RoutePlanningScreen에서 호출한 경우
        if (mode === 'start' || mode === 'end') {
          navigation.navigate('RoutePlanning', {
            selectedPlace: placeData,
            mode: mode,
          });
          Keyboard.dismiss();
          return;
        }

        // 일반 검색 모드 (MapScreen에서 호출)
        openPlaceSheet(placeData);
        closeSearch();
        // 지도 탭으로 이동
        navigation.dispatch(
          CommonActions.navigate({
            name: 'MapStack',
            params: {
              screen: 'MapMain',
            },
          })
        );
        Keyboard.dismiss();
        return;
      }

      // 좌표가 없으면 상세 정보 조회
      const response = await mapAPI.getPlaceDetail(place.id);
      if (response.data) {
        const placeData = {
          ...response.data,
          address: place.address || response.data.description,
        };

        // RoutePlanningScreen에서 호출한 경우
        if (mode === 'start' || mode === 'end') {
          navigation.navigate('RoutePlanning', {
            selectedPlace: placeData,
            mode: mode,
          });
          Keyboard.dismiss();
          return;
        }

        // 일반 검색 모드
        openPlaceSheet(placeData);
        closeSearch();
        // 지도 탭으로 이동
        navigation.dispatch(
          CommonActions.navigate({
            name: 'MapStack',
            params: {
              screen: 'MapMain',
            },
          })
        );
        Keyboard.dismiss();
      }
    } catch (error) {
      console.error('Failed to get place detail:', error);
      // 실패 시에도 기본 정보로 표시 (좌표가 있으면)
      if (place.latitude && place.longitude) {
        const placeData = {
          id: place.id,
          name: place.name,
          description: place.address,
          address: place.address,
          latitude: place.latitude,
          longitude: place.longitude,
          category: null,
        };

        if (mode === 'start' || mode === 'end') {
          navigation.navigate('RoutePlanning', {
            selectedPlace: placeData,
            mode: mode,
          });
        } else {
          openPlaceSheet(placeData);
          closeSearch();
          // 지도 탭으로 이동
          navigation.dispatch(
            CommonActions.navigate({
              name: 'MapStack',
              params: {
                screen: 'MapMain',
              },
            })
          );
        }
      } else {
        Alert.alert('오류', '장소 정보를 불러올 수 없습니다.');
      }
      Keyboard.dismiss();
    }
  };

  const getCategoryIcon = (category) => {
    if (category === 'airport') return 'flight';
    if (category === 'government') return 'account-balance';
    return 'location';
  };

  // Phase 3: 카테고리 검색 핸들러
  const handleCategorySearch = async (category) => {
    setQuery(category.query);
    setShowRecent(false);
    setIsSearching(true);
    await handleSearch(category.query);
    setIsSearching(false);
  };

  const renderRecentItem = ({ item }) => (
    <TouchableOpacity style={styles.listItem} onPress={() => handleSelectPlace(item)}>
      <View style={styles.listItemIcon}>
        <Icon name={getCategoryIcon(item.category)} size={24} color={Colors.primary} />
      </View>
      <View style={styles.listItemContent}>
        <Text style={styles.listItemTitle}>{item.name}</Text>
      </View>
    </TouchableOpacity>
  );

  const renderSearchResult = ({ item }) => (
    <TouchableOpacity style={styles.listItem} onPress={() => handleSelectPlace(item)}>
      <View style={styles.listItemIcon}>
        <Icon name="location" size={24} color={Colors.primary} />
      </View>
      <View style={styles.listItemContent}>
        <Text style={styles.listItemTitle}>{item.name}</Text>
        {item.address && (
          <Text style={styles.listItemSubtitle} numberOfLines={1}>{item.address}</Text>
        )}
      </View>
    </TouchableOpacity>
  );

  return (
    <View style={[styles.container, { paddingTop: insets.top }]}>
      {/* 헤더 */}
      <View style={styles.header}>
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => {
            if (mode === 'start' || mode === 'end') {
              navigation.goBack();
            } else {
              closeSearch();
              // 지도 탭으로 이동
              navigation.dispatch(
                CommonActions.navigate({
                  name: 'MapStack',
                  params: {
                    screen: 'MapMain',
                  },
                })
              );
            }
          }}
        >
          <Text style={styles.backButtonText}>←</Text>
        </TouchableOpacity>
        <View style={styles.searchInputContainer}>
          <TextInput
            style={styles.searchInput}
            placeholder={mode === 'start' ? '출발지 검색' : mode === 'end' ? '목적지 검색' : '장소 검색'}
            placeholderTextColor={Colors.textSecondary}
            value={query}
            onChangeText={setQuery}
            autoFocus
          />
          {isSearching && (
            <ActivityIndicator
              size="small"
              color={Colors.primary}
              style={styles.loadingIndicator}
            />
          )}
        </View>
      </View>

      {/* 지도에서 선택 버튼 (경로찾기 모드일 때만 표시) */}
      {(mode === 'start' || mode === 'end') && (
        <TouchableOpacity
          style={styles.mapSelectButton}
          onPress={() => {
            navigation.goBack();
            // MapScreen으로 이동하면서 selectLocationMode 전달
            navigation.dispatch(
              CommonActions.navigate({
                name: 'MapStack',
                params: {
                  screen: 'MapMain',
                  params: {
                    selectLocationMode: mode,
                  },
                },
              })
            );
          }}
          activeOpacity={0.8}
        >
          <Icon name="map" size={20} color={Colors.primary} />
          <Text style={styles.mapSelectButtonText}>지도에서 선택</Text>
        </TouchableOpacity>
      )}

      {/* 검색 결과 / 최근 검색어 */}
      <KeyboardAvoidingView
        style={styles.keyboardAvoidingView}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 0 : 20}
      >
        <ScrollView
          style={styles.listContainer}
          contentContainerStyle={styles.listContent}
          keyboardShouldPersistTaps="handled"
          showsVerticalScrollIndicator={false}
        >
          {/* Phase 3: 카테고리 빠른 검색 */}
          {showRecent && query.length === 0 && (
            <View style={styles.categoriesSection}>
              <Text style={styles.sectionTitle}>빠른 검색</Text>
              <ScrollView
                horizontal
                showsHorizontalScrollIndicator={false}
                style={styles.categoriesScroll}
                contentContainerStyle={styles.categoriesContent}
              >
                {SEARCH_CATEGORIES.map((category) => (
                  <TouchableOpacity
                    key={category.id}
                    style={styles.categoryButton}
                    onPress={() => handleCategorySearch(category)}
                    activeOpacity={0.7}
                  >
                    <View style={styles.categoryIcon}>
                      <Icon name={category.icon} size={24} color={Colors.primary} />
                    </View>
                    <Text style={styles.categoryText}>{category.name}</Text>
                  </TouchableOpacity>
                ))}
              </ScrollView>
            </View>
          )}

          {showRecent ? (
            <>
              <Text style={styles.sectionTitle}>최근 검색</Text>
              {RECENT_SEARCHES.map((item) => (
                <View key={item.id}>
                  {renderRecentItem({ item })}
                </View>
              ))}
            </>
          ) : (
            <>
              {isSearching ? (
                <View style={styles.loadingContainer}>
                  <ActivityIndicator size="large" color={Colors.primary} />
                  <Text style={styles.loadingText}>검색 중...</Text>
                </View>
              ) : searchResults.length === 0 && query.length >= 3 ? (
                <View style={styles.emptyContainer}>
                  <Icon name="search" size={48} color={Colors.textTertiary} />
                  <Text style={styles.emptyTitle}>검색 결과가 없습니다</Text>
                  <Text style={styles.emptyDescription}>
                    다른 검색어를 입력해보세요.
                  </Text>
                </View>
              ) : query.length > 0 && query.length < 3 ? (
                <View style={styles.emptyContainer}>
                  <Icon name="search" size={48} color={Colors.textTertiary} />
                  <Text style={styles.emptyTitle}>검색어를 더 입력하세요</Text>
                  <Text style={styles.emptyDescription}>
                    최소 3자 이상 입력해주세요.
                  </Text>
                </View>
              ) : searchResults.length > 0 ? (
                <>
                  <Text style={styles.sectionTitle}>
                    검색 결과 ({Math.min(searchResults.length, 15)}개)
                  </Text>
                  {searchResults.slice(0, 15).map((item, index) => (
                    <View key={item.id || String(item.place_id) || index}>
                      {renderSearchResult({ item })}
                    </View>
                  ))}
                </>
              ) : null}
            </>
          )}
        </ScrollView>
      </KeyboardAvoidingView>
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
    alignItems: 'center',
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.sm,
    backgroundColor: Colors.surface,
    borderBottomWidth: 1,
    borderBottomColor: Colors.border,
  },
  backButton: {
    width: 40,
    height: 40,
    justifyContent: 'center',
    alignItems: 'center',
  },
  backButtonText: {
    fontSize: 24,
    color: Colors.textPrimary,
  },
  searchInputContainer: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    marginLeft: Spacing.sm,
  },
  searchInput: {
    flex: 1,
    height: 40,
    ...Typography.input,
    color: Colors.textPrimary,
  },
  loadingIndicator: {
    marginLeft: Spacing.sm,
  },
  keyboardAvoidingView: {
    flex: 1,
  },
  listContainer: {
    flex: 1,
  },
  listContent: {
    paddingHorizontal: Spacing.md,
    paddingBottom: Spacing.xl,
  },
  loadingContainer: {
    padding: Spacing.xxxl,
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 200,
  },
  loadingText: {
    ...Typography.body,
    color: Colors.textSecondary,
    marginTop: Spacing.md,
  },
  sectionTitle: {
    ...Typography.label,
    color: Colors.textSecondary,
    marginTop: Spacing.lg,
    marginBottom: Spacing.sm,
    paddingHorizontal: Spacing.sm,
  },
  emptyContainer: {
    padding: Spacing.xxxl,
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 200,
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
  },
  emptyText: {
    ...Typography.body,
    color: Colors.textSecondary,
  },
  listItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: Spacing.md,
    paddingHorizontal: Spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: Colors.borderLight,
  },
  listItemIcon: {
    width: 40,
    height: 40,
    justifyContent: 'center',
    alignItems: 'center',
  },
  listItemContent: {
    flex: 1,
    marginLeft: Spacing.md,
  },
  listItemTitle: {
    ...Typography.body,
    color: Colors.textPrimary,
  },
  listItemSubtitle: {
    ...Typography.bodySmall,
    color: Colors.textSecondary,
    marginTop: Spacing.xs,
  },
  // Phase 3: 카테고리 검색 스타일
  categoriesSection: {
    marginBottom: Spacing.md,
  },
  categoriesScroll: {
    marginTop: Spacing.sm,
  },
  categoriesContent: {
    gap: Spacing.sm,
    paddingHorizontal: Spacing.xs,
  },
  categoryButton: {
    alignItems: 'center',
    paddingVertical: Spacing.sm,
    paddingHorizontal: Spacing.md,
    minWidth: 80,
  },
  categoryIcon: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: Colors.primary + '15',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: Spacing.xs,
  },
  categoryText: {
    ...Typography.labelSmall,
    color: Colors.textPrimary,
    textAlign: 'center',
  },
  mapSelectButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: Colors.primary + '10',
    borderRadius: 8,
    paddingVertical: Spacing.sm,
    paddingHorizontal: Spacing.md,
    marginHorizontal: Spacing.md,
    marginVertical: Spacing.sm,
    gap: Spacing.xs,
  },
  mapSelectButtonText: {
    ...Typography.labelSmall,
    color: Colors.primary,
    fontWeight: '600',
  },
});

