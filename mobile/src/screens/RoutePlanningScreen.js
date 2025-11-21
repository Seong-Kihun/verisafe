/**
 * RoutePlanningScreen - 경로 찾기 화면
 * 
 * 책임:
 * 1. 출발지/목적지 입력
 * 2. 이동 수단 선택
 * 3. 경로 계산 및 목록 표시
 * 4. 경로 선택 시 위험 정보 브리핑
 */

import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  ActivityIndicator,
  Alert,
  TouchableOpacity,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useNavigation, useRoute, CommonActions } from '@react-navigation/native';
import { useTranslation } from 'react-i18next';
import { Colors, Spacing, Typography } from '../styles';
import { useRoutePlanningContext } from '../contexts/RoutePlanningContext';
import { useMapContext } from '../contexts/MapContext';
import { useHazardFilter } from '../contexts/HazardFilterContext';
import { routeAPI } from '../services/api';
import LocationInput from '../components/LocationInput';
import TransportationModeSelector from '../components/TransportationModeSelector';
import RouteCard from '../components/RouteCard';
import RouteHazardBriefing from '../components/RouteHazardBriefing';
import Icon from '../components/icons/Icon';
import { saveRecentDestination } from '../components/QuickAccessPanel';

export default function RoutePlanningScreen() {
  const { t } = useTranslation();
  const insets = useSafeAreaInsets();
  const navigation = useNavigation();
  const route = useRoute();

  // 경쟁 상태 방지를 위한 요청 ID 관리
  const requestIdRef = useRef(0);

  const {
    startLocation,
    endLocation,
    transportationMode,
    routes,
    selectedRoute,
    isHazardBriefingOpen,
    setStart,
    setEnd,
    setTransportation,
    setRoutesList,
    selectRoute,
    openHazardBriefing,
    closeHazardBriefing,
    setCalculating,
    isCalculating,
  } = useRoutePlanningContext();

  const { setRouteResponse } = useMapContext();
  const { excludedHazardTypes } = useHazardFilter();

  // 정렬 옵션 상태
  const [sortBy, setSortBy] = useState('time'); // 'time', 'distance', 'risk'

  // 정렬된 경로 목록 (useMemo는 항상 호출되어야 함)
  const sortedRoutes = useMemo(() => {
    if (routes.length === 0) return [];
    return [...routes].sort((a, b) => {
      if (sortBy === 'time') {
        return (a.duration || 0) - (b.duration || 0);
      } else if (sortBy === 'distance') {
        return (a.distance || 0) - (b.distance || 0);
      } else if (sortBy === 'risk') {
        return (a.risk_score || 10) - (b.risk_score || 10);
      }
      return 0;
    });
  }, [routes, sortBy]);

  // 경로 계산 함수 (useEffect보다 먼저 정의)
  // 경쟁 상태 방지: 가장 최근 요청의 결과만 적용
  const calculateRoutes = useCallback(async () => {
    if (!startLocation || !endLocation) return;

    // 새로운 요청 ID 생성
    requestIdRef.current += 1;
    const currentRequestId = requestIdRef.current;

    setCalculating(true);
    try {
      const response = await routeAPI.calculate(
        { lat: startLocation.lat, lng: startLocation.lng },
        { lat: endLocation.lat, lng: endLocation.lng },
        'safe',
        transportationMode,
        excludedHazardTypes
      );

      // 경쟁 상태 체크: 이 요청이 아직 최신 요청인지 확인
      if (currentRequestId !== requestIdRef.current) {
        console.log(`[RoutePlanning] 요청 무시됨 (구버전): 요청 ID ${currentRequestId}, 현재 ID ${requestIdRef.current}`);
        return; // 더 이상 최신 요청이 아니면 무시
      }

      if (response.data && response.data.routes) {
        console.log(`[RoutePlanning] 경로 계산 완료: ${response.data.routes.length}개 경로`);
        response.data.routes.forEach((route, i) => {
          console.log(`[RoutePlanning] 경로 ${i+1} (${route.type}): 거리=${route.distance}km, 위험도=${route.risk_score}/10, 시간=${route.duration}분`);
        });

        setRoutesList(response.data.routes);
        // 모든 경로를 MapContext에 저장 (지도에 표시하기 위해)
        setRouteResponse({ routes: response.data.routes });
        // 첫 번째 경로는 자동 선택하지 않음 (사용자가 선택하도록)
        // if (response.data.routes.length > 0) {
        //   selectRoute(response.data.routes[0]);
        // }
      } else {
        Alert.alert(t('route.noRouteTitle'), t('route.noRouteMessage'));
        setRoutesList([]);
      }
    } catch (error) {
      // 경쟁 상태 체크: 에러 처리도 최신 요청만 적용
      if (currentRequestId !== requestIdRef.current) {
        console.log(`[RoutePlanning] 에러 무시됨 (구버전): 요청 ID ${currentRequestId}`);
        return;
      }

      console.error('[RoutePlanning] Failed to calculate routes:', error);

      // 에러 타입에 따른 상세 메시지
      let errorTitle = t('common.error');
      let errorMessage = t('route.calculationError');

      if (error.userMessage) {
        // API 인터셉터에서 설정한 사용자 친화적 메시지
        errorMessage = error.userMessage;
      } else if (error.response) {
        // HTTP 응답 에러
        const status = error.response.status;
        if (status === 400) {
          errorMessage = '잘못된 요청입니다. 출발지와 목적지를 확인해주세요.';
        } else if (status === 404) {
          errorMessage = '경로를 찾을 수 없습니다. 다른 출발지나 목적지를 선택해보세요.';
        } else if (status === 500) {
          errorMessage = '서버 오류가 발생했습니다. 잠시 후 다시 시도해주세요.';
        }
      } else if (error.request) {
        // 네트워크 에러
        errorMessage = '서버에 연결할 수 없습니다. 네트워크 연결을 확인해주세요.';
      }

      // 위험 필터가 너무 많은 경우 추가 안내
      if (excludedHazardTypes.length > 5) {
        errorMessage += '\n\n너무 많은 위험 유형을 제외하면 경로를 찾기 어려울 수 있습니다. 필터를 조정해보세요.';
      }

      Alert.alert(errorTitle, errorMessage);
      setRoutesList([]);
    } finally {
      // 경쟁 상태 체크: 로딩 상태도 최신 요청만 변경
      if (currentRequestId === requestIdRef.current) {
        setCalculating(false);
      }
    }
  }, [startLocation, endLocation, transportationMode, excludedHazardTypes, setRouteResponse, t, setRoutesList, setCalculating]);

  // Debounced 버전의 경로 계산 함수 (필터 변경 시 사용)
  const timeoutRef = useRef(null);
  const debouncedCalculateRoutes = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    timeoutRef.current = setTimeout(() => {
      calculateRoutes();
    }, 500);
  }, [calculateRoutes]);

  // route params에서 목적지 가져오기 (PlaceDetailSheet에서 경로 버튼 클릭 시)
  useEffect(() => {
    if (route.params?.destination) {
      const dest = route.params.destination;
      setEnd({
        lat: dest.latitude,
        lng: dest.longitude,
        name: dest.name,
        address: dest.description || dest.address,
      });
    }
  }, [route.params]);

  // SearchScreen에서 선택된 장소 처리
  useEffect(() => {
    if (route.params?.selectedPlace && route.params?.mode) {
      const place = route.params.selectedPlace;
      const location = {
        lat: place.latitude,
        lng: place.longitude,
        name: place.name,
        address: place.address || place.description,
      };

      if (route.params.mode === 'start') {
        setStart(location);
      } else if (route.params.mode === 'end') {
        setEnd(location);
        // 최근 목적지에 자동 저장
        saveRecentDestination(location);
      }

      // params 초기화 (중복 처리 방지)
      navigation.setParams({
        selectedPlace: undefined,
        mode: undefined,
      });
    }
  }, [route.params?.selectedPlace, route.params?.mode]);

  // 출발지/목적지가 모두 입력되면 자동으로 경로 계산
  // excludedHazardTypes가 변경되면 경로 재계산 (debounced)
  useEffect(() => {
    if (startLocation && endLocation) {
      // 필터 변경은 debounced, 출발지/목적지는 즉시 실행
      debouncedCalculateRoutes();
    } else {
      setRoutesList([]);
    }

    // Cleanup: 컴포넌트 언마운트 시 pending된 timeout 취소
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [startLocation, endLocation, transportationMode, excludedHazardTypes, debouncedCalculateRoutes]);

  const handleStartPress = () => {
    navigation.navigate('Search', { 
      mode: 'start',
      onSelect: (place) => {
        setStart({
          lat: place.latitude,
          lng: place.longitude,
          name: place.name,
          address: place.address,
        });
      }
    });
  };

  const handleEndPress = () => {
    navigation.navigate('Search', {
      mode: 'end',
      onSelect: (place) => {
        const destination = {
          lat: place.latitude,
          lng: place.longitude,
          name: place.name,
          address: place.address,
        };
        setEnd(destination);

        // Phase 3: 최근 목적지에 자동 저장
        saveRecentDestination(destination);
      }
    });
  };

  const handleRouteSelect = async (route) => {
    selectRoute(route);
    // MapContext에 모든 경로 정보 저장 (지도에 모든 경로 표시하기 위해)
    // 선택된 경로만이 아니라 모든 경로를 유지
    setRouteResponse({ routes: routes });
    // 지도 탭으로 이동 (TabNavigator의 MapStack 탭)
    // 경로 위험 정보 모달은 자동으로 열지 않음 (사용자가 상세 정보 버튼을 누를 때 열림)
    navigation.dispatch(
      CommonActions.navigate({
        name: 'MapStack',
        params: {
          screen: 'MapMain',
        },
      })
    );
  };

  return (
    <View style={styles.container}>
      <ScrollView 
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* 출발지/목적지 입력 */}
        <View style={styles.section}>
          <LocationInput
            label={t('route.start')}
            value={startLocation}
            placeholder={t('route.startPlaceholder')}
            onPress={handleStartPress}
            icon="location"
          />
          <LocationInput
            label={t('route.destination')}
            value={endLocation}
            placeholder={t('route.destinationPlaceholder')}
            onPress={handleEndPress}
            icon="navigation"
          />
        </View>

        {/* 이동 수단 선택 */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>{t('route.transportation')}</Text>
          <TransportationModeSelector
            selectedMode={transportationMode}
            onSelect={setTransportation}
          />
        </View>


        {/* 경로 목록 */}
        {isCalculating ? (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color={Colors.primary} />
            <Text style={styles.loadingText}>{t('route.calculating')}</Text>
          </View>
        ) : routes.length > 0 ? (
          <View style={styles.section}>
            {/* 경로 목록 */}
            <View style={styles.routeListContainer}>
              <View style={styles.routeListHeader}>
                <View style={styles.headerTitleContainer}>
                  <Text style={styles.sectionTitle}>{t('route.routeOptions')} ({routes.length})</Text>
                  <Text style={styles.headerSubtitle}>
                    {t('route.compareRoutes')}
                  </Text>
                </View>
                <View style={styles.sortButtons}>
                  <TouchableOpacity
                    style={[styles.sortButton, sortBy === 'time' && styles.sortButtonActive]}
                    onPress={() => setSortBy('time')}
                  >
                    <Text style={[styles.sortButtonText, sortBy === 'time' && styles.sortButtonTextActive]}>
                      {t('route.sortByTime')}
                    </Text>
                  </TouchableOpacity>
                  <TouchableOpacity
                    style={[styles.sortButton, sortBy === 'distance' && styles.sortButtonActive]}
                    onPress={() => setSortBy('distance')}
                  >
                    <Text style={[styles.sortButtonText, sortBy === 'distance' && styles.sortButtonTextActive]}>
                      {t('route.sortByDistance')}
                    </Text>
                  </TouchableOpacity>
                  <TouchableOpacity
                    style={[styles.sortButton, sortBy === 'risk' && styles.sortButtonActive]}
                    onPress={() => setSortBy('risk')}
                  >
                    <Text style={[styles.sortButtonText, sortBy === 'risk' && styles.sortButtonTextActive]}>
                      {t('route.sortBySafety')}
                    </Text>
                  </TouchableOpacity>
                </View>
              </View>
              {sortedRoutes.map((routeItem) => (
                <RouteCard
                  key={routeItem.id}
                  route={routeItem}
                  onSelect={handleRouteSelect}
                  isSelected={selectedRoute?.id === routeItem.id}
                />
              ))}
            </View>
          </View>
        ) : startLocation && endLocation ? (
          <View style={styles.emptyContainer}>
            <Icon name="route" size={48} color={Colors.textTertiary} />
            <Text style={styles.emptyTitle}>{t('route.noRouteFound')}</Text>
            <Text style={styles.emptyDescription}>
              {t('route.noRouteDescription')}
            </Text>
          </View>
        ) : null}

        {/* 안내 메시지 */}
        {!startLocation || !endLocation ? (
          <View style={styles.infoContainer}>
            <Icon name="navigation" size={48} color={Colors.primary + '40'} />
            <Text style={styles.infoText}>
              {t('route.infoMessage')}
            </Text>
          </View>
        ) : null}
      </ScrollView>

      {/* 위험 정보 브리핑 모달 */}
      <RouteHazardBriefing
        route={selectedRoute}
        isVisible={isHazardBriefingOpen}
        onClose={closeHazardBriefing}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    paddingTop: Spacing.sm,
    paddingHorizontal: Spacing.lg,
    paddingBottom: Spacing.lg,
  },
  section: {
    marginTop: Spacing.md,
    marginBottom: Spacing.xl,
  },
  routeListContainer: {
    marginTop: Spacing.lg,
  },
  routeListHeader: {
    marginBottom: Spacing.md,
  },
  headerTitleContainer: {
    marginBottom: Spacing.sm,
  },
  headerSubtitle: {
    ...Typography.bodySmall,
    color: Colors.textSecondary,
    marginTop: Spacing.xs,
    lineHeight: 18,
  },
  sortButtons: {
    flexDirection: 'row',
    gap: Spacing.sm,
    marginTop: Spacing.sm,
  },
  sortButton: {
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.xs,
    borderRadius: 8,
    backgroundColor: Colors.surface,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  sortButtonActive: {
    backgroundColor: Colors.primary,
    borderColor: Colors.primary,
  },
  sortButtonText: {
    ...Typography.labelSmall,
    color: Colors.textSecondary,
    fontWeight: '600',
  },
  sortButtonTextActive: {
    color: Colors.textInverse,
  },
  sectionTitle: {
    ...Typography.h3,
    color: Colors.textPrimary,
    marginBottom: Spacing.md,
  },
  loadingContainer: {
    padding: Spacing.xl,
    alignItems: 'center',
  },
  loadingText: {
    ...Typography.body,
    color: Colors.textSecondary,
    marginTop: Spacing.md,
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
    textAlign: 'center',
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
  infoContainer: {
    padding: Spacing.xl,
    alignItems: 'center',
    backgroundColor: Colors.primary + '10',
    borderRadius: 16,
    marginTop: Spacing.lg,
    marginHorizontal: Spacing.lg,
  },
  infoText: {
    ...Typography.body,
    color: Colors.textPrimary,
    textAlign: 'center',
    marginTop: Spacing.md,
    lineHeight: 24,
  },
});

