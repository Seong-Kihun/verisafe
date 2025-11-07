/**
 * RoutePlanningScreen - 경로 찾기 화면
 * 
 * 책임:
 * 1. 출발지/목적지 입력
 * 2. 이동 수단 선택
 * 3. 경로 계산 및 목록 표시
 * 4. 경로 선택 시 위험 정보 브리핑
 */

import React, { useState, useEffect, useMemo, useCallback } from 'react';
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
import { routeAPI } from '../services/api';
import LocationInput from '../components/LocationInput';
import TransportationModeSelector from '../components/TransportationModeSelector';
import RouteCard from '../components/RouteCard';
import RouteComparison from '../components/RouteComparison';
import RouteHazardBriefing from '../components/RouteHazardBriefing';
import Icon from '../components/icons/Icon';
import { saveRecentDestination } from '../components/QuickAccessPanel';

export default function RoutePlanningScreen() {
  const { t } = useTranslation();
  const insets = useSafeAreaInsets();
  const navigation = useNavigation();
  const route = useRoute();

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
  const calculateRoutes = useCallback(async () => {
    if (!startLocation || !endLocation) return;

    setCalculating(true);
    try {
      const response = await routeAPI.calculate(
        { lat: startLocation.lat, lng: startLocation.lng },
        { lat: endLocation.lat, lng: endLocation.lng },
        'safe',
        transportationMode
      );

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
      console.error('[RoutePlanning] Failed to calculate routes:', error);
      Alert.alert(t('common.error'), t('route.calculationError'));
      setRoutesList([]);
    } finally {
      setCalculating(false);
    }
  }, [startLocation, endLocation, transportationMode, setRouteResponse]);

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

  // 출발지/목적지가 모두 입력되면 자동으로 경로 계산
  useEffect(() => {
    if (startLocation && endLocation) {
      calculateRoutes();
    } else {
      setRoutesList([]);
    }
  }, [startLocation, endLocation, transportationMode, calculateRoutes]);

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
    // 위험 정보 브리핑 열기
    openHazardBriefing();
    // 지도 탭으로 이동 (TabNavigator의 MapStack 탭)
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
    <View style={[styles.container, { paddingTop: insets.top }]}>
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
            <Text style={styles.sectionTitle}>{t('route.comparison')}</Text>
            <RouteComparison
              routes={routes}
              selectedRoute={selectedRoute}
              onSelect={null}
            />
            {/* 전체 경로 목록 */}
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
    padding: Spacing.lg,
  },
  section: {
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

