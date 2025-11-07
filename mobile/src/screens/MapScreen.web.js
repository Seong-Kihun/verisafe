/**
 * MapScreen.web.js - 웹용 지도 화면 (react-leaflet 사용)
 * 웹 환경에서만 사용됨
 */

import React, { useState, useEffect, useRef } from 'react';
import { 
  View, Text, StyleSheet, ActivityIndicator, Alert, TouchableOpacity, ScrollView
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import * as Location from 'expo-location';
import { Colors, Spacing, getRiskColor, Typography } from '../styles';
import { mapAPI, routeAPI } from '../services/api';
import { useMapContext } from '../contexts/MapContext';
import { useRoutePlanningContext } from '../contexts/RoutePlanningContext';
import PlaceDetailSheet from '../components/PlaceDetailSheet';
import RouteResultSheet from '../components/RouteResultSheet';
import RouteHazardBriefing from '../components/RouteHazardBriefing';
import WebMapView from '../components/WebMapView';
import SearchBar from '../components/SearchBar';
import Icon from '../components/icons/Icon';
import LayerToggleMenu from '../components/LayerToggleMenu';
import FloatingActionButton from '../components/FloatingActionButton';
import SafetyIndicator from '../components/SafetyIndicator';

// 위험 유형별 필터 버튼
const HAZARD_TYPES = [
  { id: 'armed_conflict', name: '무력충돌', icon: 'conflict', color: '#EF4444' },
  { id: 'protest_riot', name: '시위/폭동', icon: 'protest', color: '#F59E0B' },
  { id: 'checkpoint', name: '검문소', icon: 'checkpoint', color: '#FF6B6B' },
  { id: 'road_damage', name: '도로 손상', icon: 'roadDamage', color: '#F97316' },
  { id: 'natural_disaster', name: '자연재해', icon: 'naturalDisaster', color: '#DC2626' },
  { id: 'other', name: '기타', icon: 'other', color: '#6B7280' },
];

const JUBA_CENTER = {
  latitude: 4.8594,
  longitude: 31.5713,
  latitudeDelta: 0.05,
  longitudeDelta: 0.05,
};

export default function MapScreen() {
  const insets = useSafeAreaInsets();
  const navigation = useNavigation();
  const { 
    isPlaceSheetOpen, 
    isRouteSheetOpen, 
    selectedPlace, 
    routeResponse,
    userLocation,
    updateUserLocation,
    openPlaceSheet
  } = useMapContext();

  const {
    selectedRoute,
    routes,
    startLocation,
    endLocation,
    isHazardBriefingOpen,
    closeHazardBriefing,
    selectRoute
  } = useRoutePlanningContext();

  const { setRouteResponse } = useMapContext();

  const [loading, setLoading] = useState(true);
  const [landmarks, setLandmarks] = useState([]);
  const [hazards, setHazards] = useState([]); // 전체 위험 정보 (경로가 없을 때)
  const [routeHazards, setRouteHazards] = useState([]); // 경로 근처 위험 정보
  const [mapRegion, setMapRegion] = useState(JUBA_CENTER);
  // 기본적으로 주요 위험 유형들을 표시 (사용자가 바로 볼 수 있도록)
  const [activeHazardTypes, setActiveHazardTypes] = useState([
    'armed_conflict',
    'conflict',
    'protest_riot',
    'protest',
    'checkpoint',
    'natural_disaster',
    'flood',
    'landslide',
  ]); // 여러 위험 유형 선택 가능
  const [locationPermission, setLocationPermission] = useState(false);
  const [lastTap, setLastTap] = useState(null);
  const lastTapTimeoutRef = useRef(null);
  const [isLayerMenuOpen, setIsLayerMenuOpen] = useState(false);

  useEffect(() => {
    loadMapData();
    requestLocationPermission();

    // Cleanup: 컴포넌트 언마운트 시 timeout 정리
    return () => {
      if (lastTapTimeoutRef.current) {
        clearTimeout(lastTapTimeoutRef.current);
        lastTapTimeoutRef.current = null;
      }
    };
  }, []);

  const requestLocationPermission = async () => {
    try {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status === 'granted') {
        setLocationPermission(true);
        const location = await Location.getCurrentPositionAsync({});
        const loc = {
          latitude: location.coords.latitude,
          longitude: location.coords.longitude,
        };
        updateUserLocation(loc);
      }
    } catch (error) {
      console.error('Location permission error:', error);
    }
  };

  // selectedPlace가 변경될 때 지도 포커스 (하지만 지도 클릭으로 인한 변경은 제외)
  useEffect(() => {
    // selectedPlace가 있고, isPlaceSheetOpen이 true이면 사용자가 직접 선택한 것이므로 포커스
    // 하지만 지도 클릭으로 인한 역지오코딩은 제외 (지도가 새로고침되지 않도록)
    if (selectedPlace && isPlaceSheetOpen) {
      // 이미 해당 위치 근처에 있으면 포커스하지 않음 (지도 새로고침 방지)
      const currentLat = mapRegion.latitude;
      const currentLng = mapRegion.longitude;
      const placeLat = selectedPlace.latitude;
      const placeLng = selectedPlace.longitude;
      
      // 현재 위치와 선택한 위치의 거리 계산 (대략적으로)
      const latDiff = Math.abs(currentLat - placeLat);
      const lngDiff = Math.abs(currentLng - placeLng);
      const currentDelta = mapRegion.latitudeDelta;
      
      // 현재 보이는 영역 밖에 있으면 포커스
      // 하지만 지도 클릭으로 인한 선택(type === 'osm')은 제외
      if (selectedPlace.type !== 'osm' && (latDiff > currentDelta * 0.5 || lngDiff > currentDelta * 0.5)) {
        setMapRegion({
          latitude: placeLat,
          longitude: placeLng,
          latitudeDelta: 0.02,
          longitudeDelta: 0.02,
        });
      }
    }
  }, [selectedPlace?.id]); // selectedPlace.id만 감시 (지도 클릭으로 인한 변경은 무시)

  // 선택된 경로가 변경되면 경로 근처 위험 정보 로드 (지도 범위는 자동 변경하지 않음)
  useEffect(() => {
    if (selectedRoute && selectedRoute.polyline && selectedRoute.polyline.length > 0) {
      // 경로 근처 위험 정보 로드
      loadRouteHazards(selectedRoute);
      
      // 경로가 현재 보이는 영역 밖에 있으면 경로를 보이도록 조정 (선택적)
      const lats = selectedRoute.polyline.map(coord => coord[0]);
      const lngs = selectedRoute.polyline.map(coord => coord[1]);
      
      const minLat = Math.min(...lats);
      const maxLat = Math.max(...lats);
      const minLng = Math.min(...lngs);
      const maxLng = Math.max(...lngs);
      
      const routeCenterLat = (minLat + maxLat) / 2;
      const routeCenterLng = (minLng + maxLng) / 2;
      
      // 현재 지도 중심과 경로 중심의 거리 계산
      const currentCenterLat = mapRegion.latitude;
      const currentCenterLng = mapRegion.longitude;
      const latDiff = Math.abs(currentCenterLat - routeCenterLat);
      const lngDiff = Math.abs(currentCenterLng - routeCenterLng);
      const currentLatDelta = mapRegion.latitudeDelta;
      const currentLngDelta = mapRegion.longitudeDelta;
      
      // 경로가 현재 보이는 영역 밖에 있으면 경로를 포함하도록 조정
      const routeLatDelta = Math.max((maxLat - minLat) * 1.5, 0.01);
      const routeLngDelta = Math.max((maxLng - minLng) * 1.5, 0.01);
      
      // 경로가 현재 보이는 영역 밖에 있거나 경로가 너무 작아서 보이지 않으면 조정
      if (latDiff > currentLatDelta * 0.5 || lngDiff > currentLngDelta * 0.5 || 
          routeLatDelta > currentLatDelta || routeLngDelta > currentLngDelta) {
        // 경로를 포함하는 최소 범위 계산 (현재 범위와 경로 범위를 모두 포함)
        const combinedMinLat = Math.min(minLat, currentCenterLat - currentLatDelta / 2);
        const combinedMaxLat = Math.max(maxLat, currentCenterLat + currentLatDelta / 2);
        const combinedMinLng = Math.min(minLng, currentCenterLng - currentLngDelta / 2);
        const combinedMaxLng = Math.max(maxLng, currentCenterLng + currentLngDelta / 2);
        
        const newCenterLat = (combinedMinLat + combinedMaxLat) / 2;
        const newCenterLng = (combinedMinLng + combinedMaxLng) / 2;
        const newLatDelta = Math.max((combinedMaxLat - combinedMinLat) * 1.2, 0.01);
        const newLngDelta = Math.max((combinedMaxLng - combinedMinLng) * 1.2, 0.01);
        
        setMapRegion({
          latitude: newCenterLat,
          longitude: newCenterLng,
          latitudeDelta: newLatDelta,
          longitudeDelta: newLngDelta,
        });
      }
    } else {
      // 경로가 없으면 경로 위험 정보 초기화
      setRouteHazards([]);
    }
  }, [selectedRoute]);

  // 경로 근처 위험 정보 로드
  const loadRouteHazards = async (route) => {
    if (!route || !route.polyline || !route.id) return;
    
    try {
      const response = await routeAPI.getRouteHazards(route.id, route.polyline);
      
      // 백엔드 응답에서 위험 정보 추출
      const hazardsData = response.data;
      const routeHazardsList = [];
      
      // hazards 배열에서 위험 정보 추출
      if (hazardsData.hazards && Array.isArray(hazardsData.hazards)) {
        hazardsData.hazards.forEach((hazard) => {
          routeHazardsList.push({
            id: hazard.hazard_id || `hazard_${hazard.latitude}_${hazard.longitude}`,
            latitude: hazard.latitude,
            longitude: hazard.longitude,
            risk_score: hazard.risk_score,
            hazard_type: hazard.hazard_type,
            description: hazard.description || '',
            radius: 0.1, // 기본 반경 (km 단위)
          });
        });
      }
      
      setRouteHazards(routeHazardsList);
    } catch (error) {
      console.error('[MapScreen] Failed to load route hazards:', error);
      setRouteHazards([]);
    }
  };

  const loadMapData = async () => {
    try {
      console.log('[MapScreen Web DEBUG] 지도 데이터 로딩 시작...');

      const response = await mapAPI.getBounds(4.8, 31.5, 4.9, 31.6);

      console.log('[MapScreen Web DEBUG] API 응답 상태:', response.status);
      console.log('[MapScreen Web DEBUG] API 응답 전체:', JSON.stringify(response.data, null, 2));
      console.log('[MapScreen Web DEBUG] landmarks 개수:', response.data.landmarks?.length || 0);
      console.log('[MapScreen Web DEBUG] hazards 개수:', response.data.hazards?.length || 0);

      if (response.data.hazards && response.data.hazards.length > 0) {
        console.log('[MapScreen Web DEBUG] 첫 번째 hazard 샘플:', JSON.stringify(response.data.hazards[0], null, 2));
      } else {
        console.warn('[MapScreen Web DEBUG] ⚠️ hazards가 비어있음!');
      }

      setLandmarks(response.data.landmarks || []);
      setHazards(response.data.hazards || []);

      console.log('[MapScreen Web DEBUG] ✅ landmarks 설정 완료:', response.data.landmarks?.length || 0, '개');
      console.log('[MapScreen Web DEBUG] ✅ hazards 설정 완료:', response.data.hazards?.length || 0, '개');
    } catch (error) {
      console.error('[MapScreen Web DEBUG] ❌ 지도 데이터 로딩 실패');
      console.error('[MapScreen Web DEBUG] 에러 메시지:', error.message);
      console.error('[MapScreen Web DEBUG] 에러 코드:', error.code);
      console.error('[MapScreen Web DEBUG] 에러 응답:', error.response?.data);
      console.error('[MapScreen Web DEBUG] 에러 상태:', error.response?.status);
      console.error('[MapScreen] Failed to load map data:', error);
      console.error('[MapScreen] Error code:', error.code);
      console.error('[MapScreen] Error message:', error.message);
      console.error('[MapScreen] Error response:', error.response?.status, error.response?.data);
      
      let errorMessage = '지도 데이터를 불러올 수 없습니다.';
      
      if (error.code === 'ECONNABORTED' || error.message?.includes('timeout')) {
        errorMessage = '서버 응답 시간이 초과되었습니다.\n\n가능한 원인:\n• 백엔드 서버가 실행 중인지 확인\n• 네트워크 연결 상태 확인\n• 방화벽 설정 확인';
      } else if (error.code === 'ECONNREFUSED' || error.message?.includes('Network Error')) {
        errorMessage = '백엔드 서버에 연결할 수 없습니다.\n\n확인 사항:\n• 백엔드 서버가 실행 중인지 확인\n• API_BASE_URL 설정 확인\n• 네트워크 연결 상태 확인';
      } else if (error.response?.status === 404) {
        errorMessage = '백엔드 서버를 찾을 수 없습니다.\n\n확인 사항:\n• 서버가 실행 중인지 확인\n• API 경로가 올바른지 확인';
      } else if (error.response?.status >= 500) {
        errorMessage = '서버 내부 오류가 발생했습니다.\n\n백엔드 로그를 확인해주세요.';
      }
      
      Alert.alert('오류', errorMessage);
    } finally {
      setLoading(false);
    }
  };

  // 위험 유형 필터 토글 (중복 선택 가능)
  const handleHazardTypeFilter = (hazardTypeId) => {
    setActiveHazardTypes(prev => {
      if (prev.includes(hazardTypeId)) {
        // 이미 선택된 경우 제거
        return prev.filter(type => type !== hazardTypeId);
      } else {
        // 선택되지 않은 경우 추가
        return [...prev, hazardTypeId];
      }
    });
  };

  const toggleLayerMenu = () => {
    setIsLayerMenuOpen(prev => !prev);
  };

  const handleMyLocation = async () => {
    if (!locationPermission || !userLocation) {
      await requestLocationPermission();
      return;
    }

    setMapRegion({
      latitude: userLocation.latitude,
      longitude: userLocation.longitude,
      latitudeDelta: 0.01,
      longitudeDelta: 0.01,
    });
  };

  // 지도 클릭 핸들러 - 더블 탭/단일 탭 구분
  const handleMapPress = async (lat, lng) => {
    const now = Date.now();
    const DOUBLE_TAP_DELAY = 300;
    
    // 이전 탭의 타임아웃 취소
    if (lastTapTimeoutRef.current) {
      clearTimeout(lastTapTimeoutRef.current);
      lastTapTimeoutRef.current = null;
    }
    
    // 더블 탭 감지
    if (lastTap && (now - lastTap) < DOUBLE_TAP_DELAY) {
      // 더블 탭 - 줌 인만 수행
      const newDelta = mapRegion.latitudeDelta / 2;
      setMapRegion({
        latitude: lat,
        longitude: lng,
        latitudeDelta: Math.max(newDelta, 0.001),
        longitudeDelta: Math.max(newDelta, 0.001),
      });
      setLastTap(null);
      return; // 더블 탭이면 장소 선택은 하지 않음
    }
    
    // 단일 탭 - 장소 선택 (역지오코딩)
    const currentTap = now;
    setLastTap(currentTap);
    
    // 단일 탭이면 잠시 후 장소 선택 (더블 탭인지 확인하기 위해)
    lastTapTimeoutRef.current = setTimeout(async () => {
      // 더블 탭이 아니면 (lastTap이 변경되지 않았으면)
      if (lastTap === currentTap) {
        try {
          // 좌표로 역지오코딩하여 장소 정보 조회
          const response = await mapAPI.reverseGeocode(lat, lng);
          if (response.data) {
            const placeData = response.data;
            openPlaceSheet({
              id: placeData.id,
              latitude: placeData.latitude,
              longitude: placeData.longitude,
              name: placeData.name || '선택한 위치',
              address: placeData.description || `${lat.toFixed(4)}, ${lng.toFixed(4)}`,
              category: placeData.category || 'other',
              description: placeData.description,
              type: 'osm',
            });
          }
        } catch (error) {
          console.error('Failed to reverse geocode:', error);
          // 에러 시에도 기본 정보로 PlaceDetailSheet 열기
          openPlaceSheet({
            latitude: lat,
            longitude: lng,
            name: '선택한 위치',
            address: `${lat.toFixed(4)}, ${lng.toFixed(4)}`,
            category: 'other',
            type: 'osm',
          });
        }
      }
      lastTapTimeoutRef.current = null;
    }, DOUBLE_TAP_DELAY);
  };

  // 더블 탭 줌 핸들러 (WebMapView에 전달) - 더 이상 사용 안 함
  const handleDoublePress = (lat, lng) => {
    // handleMapPress에서 처리
  };

  // 롱 프레스 핸들러 - 장소 선택 또는 빠른 제보
  const handleLongPress = async (lat, lng) => {
    // 옵션 선택 다이얼로그
    Alert.alert(
      '지도 작업',
      `위도: ${lat.toFixed(5)}\n경도: ${lng.toFixed(5)}`,
      [
        {
          text: '📍 여기 제보하기',
          onPress: () => {
            navigation.navigate('Report', {
              location: { latitude: lat, longitude: lng },
            });
          },
        },
        {
          text: '🔍 장소 정보 보기',
          onPress: async () => {
            try {
              // 좌표로 역지오코딩하여 장소 정보 조회
              const response = await mapAPI.reverseGeocode(lat, lng);
              if (response.data) {
                const placeData = response.data;
                openPlaceSheet({
                  id: placeData.id,
                  latitude: placeData.latitude,
                  longitude: placeData.longitude,
                  name: placeData.name || '선택한 위치',
                  address: placeData.description || `${lat.toFixed(4)}, ${lng.toFixed(4)}`,
                  category: placeData.category || 'other',
                  description: placeData.description,
                  type: 'osm',
                });
              }
            } catch (error) {
              console.error('Failed to reverse geocode:', error);
              // 에러 시에도 기본 정보로 PlaceDetailSheet 열기
              openPlaceSheet({
                latitude: lat,
                longitude: lng,
                name: '선택한 위치',
                address: `${lat.toFixed(4)}, ${lng.toFixed(4)}`,
                category: 'other',
                type: 'osm',
              });
            }
          },
        },
        {
          text: '취소',
          style: 'cancel',
        },
      ],
      { cancelable: true }
    );
  };

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color={Colors.primary} />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <WebMapView
        landmarks={landmarks}
        hazards={(() => {
          console.log('[MapScreen Web DEBUG] 렌더링 체크:');
          console.log('[MapScreen Web DEBUG] - activeHazardTypes:', activeHazardTypes);
          console.log('[MapScreen Web DEBUG] - hazards 개수:', hazards?.length || 0);
          console.log('[MapScreen Web DEBUG] - activeHazardTypes.length === 0:', activeHazardTypes.length === 0);

          // 선택된 위험 유형이 있을 때만 표시 (초기에는 아무것도 표시하지 않음)
          if (activeHazardTypes.length === 0) {
            console.log('[MapScreen Web DEBUG] ⚠️ 위험 정보 렌더링 건너뜀 (activeHazardTypes.length === 0)');
            return [];
          }

          // 기본 지도의 위험 정보를 사용 (경로 선택 여부와 관계없이 동일한 위험 정보 표시)
          const hazardsToShow = hazards.filter(hazard => activeHazardTypes.includes(hazard.hazard_type));
          console.log('[MapScreen Web DEBUG] ✅ 렌더링할 hazards:', hazardsToShow.length, '개');
          return hazardsToShow;
        })()}
        routeResponse={routeResponse}
        selectedRoute={selectedRoute}
        routes={routes}
        activeHazardTypes={activeHazardTypes}
        mapRegion={mapRegion}
        userLocation={userLocation}
        startLocation={startLocation}
        endLocation={endLocation}
        style={styles.map}
        onPress={handleMapPress}
        onDoublePress={handleDoublePress}
        onLongPress={handleLongPress}
        onMarkerPress={(place) => {
          // 마커 클릭 시 장소 정보 카드 표시
          openPlaceSheet(place);
        }}
      />

      {/* 플로팅 검색 바 & 레이어 버튼 */}
      <View style={{
        position: 'absolute',
        top: insets.top,
        left: 0,
        right: 0,
        zIndex: 1000,
      }}>
        {/* 검색바 + 레이어 버튼 */}
        <View style={{
          paddingTop: Spacing.xs,
          paddingHorizontal: Spacing.md,
          flexDirection: 'row',
          alignItems: 'center',
          gap: Spacing.sm,
        }}>
          <View style={{ flex: 1 }}>
            <SearchBar
              onPress={() => navigation.navigate('Search')}
              placeholder="어디로 갈까요?"
            />
          </View>
          <TouchableOpacity
            style={styles.layerButton}
            onPress={toggleLayerMenu}
            activeOpacity={0.8}
          >
            <Icon name="layers" size={24} color={Colors.primary} />
          </TouchableOpacity>
        </View>

        {/* 안전도 인디케이터 */}
        <SafetyIndicator
          userLocation={userLocation}
          onPress={() => {
            // 주변 위험 정보 상세 보기
            Alert.alert(
              '주변 안전 정보',
              '주변 위험 정보를 확인하려면 레이어 버튼을 눌러 위험 유형을 선택하세요.'
            );
          }}
        />
      </View>

      {/* 내 위치 버튼 */}
      <TouchableOpacity
        style={styles.myLocationButton}
        onPress={handleMyLocation}
        activeOpacity={0.8}
      >
        <Icon name="myLocation" size={24} color={Colors.primary} />
      </TouchableOpacity>

      {/* FAB - 경로 찾기 버튼 */}
      <FloatingActionButton />

      {/* 경로 토글 버튼 - 안전 경로/최소시간 경로 전환 */}
      {selectedRoute && routes.length > 1 && (() => {
        const safeRoute = routes.find(r => r.type === 'safe');
        const fastRoute = routes.find(r => r.type === 'fast');
        
        const handleToggleRoute = (route) => {
          selectRoute(route);
          setRouteResponse({ routes: [route] });
        };
        
        return (safeRoute || fastRoute) ? (
          <View style={styles.routeToggleContainer}>
            {safeRoute && (
              <TouchableOpacity
                style={[
                  styles.routeToggleButton,
                  selectedRoute.type === 'safe' && styles.routeToggleButtonActive
                ]}
                onPress={() => handleToggleRoute(safeRoute)}
                activeOpacity={0.8}
              >
                <Icon 
                  name="safe" 
                  size={20} 
                  color={selectedRoute.type === 'safe' ? Colors.textInverse : Colors.textSecondary} 
                />
                <Text style={[
                  styles.routeToggleText,
                  selectedRoute.type === 'safe' && styles.routeToggleTextActive
                ]}>
                  안전 경로
                </Text>
              </TouchableOpacity>
            )}
            
            {fastRoute && (
              <TouchableOpacity
                style={[
                  styles.routeToggleButton,
                  selectedRoute.type === 'fast' && styles.routeToggleButtonActive
                ]}
                onPress={() => handleToggleRoute(fastRoute)}
                activeOpacity={0.8}
              >
                <Icon 
                  name="fast" 
                  size={20} 
                  color={selectedRoute.type === 'fast' ? Colors.textInverse : Colors.textSecondary} 
                />
                <Text style={[
                  styles.routeToggleText,
                  selectedRoute.type === 'fast' && styles.routeToggleTextActive
                ]}>
                  최소시간
                </Text>
              </TouchableOpacity>
            )}
          </View>
        ) : null;
      })()}

      {/* PlaceDetailSheet */}
      {isPlaceSheetOpen && <PlaceDetailSheet />}
      
      {/* RouteResultSheet */}
      {isRouteSheetOpen && <RouteResultSheet />}

      {/* RouteHazardBriefing - 경로 위험 정보 시트 */}
      {isHazardBriefingOpen && selectedRoute && (
        <RouteHazardBriefing
          route={selectedRoute}
          isVisible={isHazardBriefingOpen}
          onClose={closeHazardBriefing}
        />
      )}

      {/* LayerToggleMenu - 레이어 선택 메뉴 */}
      <LayerToggleMenu
        visible={isLayerMenuOpen}
        onClose={() => setIsLayerMenuOpen(false)}
        activeTypes={activeHazardTypes}
        onToggle={handleHazardTypeFilter}
      />

      {/* OpenStreetMap 저작권 표시 */}
      <View style={styles.osmAttribution}>
        <Text style={styles.osmAttributionText}>
          © OpenStreetMap contributors
        </Text>
      </View>
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
  map: {
    flex: 1,
  },
  appTitle: {
    ...Typography.h2,
    fontSize: 24,
    fontWeight: '700',
    color: Colors.primary,
    letterSpacing: 0.5,
  },
  layerButton: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: Colors.surface,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: Colors.shadowDark,
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.4,
    shadowRadius: 8,
    elevation: 6,
  },
  myLocationButton: {
    position: 'absolute',
    right: Spacing.lg,
    bottom: Spacing.xl + 72,
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: Colors.surface,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: Colors.shadowDark,
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.4,
    shadowRadius: 8,
    elevation: 6,
    zIndex: 1000,
  },
  routeToggleContainer: {
    position: 'absolute',
    bottom: Spacing.xl + 80,
    left: Spacing.lg,
    right: Spacing.lg,
    flexDirection: 'row',
    gap: Spacing.sm,
  },
  routeToggleButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: Colors.surface,
    paddingVertical: Spacing.md,
    paddingHorizontal: Spacing.lg,
    borderRadius: 24,
    gap: Spacing.sm,
    shadowColor: Colors.shadowDark,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 4,
    elevation: 3,
  },
  routeToggleButtonActive: {
    backgroundColor: Colors.primary,
    shadowOpacity: 0.3,
    elevation: 5,
  },
  routeToggleText: {
    ...Typography.labelMedium,
    color: Colors.textSecondary,
  },
  routeToggleTextActive: {
    color: Colors.textInverse,
    fontWeight: '600',
  },
  osmAttribution: {
    position: 'absolute',
    bottom: Spacing.xs,
    left: Spacing.xs,
    backgroundColor: 'rgba(255, 255, 255, 0.8)',
    paddingHorizontal: Spacing.xs,
    paddingVertical: 2,
    borderRadius: 4,
  },
  osmAttributionText: {
    fontSize: 10,
    color: Colors.textSecondary,
  },
});

