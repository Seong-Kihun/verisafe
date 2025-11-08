/**
 * MapScreen.native.js - 모바일용 지도 화면 (react-native-maps 사용)
 * iOS/Android에서만 사용됨
 */

import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
  View, Text, StyleSheet, ActivityIndicator, Alert, TouchableOpacity, ScrollView, Linking
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useNavigation, useFocusEffect } from '@react-navigation/native';
import * as Location from 'expo-location';
import MapView, { Marker, Polyline, Circle, UrlTile } from 'react-native-maps';
import { Colors, Spacing, getRiskColor, getRouteColor, Typography } from '../styles';
import { mapAPI, routeAPI, emergencyAPI } from '../services/api';
import { useMapContext } from '../contexts/MapContext';
import { useRoutePlanningContext } from '../contexts/RoutePlanningContext';
import { useHazardFilter } from '../contexts/HazardFilterContext';
import { emergencyContactsStorage, userProfileStorage } from '../services/storage';
import { sendSOSSMS } from '../services/sms';
import PlaceDetailSheet from '../components/PlaceDetailSheet';
import RouteResultSheet from '../components/RouteResultSheet';
import RouteHazardBriefing from '../components/RouteHazardBriefing';
import SearchBar from '../components/SearchBar';
import Icon from '../components/icons/Icon';
import LayerToggleMenu from '../components/LayerToggleMenu';
import FloatingActionButton from '../components/FloatingActionButton';
import SafetyIndicator from '../components/SafetyIndicator';
import SOSConfirmModal from '../components/SOSConfirmModal';
import { HAZARD_TYPES } from '../constants/hazardTypes';

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
    openPlaceSheet,
    setRouteResponse,
    userCountry
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

  const { excludedHazardTypes, toggleHazardType } = useHazardFilter();

  const [loading, setLoading] = useState(true);
  const [landmarks, setLandmarks] = useState([]);
  const [hazards, setHazards] = useState([]); // 전체 위험 정보 (경로가 없을 때)
  const [routeHazards, setRouteHazards] = useState([]); // 경로 근처 위험 정보
  const [mapRegion, setMapRegion] = useState(JUBA_CENTER);
  const mapRef = useRef(null); // MapView 참조
  const isUserPanningRef = useRef(false); // 사용자가 직접 지도를 이동시키는 중인지 추적
  const [locationPermission, setLocationPermission] = useState(false);
  const [lastTap, setLastTap] = useState(null);
  const lastTapTimeoutRef = useRef(null);
  const [isLayerMenuOpen, setIsLayerMenuOpen] = useState(false);
  const [timeFilter, setTimeFilter] = useState('all'); // 시간 필터 (all, 24h, 48h, 7d)
  const [isSOSModalOpen, setIsSOSModalOpen] = useState(false);
  const [emergencyContacts, setEmergencyContacts] = useState([]);

  // 모든 위험 유형 목록 (excludedHazardTypes를 제외한 유형들을 표시)
  // useMemo로 최적화하여 불필요한 재계산 방지
  const activeHazardTypes = useMemo(() => {
    const ALL_HAZARD_TYPE_IDS = HAZARD_TYPES.map(t => t.id);
    return ALL_HAZARD_TYPE_IDS.filter(id => !excludedHazardTypes.includes(id));
  }, [excludedHazardTypes]);

  // 초기 마운트 시 권한 요청 및 cleanup
  useEffect(() => {
    requestLocationPermission();

    // Cleanup: 컴포넌트 언마운트 시 timeout 정리
    return () => {
      if (lastTapTimeoutRef.current) {
        clearTimeout(lastTapTimeoutRef.current);
        lastTapTimeoutRef.current = null;
      }
    };
  }, []);

  // 국가 변경 시 지도 데이터 재로드
  useEffect(() => {
    loadMapData();
  }, [userCountry]);

  // 화면 포커스 시 긴급 연락처 재로드 (다른 화면에서 편집했을 수 있음)
  useFocusEffect(
    React.useCallback(() => {
      loadEmergencyContacts();
    }, [])
  );

  // 긴급 연락처 로드
  const loadEmergencyContacts = async () => {
    try {
      const contacts = await emergencyContactsStorage.getAll();
      setEmergencyContacts(contacts);
    } catch (error) {
      console.error('[MapScreen] Failed to load emergency contacts:', error);
    }
  };

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
      console.error('[MapScreen] Location permission error:', error);
    }
  };

  // SOS 버튼 클릭
  const handleSOSButtonPress = () => {
    setIsSOSModalOpen(true);
  };

  // SOS 확인 (실제 전송)
  const handleSOSConfirm = async () => {
    try {
      if (!userLocation) {
        Alert.alert('오류', '현재 위치를 확인할 수 없습니다.');
        setIsSOSModalOpen(false);
        return;
      }

      if (emergencyContacts.length === 0) {
        Alert.alert('알림', '등록된 긴급 연락처가 없습니다.\n프로필 → 긴급 연락망에서 먼저 등록해주세요.');
        setIsSOSModalOpen(false);
        return;
      }

      // 사용자 프로필 가져오기
      const userProfile = await userProfileStorage.get();
      if (!userProfile) {
        console.error('[SOS] Failed to get user profile');
        Alert.alert('오류', '사용자 정보를 불러올 수 없습니다.');
        setIsSOSModalOpen(false);
        return;
      }

      const userName = userProfile.name || '사용자';
      const userId = userProfile.id || 1;

      // 백엔드에 SOS 이벤트 저장 시도
      let backendResponse = null;
      let backendFailed = false;

      try {
        const sosData = {
          user_id: userId,
          latitude: userLocation.latitude,
          longitude: userLocation.longitude,
          message: '긴급 SOS 요청',
        };

        backendResponse = await emergencyAPI.triggerSOS(sosData);
        console.log('[SOS] Backend response:', backendResponse);
      } catch (backendError) {
        console.error('[SOS] Backend failed, continuing with SMS:', backendError);
        backendFailed = true;
        // 백엔드 실패해도 SMS는 계속 진행
      }

      // SMS 발송 (백엔드 실패와 무관하게 진행)
      const smsSent = await sendSOSSMS(emergencyContacts, userLocation, userName);

      // 모달 닫기
      setIsSOSModalOpen(false);

      // 결과에 따른 알림
      if (!smsSent && backendFailed) {
        // 둘 다 실패
        Alert.alert(
          '⚠️ SOS 전송 실패',
          '서버와 SMS 전송에 모두 실패했습니다.\n긴급 상황이므로 직접 전화를 걸어주세요.',
          [
            { text: '취소', style: 'cancel' },
            {
              text: '전화 걸기',
              onPress: () => {
                if (emergencyContacts.length > 0) {
                  const firstContact = emergencyContacts.sort((a, b) => a.priority - b.priority)[0];
                  Linking.openURL(`tel:${firstContact.phone}`);
                }
              }
            }
          ]
        );
        return;
      }

      // 성공 알림 구성
      const nearestHavenInfo = backendResponse?.nearest_safe_haven
        ? `\n\n📍 가장 가까운 대피처:\n${backendResponse.nearest_safe_haven.name}\n거리: ${Math.round(backendResponse.nearest_safe_haven.distance)}m`
        : '';

      const smsInfo = smsSent
        ? `\n✅ ${emergencyContacts.length}명에게 SMS 전송 완료`
        : `\n⚠️ SMS 전송 취소됨`;

      const backendInfo = backendFailed
        ? '\n⚠️ 서버 연결 실패 (오프라인 모드)'
        : '';

      Alert.alert(
        '🆘 SOS 발송 완료',
        `긴급 알림이 전송되었습니다.${smsInfo}${backendInfo}${nearestHavenInfo}`,
        [{ text: '확인' }]
      );
    } catch (error) {
      console.error('[SOS] Critical error in SOS flow:', error);
      setIsSOSModalOpen(false);

      // 최종 fallback - 전화 걸기 옵션 제공
      Alert.alert(
        '심각한 오류',
        'SOS 전송 중 오류가 발생했습니다.\n직접 전화를 걸어주세요.',
        [
          { text: '취소', style: 'cancel' },
          {
            text: '전화 걸기',
            onPress: () => {
              if (emergencyContacts.length > 0) {
                const firstContact = emergencyContacts.sort((a, b) => a.priority - b.priority)[0];
                Linking.openURL(`tel:${firstContact.phone}`);
              }
            }
          }
        ]
      );
    }
  };

  // SOS 취소
  const handleSOSCancel = () => {
    setIsSOSModalOpen(false);
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
        // 프로그래매틱하게 지도를 이동시킬 때는 isUserPanningRef를 false로 유지
        // animateToRegion을 사용하여 부드럽게 이동
        if (mapRef.current) {
          mapRef.current.animateToRegion({
            latitude: placeLat,
            longitude: placeLng,
            latitudeDelta: 0.02,
            longitudeDelta: 0.02,
          }, 500);
          // animateToRegion은 onRegionChangeComplete를 트리거하지만,
          // isUserPanningRef.current가 false이므로 setMapRegion이 호출되지 않음
          // 따라서 수동으로 mapRegion 업데이트
          setMapRegion({
            latitude: placeLat,
            longitude: placeLng,
            latitudeDelta: 0.02,
            longitudeDelta: 0.02,
          });
        } else {
          setMapRegion({
            latitude: placeLat,
            longitude: placeLng,
            latitudeDelta: 0.02,
            longitudeDelta: 0.02,
          });
        }
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
        
        if (mapRef.current) {
          mapRef.current.animateToRegion({
            latitude: newCenterLat,
            longitude: newCenterLng,
            latitudeDelta: newLatDelta,
            longitudeDelta: newLngDelta,
          }, 500);
          setMapRegion({
            latitude: newCenterLat,
            longitude: newCenterLng,
            latitudeDelta: newLatDelta,
            longitudeDelta: newLngDelta,
          });
        }
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
      // 응답 형식: { hazards: [RouteHazardInfo], hazards_by_type: {...}, summary: {...} }
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
            radius: 0.1, // 기본 반경 (km 단위), 실제로는 DB에서 가져와야 하지만 일단 기본값 사용
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
      console.log('[MapScreen DEBUG] 지도 데이터 로딩 시작...');
      console.log('[MapScreen DEBUG] 선택된 국가:', userCountry?.name || '기본(남수단)');

      const response = await mapAPI.getBounds(4.8, 31.5, 4.9, 31.6, userCountry?.code);

      console.log('[MapScreen DEBUG] API 응답 상태:', response.status);
      console.log('[MapScreen DEBUG] API 응답 전체:', JSON.stringify(response.data, null, 2));
      console.log('[MapScreen DEBUG] landmarks 개수:', response.data.landmarks?.length || 0);
      console.log('[MapScreen DEBUG] hazards 개수:', response.data.hazards?.length || 0);

      if (response.data.hazards && response.data.hazards.length > 0) {
        console.log('[MapScreen DEBUG] 첫 번째 hazard 샘플:', JSON.stringify(response.data.hazards[0], null, 2));
      } else {
        console.warn('[MapScreen DEBUG] ⚠️ hazards가 비어있음!');
      }

      setLandmarks(response.data.landmarks || []);
      setHazards(response.data.hazards || []);

      console.log('[MapScreen DEBUG] ✅ landmarks 설정 완료:', response.data.landmarks?.length || 0, '개');
      console.log('[MapScreen DEBUG] ✅ hazards 설정 완료:', response.data.hazards?.length || 0, '개');
    } catch (error) {
      console.error('[MapScreen DEBUG] ❌ 지도 데이터 로딩 실패');
      console.error('[MapScreen DEBUG] 에러 메시지:', error.message);
      console.error('[MapScreen DEBUG] 에러 코드:', error.code);
      console.error('[MapScreen DEBUG] 에러 응답:', error.response?.data);
      console.error('[MapScreen DEBUG] 에러 상태:', error.response?.status);
      console.error('[MapScreen] Failed to load map data:', error);

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
  // HazardFilterContext와 연동하여 경로 계산에도 반영
  const handleHazardTypeFilter = (hazardTypeId) => {
    toggleHazardType(hazardTypeId);
  };

  // 시간 필터 변경 핸들러
  const handleTimeFilterChange = (filterId) => {
    console.log('[MapScreen] 시간 필터 변경:', filterId);
    setTimeFilter(filterId);
  };

  // 시간 필터에 따라 위험 정보 필터링
  const getFilteredHazardsByTime = (hazardsList) => {
    if (timeFilter === 'all') {
      return hazardsList;
    }

    const now = new Date();
    let hoursLimit;

    switch (timeFilter) {
      case '24h':
        hoursLimit = 24;
        break;
      case '48h':
        hoursLimit = 48;
        break;
      case '7d':
        hoursLimit = 168; // 7 days
        break;
      default:
        return hazardsList;
    }

    return hazardsList.filter(hazard => {
      // start_date 또는 created_at이 있는 경우만 필터링
      const hazardDate = hazard.start_date ? new Date(hazard.start_date) :
                         hazard.created_at ? new Date(hazard.created_at) : null;

      if (!hazardDate) {
        // 날짜 정보가 없으면 항상 표시
        return true;
      }

      const hoursDiff = (now - hazardDate) / (1000 * 60 * 60);
      return hoursDiff <= hoursLimit;
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

    // 프로그래매틱하게 지도를 이동시킬 때는 animateToRegion 사용
    if (mapRef.current) {
      mapRef.current.animateToRegion({
        latitude: userLocation.latitude,
        longitude: userLocation.longitude,
        latitudeDelta: 0.01,
        longitudeDelta: 0.01,
      }, 500);
      // animateToRegion은 onRegionChangeComplete를 트리거하지만,
      // isUserPanningRef.current가 false이므로 setMapRegion이 호출되지 않음
      // 따라서 수동으로 mapRegion 업데이트
      setMapRegion({
        latitude: userLocation.latitude,
        longitude: userLocation.longitude,
        latitudeDelta: 0.01,
        longitudeDelta: 0.01,
      });
    } else {
      setMapRegion({
        latitude: userLocation.latitude,
        longitude: userLocation.longitude,
        latitudeDelta: 0.01,
        longitudeDelta: 0.01,
      });
    }
  };

  // 지도 클릭 핸들러 - 더블 탭/단일 탭 구분
  const handleMapPress = async (event) => {
    const now = Date.now();
    const DOUBLE_TAP_DELAY = 300;
    const { latitude, longitude } = event.nativeEvent.coordinate;
    
    // 이전 탭의 타임아웃 취소
    if (lastTapTimeoutRef.current) {
      clearTimeout(lastTapTimeoutRef.current);
      lastTapTimeoutRef.current = null;
    }
    
    // 더블 탭 감지
    if (lastTap && (now - lastTap) < DOUBLE_TAP_DELAY) {
      // 더블 탭 - 줌 인만 수행
      const currentDelta = mapRegion.latitudeDelta;
      const newDelta = currentDelta / 2;
      
      if (mapRef.current) {
        mapRef.current.animateToRegion({
          latitude,
          longitude,
          latitudeDelta: Math.max(newDelta, 0.001),
          longitudeDelta: Math.max(newDelta, 0.001),
        }, 300);
        // animateToRegion은 onRegionChangeComplete를 트리거하지만,
        // isUserPanningRef.current가 false이므로 setMapRegion이 호출되지 않음
        // 따라서 수동으로 mapRegion 업데이트
        setMapRegion({
          latitude,
          longitude,
          latitudeDelta: Math.max(newDelta, 0.001),
          longitudeDelta: Math.max(newDelta, 0.001),
        });
      } else {
        setMapRegion({
          latitude,
          longitude,
          latitudeDelta: Math.max(newDelta, 0.001),
          longitudeDelta: Math.max(newDelta, 0.001),
        });
      }
      setLastTap(null);
      return; // 더블 탭이면 장소 선택은 하지 않음
    }
    
    // 단일 탭 - 장소 선택 (역지오코딩)
    setLastTap(now);
    
    // 단일 탭이면 잠시 후 장소 선택 (더블 탭인지 확인하기 위해)
    lastTapTimeoutRef.current = setTimeout(async () => {
      // 더블 탭이 아니면 (lastTap이 변경되지 않았으면)
      if (lastTap === now) {
        try {
          // 좌표로 역지오코딩하여 장소 정보 조회
          const response = await mapAPI.reverseGeocode(latitude, longitude);
          if (response.data) {
            const placeData = response.data;
            openPlaceSheet({
              id: placeData.id,
              latitude: placeData.latitude,
              longitude: placeData.longitude,
              name: placeData.name || '선택한 위치',
              address: placeData.description || `${latitude.toFixed(4)}, ${longitude.toFixed(4)}`,
              category: placeData.category || 'other',
              description: placeData.description,
              type: 'osm',
            });
          }
        } catch (error) {
          console.error('Failed to reverse geocode:', error);
          // 에러 시에도 기본 정보로 PlaceDetailSheet 열기
          openPlaceSheet({
            latitude,
            longitude,
            name: '선택한 위치',
            address: `${latitude.toFixed(4)}, ${longitude.toFixed(4)}`,
            category: 'other',
            type: 'osm',
          });
        }
      }
      lastTapTimeoutRef.current = null;
    }, DOUBLE_TAP_DELAY);
  };

  // 롱 프레스 핸들러 - 장소 선택 또는 빠른 제보
  const handleLongPress = async (event) => {
    const { latitude, longitude } = event.nativeEvent.coordinate;

    // 옵션 선택 다이얼로그
    Alert.alert(
      '지도 작업',
      `위도: ${latitude.toFixed(5)}\n경도: ${longitude.toFixed(5)}`,
      [
        {
          text: '📍 여기 제보하기',
          onPress: () => {
            navigation.navigate('Report', {
              location: { latitude, longitude },
            });
          },
        },
        {
          text: '🔍 장소 정보 보기',
          onPress: async () => {
            try {
              // 좌표로 역지오코딩하여 장소 정보 조회
              const response = await mapAPI.reverseGeocode(latitude, longitude);
              if (response.data) {
                const placeData = response.data;
                openPlaceSheet({
                  id: placeData.id,
                  latitude: placeData.latitude,
                  longitude: placeData.longitude,
                  name: placeData.name || '선택한 위치',
                  address: placeData.description || `${latitude.toFixed(4)}, ${longitude.toFixed(4)}`,
                  category: placeData.category || 'other',
                  description: placeData.description,
                  type: 'osm',
                });
              }
            } catch (error) {
              console.error('Failed to reverse geocode:', error);
              // 에러 시에도 기본 정보로 PlaceDetailSheet 열기
              openPlaceSheet({
                latitude,
                longitude,
                name: '선택한 위치',
                address: `${latitude.toFixed(4)}, ${longitude.toFixed(4)}`,
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
        <Text style={{ marginTop: Spacing.md, color: Colors.textSecondary }}>
          지도 로딩 중...
        </Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <MapView
        ref={mapRef}
        style={styles.map}
        initialRegion={JUBA_CENTER}
        mapType="none"
        showsUserLocation={false}
        showsMyLocationButton={false}
        showsZoomControls={false}
        zoomControlEnabled={false}
        onPress={handleMapPress}
        onLongPress={handleLongPress}
        onPanDrag={() => {
          // 사용자가 지도를 드래그하기 시작하면 플래그 설정
          isUserPanningRef.current = true;
        }}
        onRegionChangeComplete={(region) => {
          // 사용자가 직접 지도를 이동시킨 경우에만 region 업데이트
          // onPanDrag가 호출된 후에만 업데이트 (지도 클릭으로 인한 변경 제외)
          if (isUserPanningRef.current) {
            setMapRegion(region);
            isUserPanningRef.current = false; // 플래그 리셋
          }
        }}
        onMapReady={() => {
          console.log('[MapScreen] MapView ready');
        }}
        onError={(error) => {
          console.error('[MapScreen] MapView error:', error);
        }}
      >
        {/* OpenStreetMap 타일 */}
        <UrlTile
          urlTemplate="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
          maximumZ={19}
          flipY={false}
        />
        {/* 위험 정보 마커 - 경로가 있으면 경로 근처 위험 정보만, 없으면 전체 위험 정보 */}
        {(() => {
          // 위험 유형을 category로 변환
            const getHazardCategory = (hazardType) => {
            const categoryMap = {
              'armed_conflict': 'danger',
              'conflict': 'danger',
              'protest_riot': 'danger',
              'protest': 'danger',
              'checkpoint': 'danger',
              'road_damage': 'danger',
              'natural_disaster': 'danger',
              'flood': 'danger',
              'landslide': 'danger',
              'other': 'danger',
            };
            return categoryMap[hazardType] || 'danger';
          };

          const getHazardName = (hazardType) => {
            const nameMap = {
              'armed_conflict': '무력충돌',
              'conflict': '충돌',
              'protest_riot': '시위/폭동',
              'protest': '시위',
              'checkpoint': '검문소',
              'road_damage': '도로 손상',
              'natural_disaster': '자연재해',
              'flood': '홍수',
              'landslide': '산사태',
              'safe_haven': '대피처',
              'other': '기타 위험',
            };
            return nameMap[hazardType] || '알 수 없음';
          };

          console.log('[MapScreen DEBUG] 렌더링 체크:');
          console.log('[MapScreen DEBUG] - activeHazardTypes:', activeHazardTypes);
          console.log('[MapScreen DEBUG] - hazards 개수:', hazards?.length || 0);
          console.log('[MapScreen DEBUG] - activeHazardTypes.length === 0:', activeHazardTypes.length === 0);

          // 활성화된 위험 유형이 있고 위험 정보가 있을 때만 표시
          // (모든 유형이 제외되거나 위험 정보가 없으면 아무것도 표시하지 않음)
          if (activeHazardTypes.length === 0 || !hazards || hazards.length === 0) {
            console.log('[MapScreen DEBUG] ⚠️ 위험 정보 렌더링 건너뜀 (조건 불만족)');
            return [];
          }

          // 기본 지도의 위험 정보를 사용 (경로 선택 여부와 관계없이 동일한 위험 정보 표시)
          // 1. 위험 유형 필터 적용
          const typeFilteredHazards = hazards.filter(hazard => hazard && activeHazardTypes.includes(hazard.hazard_type));
          // 2. 시간 필터 적용
          const hazardsToShow = getFilteredHazardsByTime(typeFilteredHazards);
          console.log('[MapScreen DEBUG] ✅ 렌더링할 hazards:', hazardsToShow.length, '개 (시간 필터:', timeFilter, ')');

          return hazardsToShow.map((hazard) => {
            // 위험 정보 반경 (km → m 변환) - 백엔드에서 제공하는 radius 사용
            // 스코어링 테이블의 default_radius_km 값이 사용됨
            const radiusMeters = (hazard.radius || 0.1) * 1000; // km → m 변환
            const riskColor = getRiskColor(hazard.risk_score);
            
            return (
              <React.Fragment key={hazard.id}>
                {/* 위험 범위 Circle */}
                <Circle
                  center={{
                    latitude: hazard.latitude,
                    longitude: hazard.longitude,
                  }}
                  radius={radiusMeters}
                  fillColor={`${riskColor}30`} // 30% 투명도
                  strokeColor={riskColor}
                  strokeWidth={2}
                />
                
                {/* 위험 정보 마커 */}
                <Marker
                  coordinate={{
                    latitude: hazard.latitude,
                    longitude: hazard.longitude,
                  }}
                  title={`위험: ${getHazardName(hazard.hazard_type)}`}
                  description={hazard.description}
                  pinColor={riskColor}
                  onPress={() => {
                    // 위험 정보 클릭 시 장소 정보 카드 표시
                    openPlaceSheet({
                      id: hazard.id,
                      name: getHazardName(hazard.hazard_type),
                      address: hazard.description || '',
                      latitude: hazard.latitude,
                      longitude: hazard.longitude,
                      category: getHazardCategory(hazard.hazard_type),
                      description: hazard.description,
                      risk_score: hazard.risk_score,
                      hazard_type: hazard.hazard_type,
                      type: 'hazard',
                    });
                  }}
                />
              </React.Fragment>
            );
          });
        })()}

        {/* 사용자 위치 - 파란색 동그라미 (고정 크기) */}
        {userLocation && (
          <>
            <Circle
              center={{
                latitude: userLocation.latitude,
                longitude: userLocation.longitude,
              }}
              radius={30} // 30m 반경 (더 작게)
              fillColor="rgba(0, 71, 171, 0.4)" // 파란색 채우기 (더 진하게)
              strokeColor="#0047AB" // 파란색 테두리
              strokeWidth={3}
            />
            {/* 고정 크기 마커 (줌 레벨과 관계없이 동일한 크기) */}
            <Marker
              coordinate={{
                latitude: userLocation.latitude,
                longitude: userLocation.longitude,
              }}
              anchor={{ x: 0.5, y: 0.5 }}
            >
              <View style={styles.userLocationMarker}>
                <View style={styles.userLocationDot} />
              </View>
            </Marker>
          </>
        )}

        {/* 경로 폴리라인 - 모든 경로 표시 */}
        {(routeResponse?.routes || routes || []).map((route) => {
          const routeCoordinates = route.polyline?.map(coord => ({
            latitude: coord[0],
            longitude: coord[1],
          }));
          
          if (!routeCoordinates || routeCoordinates.length === 0) return null;

          // Phase 2: 경로 타입별 색상 구분
          // 선택된 경로는 더 두껍고 진한 색으로, 다른 경로는 얇고 연한 색으로
          const isSelected = selectedRoute?.id === route.id;
          const baseColor = getRouteColor(route.type); // safe=초록, fast=파랑
          const strokeColor = isSelected ? baseColor : baseColor + "80"; // 선택: 진한 색, 비선택: 반투명
          const strokeWidth = isSelected ? 8 : 4; // 선택: 8px, 비선택: 4px

          return (
            <Polyline
              key={route.id}
              coordinates={routeCoordinates}
              strokeColor={strokeColor}
              strokeWidth={strokeWidth}
              lineCap="round"
              lineJoin="round"
            />
          );
        })}

        {/* 출발지 - 사용자 위치와 동일한 스타일 (고정 크기) */}
        {((selectedRoute || routes.length > 0 || routeResponse?.routes?.length > 0) && startLocation) && (
          <>
            <Circle
              center={{
                latitude: startLocation.lat,
                longitude: startLocation.lng,
              }}
              radius={30} // 30m 반경 (사용자 위치와 동일)
              fillColor="rgba(0, 71, 171, 0.4)" // 파란색 채우기 (사용자 위치와 동일)
              strokeColor="#0047AB" // 파란색 테두리
              strokeWidth={3}
            />
            {/* 고정 크기 마커 (줌 레벨과 관계없이 동일한 크기) */}
            <Marker
              coordinate={{
                latitude: startLocation.lat,
                longitude: startLocation.lng,
              }}
              anchor={{ x: 0.5, y: 0.5 }}
            >
              <View style={styles.startLocationMarker}>
                <View style={styles.startLocationDot} />
              </View>
            </Marker>
          </>
        )}

        {/* 도착지 - 사용자 위치와 동일한 스타일 (고정 크기) */}
        {((selectedRoute || routes.length > 0 || routeResponse?.routes?.length > 0) && endLocation) && (
          <>
            <Circle
              center={{
                latitude: endLocation.lat,
                longitude: endLocation.lng,
              }}
              radius={30} // 30m 반경 (사용자 위치와 동일)
              fillColor="rgba(0, 71, 171, 0.4)" // 파란색 채우기 (사용자 위치와 동일)
              strokeColor="#0047AB" // 파란색 테두리
              strokeWidth={3}
            />
            {/* 고정 크기 마커 (줌 레벨과 관계없이 동일한 크기) */}
            <Marker
              coordinate={{
                latitude: endLocation.lat,
                longitude: endLocation.lng,
              }}
              anchor={{ x: 0.5, y: 0.5 }}
            >
              <View style={styles.endLocationMarker}>
                <View style={styles.endLocationDot} />
              </View>
            </Marker>
          </>
        )}
      </MapView>

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

      {/* SOS 긴급 버튼 */}
      <TouchableOpacity
        style={styles.sosButton}
        onPress={handleSOSButtonPress}
        activeOpacity={0.8}
      >
        <Icon name="warning" size={28} color={Colors.textInverse} />
        <Text style={styles.sosButtonText}>SOS</Text>
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
        timeFilter={timeFilter}
        onTimeFilterChange={handleTimeFilterChange}
      />

      {/* SOS 확인 모달 */}
      <SOSConfirmModal
        visible={isSOSModalOpen}
        onConfirm={handleSOSConfirm}
        onCancel={handleSOSCancel}
        emergencyContactsCount={emergencyContacts.length}
        userLocation={userLocation}
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
  },
  sosButton: {
    position: 'absolute',
    left: Spacing.lg,
    bottom: Spacing.xl + 16,
    width: 72,
    height: 72,
    borderRadius: 36,
    backgroundColor: Colors.danger,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: Colors.danger,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.5,
    shadowRadius: 12,
    elevation: 10,
  },
  sosButtonText: {
    ...Typography.labelSmall,
    color: Colors.textInverse,
    fontWeight: '700',
    fontSize: 11,
    marginTop: 2,
  },
  userLocationMarker: {
    width: 20,
    height: 20,
    borderRadius: 10,
    backgroundColor: '#0047AB',
    borderWidth: 3,
    borderColor: '#FFFFFF',
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
    elevation: 5,
  },
  userLocationDot: {
    width: '100%',
    height: '100%',
    borderRadius: 10,
    backgroundColor: '#0047AB',
  },
  startLocationMarker: {
    width: 20,
    height: 20,
    borderRadius: 10,
    backgroundColor: 'transparent',
    justifyContent: 'center',
    alignItems: 'center',
  },
  startLocationDot: {
    width: '100%',
    height: '100%',
    borderRadius: 10,
    backgroundColor: '#0047AB',
  },
  endLocationMarker: {
    width: 20,
    height: 20,
    borderRadius: 10,
    backgroundColor: 'transparent',
    justifyContent: 'center',
    alignItems: 'center',
  },
  endLocationDot: {
    width: '100%',
    height: '100%',
    borderRadius: 10,
    backgroundColor: '#0047AB',
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

