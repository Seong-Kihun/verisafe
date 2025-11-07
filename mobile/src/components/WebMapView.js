/**
 * WebMapView - 웹용 지도 컴포넌트 (react-leaflet 사용)
 * 웹 환경에서만 사용되며, 모바일과 동일한 기능 제공
 */

import React, { useEffect, useRef, useMemo } from 'react';
import { View, StyleSheet } from 'react-native';

// 웹 환경에서만 react-leaflet 동적 로드
let MapContainer, TileLayer, Marker, Popup, Polyline, Circle, useMapEvents;
if (typeof window !== 'undefined' && typeof document !== 'undefined') {
  try {
    const L = require('react-leaflet');
    MapContainer = L.MapContainer;
    TileLayer = L.TileLayer;
    Marker = L.Marker;
    Popup = L.Popup;
    Polyline = L.Polyline;
    Circle = L.Circle;
    useMapEvents = L.useMapEvents;

    // Leaflet CSS 동적 로드
    if (!document.querySelector('link[href*="leaflet"]')) {
      const link = document.createElement('link');
      link.rel = 'stylesheet';
      link.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';
      link.integrity = 'sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=';
      link.crossOrigin = '';
      document.head.appendChild(link);
    }

    // Leaflet 아이콘 이미지 경로 설정
    if (typeof window.L !== 'undefined') {
      delete window.L.Icon.Default.prototype._getIconUrl;
      window.L.Icon.Default.mergeOptions({
        iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
        iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
        shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
      });
    }
  } catch (e) {
    console.warn('react-leaflet not available:', e);
  }
}

const JUBA_CENTER = {
  lat: 4.8594,
  lng: 31.5713,
};

// latitudeDelta를 zoom 레벨로 변환
const getZoomFromDelta = (delta) => {
  if (delta > 0.1) return 10;
  if (delta > 0.05) return 11;
  if (delta > 0.02) return 12;
  if (delta > 0.01) return 13;
  return 14;
};

export default function WebMapView({
  landmarks = [],
  hazards = [],
  routeResponse,
  selectedRoute, // RoutePlanningContext에서 선택한 경로
  routes, // RoutePlanningContext의 모든 경로
  activeHazardTypes = [], // 위험 유형 필터 배열
  mapRegion,
  userLocation,
  startLocation, // 출발지
  endLocation, // 목적지
  style,
  onPress,
  onDoublePress,
  onLongPress,
  markers = [], // 추가 마커 (ReportScreen에서 사용)
  onMarkerPress, // 마커 클릭 핸들러 (MapScreen에서 전달)
}) {
  // 웹 환경이 아니거나 react-leaflet이 없으면 빈 View 반환
  if (typeof window === 'undefined' || typeof document === 'undefined' || !MapContainer) {
    return <View style={[styles.container, style]} />;
  }

  // 위험 유형 필터링 (중복 선택 가능)
  // 선택된 위험 유형이 있을 때만 표시 (초기에는 아무것도 표시하지 않음)
  const filteredHazards = useMemo(() =>
    activeHazardTypes.length === 0
      ? []
      : hazards.filter(hazard => activeHazardTypes.includes(hazard.hazard_type)),
    [hazards, activeHazardTypes]
  );

  const center = mapRegion 
    ? [mapRegion.latitude, mapRegion.longitude]
    : [JUBA_CENTER.lat, JUBA_CENTER.lng];
  
  const zoom = mapRegion 
    ? getZoomFromDelta(mapRegion.latitudeDelta)
    : 11;

  // 웹에서는 div를 사용 (react-leaflet은 DOM 요소 필요)
  // React Native Web의 createElement를 사용하여 div 생성
  const MapWrapper = ({ children, ...props }) => {
    // 웹 환경에서는 div, 모바일에서는 View
    if (typeof document !== 'undefined') {
      return React.createElement('div', props, children);
    }
    return <View {...props}>{children}</View>;
  };

  // MapContainer 내부에서 클릭 및 롱 프레스 이벤트를 처리하는 컴포넌트
  const MapClickHandler = () => {
    const longPressTimerRef = useRef(null);
    const longPressTriggeredRef = useRef(false);

    // Cleanup timer on unmount to prevent memory leaks
    useEffect(() => {
      return () => {
        if (longPressTimerRef.current) {
          clearTimeout(longPressTimerRef.current);
          longPressTimerRef.current = null;
        }
      };
    }, []);

    if (!useMapEvents) return null;

    useMapEvents({
      mousedown: (e) => {
        // 롱 프레스 타이머 시작 (500ms)
        longPressTriggeredRef.current = false;
        longPressTimerRef.current = setTimeout(() => {
          longPressTriggeredRef.current = true;
          const { lat, lng } = e.latlng;
          if (onLongPress) {
            onLongPress(lat, lng);
          }
        }, 500);
      },
      mouseup: (e) => {
        // 타이머 취소
        if (longPressTimerRef.current) {
          clearTimeout(longPressTimerRef.current);
          longPressTimerRef.current = null;
        }
      },
      mousemove: (e) => {
        // 마우스가 움직이면 롱 프레스 취소
        if (longPressTimerRef.current) {
          clearTimeout(longPressTimerRef.current);
          longPressTimerRef.current = null;
        }
      },
      click: (e) => {
        // 롱 프레스가 아닐 때만 클릭 처리
        if (!longPressTriggeredRef.current) {
          const { lat, lng } = e.latlng;
          // onPress가 있으면 호출 (역지오코딩용)
          if (onPress) {
            onPress(lat, lng);
          }
        }
        longPressTriggeredRef.current = false;
      },
    });

    return null;
  };

  return (
    <MapWrapper style={{ width: '100%', height: '100%', position: 'relative' }}>
      <MapContainer
        key="map-container" // key를 고정하여 불필요한 리마운트 방지
        center={center}
        zoom={zoom}
        style={{ width: '100%', height: '100%' }}
        zoomControl={false}
        whenCreated={(mapInstance) => {
          // 지도 인스턴스 생성 후 이벤트 리스너 설정
          // 사용자가 직접 지도를 이동시킬 때만 center 업데이트
          mapInstance.on('moveend', () => {
            const center = mapInstance.getCenter();
            const zoom = mapInstance.getZoom();
            // 이벤트는 처리하지만 mapRegion 업데이트는 하지 않음
            // (지도 클릭으로 인한 새로고침 방지)
          });
        }}
      >
        <MapClickHandler />
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {/* 위험 정보 마커 및 범위 Circle */}
        {filteredHazards.map((hazard) => {
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
              'other': '기타 위험',
            };
            return nameMap[hazardType] || '위험 지역';
          };

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

          // 위험 정보 반경 (km → m 변환) - 백엔드에서 제공하는 radius 사용
          // 스코어링 테이블의 default_radius_km 값이 사용됨
          const radiusMeters = (hazard.radius || 0.1) * 1000; // km → m 변환
          
          // 위험도에 따른 색상 (간단한 함수)
          const getRiskColor = (score) => {
            if (score >= 70) return '#F44336'; // red
            if (score >= 50) return '#FF9800'; // orange
            return '#4CAF50'; // green
          };
          
          const riskColor = getRiskColor(hazard.risk_score);
          
          // 색상을 rgba로 변환 (30% 투명도)
          const hexToRgba = (hex, alpha) => {
            const r = parseInt(hex.slice(1, 3), 16);
            const g = parseInt(hex.slice(3, 5), 16);
            const b = parseInt(hex.slice(5, 7), 16);
            return `rgba(${r}, ${g}, ${b}, ${alpha})`;
          };

          return (
            <React.Fragment key={hazard.id}>
              {/* 위험 범위 Circle */}
              {Circle && (
                <Circle
                  center={[hazard.latitude, hazard.longitude]}
                  radius={radiusMeters}
                  pathOptions={{
                    fillColor: hexToRgba(riskColor, 0.3),
                    color: riskColor,
                    weight: 2,
                    fillOpacity: 0.3,
                  }}
                />
              )}
              
              {/* 위험 정보 마커 */}
              <Marker
                position={[hazard.latitude, hazard.longitude]}
                eventHandlers={{
                  click: () => {
                    if (onMarkerPress) {
                      onMarkerPress({
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
                    }
                  },
                }}
              >
                <Popup>
                  <div>
                    <strong>위험: {getHazardName(hazard.hazard_type)}</strong>
                    <br />
                    {hazard.description}
                    <br />
                    <small>위험도: {hazard.risk_score}</small>
                  </div>
                </Popup>
              </Marker>
            </React.Fragment>
          );
        })}

        {/* 사용자 위치 - 파란색 동그라미 + 고정 크기 마커 */}
        {userLocation && (
          <>
            {Circle && (
              <Circle
                center={[userLocation.latitude, userLocation.longitude]}
                radius={30} // 30m 반경 (더 작게)
                pathOptions={{
                  fillColor: 'rgba(0, 71, 171, 0.4)', // 파란색 채우기 (더 진하게)
                  color: '#0047AB', // 파란색 테두리
                  weight: 3,
                  fillOpacity: 0.4,
                }}
              />
            )}
            {/* 고정 크기 마커 (줌 레벨과 관계없이 동일한 크기) */}
            <Marker
              position={[userLocation.latitude, userLocation.longitude]}
              icon={typeof window !== 'undefined' && window.L ? new window.L.DivIcon({
                className: 'custom-user-location-marker',
                html: `<div style="width: 20px; height: 20px; border-radius: 50%; background-color: #0047AB; border: 3px solid #FFFFFF; box-shadow: 0 2px 4px rgba(0,0,0,0.3);"></div>`,
                iconSize: [20, 20],
                iconAnchor: [10, 10],
              }) : undefined}
            />
          </>
        )}

        {/* 경로 폴리라인 - 모든 경로 표시 */}
        {(routeResponse?.routes || routes || []).map((route) => {
          const routeCoordinates = route.polyline?.map(coord => [coord[0], coord[1]]);

          if (!routeCoordinates || routeCoordinates.length === 0) return null;

          // Phase 2: 경로 타입별 색상 구분
          // 선택된 경로는 더 두껍고 진한 색으로, 다른 경로는 얇고 연한 색으로
          const isSelected = selectedRoute?.id === route.id;

          // 경로 타입에 따른 색상 설정 (safe=초록, fast=파랑)
          const getRouteColor = (routeType) => {
            const colorMap = {
              'safe': '#10B981',   // 초록색 - 안전 경로
              'fast': '#0066CC',   // 파란색 - 빠른 경로
              'alternative': '#F59E0B', // 주황색 - 대안 경로
            };
            return colorMap[routeType] || '#0066CC';
          };

          const baseColor = getRouteColor(route.type);
          const strokeColor = isSelected ? baseColor : baseColor + "80"; // 선택: 진한 색, 비선택: 반투명
          const strokeWidth = isSelected ? 8 : 4; // 선택: 8px, 비선택: 4px

          return (
            <Polyline
              key={route.id}
              positions={routeCoordinates}
              pathOptions={{
                color: strokeColor,
                weight: strokeWidth,
                opacity: 1,
                lineCap: 'round',
                lineJoin: 'round',
              }}
            />
          );
        })}

        {/* 출발지 - 사용자 위치와 동일한 스타일 (고정 크기) */}
        {((selectedRoute || routes?.length > 0 || routeResponse?.routes?.length > 0) && startLocation) && (
          <>
            {Circle && (
              <Circle
                center={[startLocation.lat, startLocation.lng]}
                radius={30} // 30m 반경 (사용자 위치와 동일)
                pathOptions={{
                  fillColor: 'rgba(0, 71, 171, 0.4)', // 파란색 채우기 (사용자 위치와 동일)
                  color: '#0047AB', // 파란색 테두리
                  weight: 3,
                  fillOpacity: 0.4,
                }}
              />
            )}
            {/* 고정 크기 마커 (줌 레벨과 관계없이 동일한 크기) */}
            <Marker
              position={[startLocation.lat, startLocation.lng]}
              icon={typeof window !== 'undefined' && window.L ? new window.L.DivIcon({
                className: 'custom-start-location-marker',
                html: `<div style="width: 20px; height: 20px; border-radius: 50%; background-color: #0047AB; border: 3px solid #FFFFFF; box-shadow: 0 2px 4px rgba(0,0,0,0.3);"></div>`,
                iconSize: [20, 20],
                iconAnchor: [10, 10],
              }) : undefined}
            />
          </>
        )}

        {/* 도착지 - 사용자 위치와 동일한 스타일 (고정 크기) */}
        {((selectedRoute || routes?.length > 0 || routeResponse?.routes?.length > 0) && endLocation) && (
          <>
            {Circle && (
              <Circle
                center={[endLocation.lat, endLocation.lng]}
                radius={30} // 30m 반경 (사용자 위치와 동일)
                pathOptions={{
                  fillColor: 'rgba(0, 71, 171, 0.4)', // 파란색 채우기 (사용자 위치와 동일)
                  color: '#0047AB', // 파란색 테두리
                  weight: 3,
                  fillOpacity: 0.4,
                }}
              />
            )}
            {/* 고정 크기 마커 (줌 레벨과 관계없이 동일한 크기) */}
            <Marker
              position={[endLocation.lat, endLocation.lng]}
              icon={typeof window !== 'undefined' && window.L ? new window.L.DivIcon({
                className: 'custom-end-location-marker',
                html: `<div style="width: 20px; height: 20px; border-radius: 50%; background-color: #0047AB; border: 3px solid #FFFFFF; box-shadow: 0 2px 4px rgba(0,0,0,0.3);"></div>`,
                iconSize: [20, 20],
                iconAnchor: [10, 10],
              }) : undefined}
            />
          </>
        )}

        {/* 추가 마커 (ReportScreen 등에서 사용) */}
        {markers.map((marker, index) => (
          <Marker
            key={marker.id || index}
            position={[marker.latitude, marker.longitude]}
          >
            {marker.title && (
              <Popup>
                <div>{marker.title}</div>
              </Popup>
            )}
          </Marker>
        ))}
      </MapContainer>
    </MapWrapper>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
});
