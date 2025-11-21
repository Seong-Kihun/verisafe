/**
 * NavigationScreen - 실시간 경로 안내 화면
 * 턴바이턴 네비게이션 전용 전체 화면
 */

import React, { useRef, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  StatusBar,
} from 'react-native';
import MapView, { Polyline, Marker, Circle } from 'react-native-maps';
import { Colors, Typography, Spacing } from '../styles';
import { useNavigation as useNav } from '../contexts/NavigationContext';
import Icon from '../components/icons/Icon';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

export default function NavigationScreen() {
  console.log('[NavigationScreen] 컴포넌트 렌더링');

  const insets = useSafeAreaInsets();
  const mapRef = useRef(null);
  const { navigationState, route, stopNavigation, isVoiceEnabled, toggleVoiceGuidance } = useNav();

  console.log('[NavigationScreen] navigationState:', navigationState);
  console.log('[NavigationScreen] route:', route);

  // 지도를 현재 위치로 자동 이동
  useEffect(() => {
    console.log('[NavigationScreen] useEffect - currentLocation:', navigationState?.currentLocation);
    if (mapRef.current && navigationState?.currentLocation) {
      mapRef.current.animateCamera({
        center: navigationState.currentLocation,
        zoom: 17,
        heading: navigationState.heading || 0,
        pitch: 60, // 3D 각도
      }, { duration: 500 });
    }
  }, [navigationState?.currentLocation]);

  const handleExit = () => {
    Alert.alert(
      '네비게이션 종료',
      '안내를 종료하시겠습니까?',
      [
        { text: '취소', style: 'cancel' },
        { text: '종료', style: 'destructive', onPress: stopNavigation }
      ]
    );
  };

  if (!navigationState || !route) {
    return (
      <View style={styles.container}>
        <Text style={styles.loadingText}>네비게이션을 준비 중입니다...</Text>
      </View>
    );
  }

  const { nextInstruction, hazardWarning, formattedDistance, formattedTime, progress } = navigationState;

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" />

      {/* 다음 안내 카드 */}
      <View style={[styles.instructionCard, { paddingTop: insets.top + 10 }]}>
        <View style={styles.instructionHeader}>
          <View style={styles.directionIcon}>
            <Text style={styles.directionIconText}>{getDirectionIcon(nextInstruction?.direction)}</Text>
          </View>
          <View style={styles.instructionTextContainer}>
            <Text style={styles.instructionText}>{nextInstruction?.instruction || '직진'}</Text>
            <Text style={styles.distanceText}>{nextInstruction?.distance || 0}m 후</Text>
          </View>
          <TouchableOpacity style={styles.voiceButton} onPress={toggleVoiceGuidance}>
            <Icon
              name={isVoiceEnabled ? 'volumeUp' : 'volumeOff'}
              size={24}
              color={isVoiceEnabled ? Colors.primary : Colors.textTertiary}
            />
          </TouchableOpacity>
        </View>
      </View>

      {/* 지도 */}
      <MapView
        ref={mapRef}
        style={styles.map}
        initialRegion={{
          ...route.coordinates[0],
          latitudeDelta: 0.01,
          longitudeDelta: 0.01,
        }}
        showsUserLocation={true}
        showsMyLocationButton={false}
        showsCompass={false}
        rotateEnabled={true}
        pitchEnabled={true}
      >
        {/* 경로 라인 */}
        <Polyline
          coordinates={route.coordinates}
          strokeColor={Colors.primary}
          strokeWidth={5}
        />

        {/* 목적지 마커 */}
        <Marker
          coordinate={route.coordinates[route.coordinates.length - 1]}
          title="목적지"
        >
          <View style={styles.destinationMarker}>
            <Icon name="flag" size={24} color={Colors.textInverse} />
          </View>
        </Marker>

        {/* 위험 마커들 */}
        {route.hazards && route.hazards.map((hazard, index) => (
          <Circle
            key={`hazard_${index}`}
            center={{ latitude: hazard.latitude, longitude: hazard.longitude }}
            radius={50}
            fillColor="rgba(239, 68, 68, 0.2)"
            strokeColor="rgba(239, 68, 68, 0.8)"
            strokeWidth={2}
          />
        ))}
      </MapView>

      {/* 위험 경고 배너 */}
      {hazardWarning && (
        <View style={styles.hazardBanner}>
          <Icon name="warning" size={20} color={Colors.warning} />
          <Text style={styles.hazardText}>{hazardWarning.message}</Text>
        </View>
      )}

      {/* 하단 정보 패널 */}
      <View style={[styles.bottomPanel, { paddingBottom: insets.bottom + 10 }]}>
        {/* 진행 바 */}
        <View style={styles.progressBar}>
          <View style={[styles.progressFill, { width: `${progress}%` }]} />
        </View>

        <View style={styles.infoRow}>
          <View style={styles.infoItem}>
            <Icon name="locationOn" size={20} color={Colors.primary} />
            <Text style={styles.infoLabel}>남은 거리</Text>
            <Text style={styles.infoValue}>{formattedDistance}</Text>
          </View>

          <View style={styles.divider} />

          <View style={styles.infoItem}>
            <Icon name="time" size={20} color={Colors.primary} />
            <Text style={styles.infoLabel}>예상 시간</Text>
            <Text style={styles.infoValue}>{formattedTime}</Text>
          </View>

          <View style={styles.divider} />

          <TouchableOpacity style={styles.exitButton} onPress={handleExit}>
            <Icon name="close" size={24} color={Colors.danger} />
            <Text style={styles.exitText}>종료</Text>
          </TouchableOpacity>
        </View>
      </View>
    </View>
  );
}

/**
 * 방향에 따른 아이콘 이모지
 */
const getDirectionIcon = (direction) => {
  const icons = {
    straight: '↑',
    slight_left: '↖',
    left: '←',
    sharp_left: '↙',
    slight_right: '↗',
    right: '→',
    sharp_right: '↘',
    u_turn: '⤴',
  };
  return icons[direction] || '↑';
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  loadingText: {
    ...Typography.body,
    color: Colors.textSecondary,
    textAlign: 'center',
    marginTop: 100,
  },
  instructionCard: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    backgroundColor: Colors.surface,
    paddingHorizontal: Spacing.lg,
    paddingBottom: Spacing.md,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
    zIndex: 10,
  },
  instructionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  directionIcon: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: Colors.primary,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: Spacing.md,
  },
  directionIconText: {
    fontSize: 32,
    color: Colors.textInverse,
  },
  instructionTextContainer: {
    flex: 1,
  },
  instructionText: {
    ...Typography.h2,
    color: Colors.textPrimary,
    fontWeight: '700',
  },
  distanceText: {
    ...Typography.body,
    color: Colors.textSecondary,
    marginTop: 4,
  },
  voiceButton: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: Colors.primaryLight + '20',
    justifyContent: 'center',
    alignItems: 'center',
  },
  map: {
    flex: 1,
  },
  destinationMarker: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: Colors.success,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
    elevation: 4,
  },
  hazardBanner: {
    position: 'absolute',
    top: 150,
    left: Spacing.lg,
    right: Spacing.lg,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: Colors.warning + 'E6',
    paddingVertical: Spacing.sm,
    paddingHorizontal: Spacing.md,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 4,
    elevation: 4,
    zIndex: 5,
  },
  hazardText: {
    ...Typography.body,
    color: Colors.textInverse,
    fontWeight: '600',
    marginLeft: Spacing.sm,
    flex: 1,
  },
  bottomPanel: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: Colors.surface,
    paddingHorizontal: Spacing.lg,
    paddingTop: Spacing.md,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: -4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  progressBar: {
    height: 4,
    backgroundColor: Colors.border,
    borderRadius: 2,
    marginBottom: Spacing.md,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    backgroundColor: Colors.primary,
  },
  infoRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  infoItem: {
    flex: 1,
    alignItems: 'center',
  },
  infoLabel: {
    ...Typography.caption,
    color: Colors.textSecondary,
    marginTop: 4,
  },
  infoValue: {
    ...Typography.h3,
    color: Colors.textPrimary,
    fontWeight: '700',
    marginTop: 2,
  },
  divider: {
    width: 1,
    height: 40,
    backgroundColor: Colors.border,
  },
  exitButton: {
    flex: 1,
    alignItems: 'center',
  },
  exitText: {
    ...Typography.caption,
    color: Colors.danger,
    marginTop: 4,
    fontWeight: '600',
  },
});
