/**
 * PlaceDetailSheet - 장소 상세 하단 시트
 * 
 * 책임:
 * 1. 선택된 장소 정보 표시
 * 2. [경로] [저장] [공유] 버튼 제공
 * 3. 닫기 기능
 */

import React, { useRef, useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  Alert,
  Animated,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import { Colors, Spacing, Typography, getRiskColor } from '../styles';
import { useMapContext } from '../contexts/MapContext';
import { savedPlacesStorage } from '../services/storage';
import Icon from './icons/Icon';

const CATEGORY_ICONS = {
  airport: 'flight',
  government: 'account-balance',
  hospital: 'local-hospital',
  hotel: 'hotel',
  danger: 'warning',
  other: 'location-on',
};

export default function PlaceDetailSheet() {
  const insets = useSafeAreaInsets();
  const navigation = useNavigation();
  const { selectedPlace, closePlaceSheet, isPlaceSheetOpen } = useMapContext();
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(100)).current;
  const [isSaved, setIsSaved] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  console.log('[PlaceDetailSheet] 렌더링:', { selectedPlace, isPlaceSheetOpen });

  // 저장 여부 확인
  useEffect(() => {
    const checkIfSaved = async () => {
      if (selectedPlace) {
        const savedPlaces = await savedPlacesStorage.getAll();
        const exists = savedPlaces.some(place => {
          // 좌표가 같으면 동일한 장소로 간주
          return Math.abs(place.latitude - selectedPlace.latitude) < 0.0001 &&
                 Math.abs(place.longitude - selectedPlace.longitude) < 0.0001;
        });
        setIsSaved(exists);
      }
    };
    checkIfSaved();
  }, [selectedPlace]);

  useEffect(() => {
    if (isPlaceSheetOpen && selectedPlace) {
      // Fade in + Slide up 애니메이션
      Animated.parallel([
        Animated.timing(fadeAnim, {
          toValue: 1,
          duration: 200,
          useNativeDriver: true,
        }),
        Animated.timing(slideAnim, {
          toValue: 0,
          duration: 300,
          useNativeDriver: true,
        }),
      ]).start();
    } else {
      // Fade out + Slide down
      Animated.parallel([
        Animated.timing(fadeAnim, {
          toValue: 0,
          duration: 200,
          useNativeDriver: true,
        }),
        Animated.timing(slideAnim, {
          toValue: 100,
          duration: 200,
          useNativeDriver: true,
        }),
      ]).start();
    }
  }, [isPlaceSheetOpen, selectedPlace]);

  if (!selectedPlace || !isPlaceSheetOpen) {
    console.log('[PlaceDetailSheet] 조건부 리턴:', { selectedPlace: !!selectedPlace, isPlaceSheetOpen });
    return null;
  }

  const handleRoute = () => {
    // RoutePlanningScreen으로 이동 (목적지 자동 입력)
    navigation.navigate('RoutePlanning', {
      destination: selectedPlace
    });
    closePlaceSheet();
  };

  const handleReport = () => {
    // ReportStack의 ReportCreate로 이동
    navigation.navigate('ReportStack', {
      screen: 'ReportCreate',
      params: {
        location: selectedPlace ? {
          latitude: selectedPlace.latitude,
          longitude: selectedPlace.longitude,
        } : null,
      }
    });
    closePlaceSheet();
  };

  const handleSave = async () => {
    if (isSaving) return;

    if (isSaved) {
      Alert.alert('알림', '이미 즐겨찾기에 저장된 장소입니다.');
      return;
    }

    setIsSaving(true);
    try {
      const placeToSave = {
        id: selectedPlace.id || `place_${Date.now()}`,
        name: selectedPlace.name,
        latitude: selectedPlace.latitude,
        longitude: selectedPlace.longitude,
        address: selectedPlace.address || selectedPlace.description || '',
        category: selectedPlace.category || 'other',
        description: selectedPlace.description || '',
      };

      const result = await savedPlacesStorage.add(placeToSave);

      if (result) {
        setIsSaved(true);
        Alert.alert(
          '저장 완료',
          '즐겨찾기에 저장되었습니다.\n\n프로필 페이지에서 확인할 수 있습니다.',
          [
            { text: '확인', style: 'default' },
            {
              text: '보러가기',
              onPress: () => {
                closePlaceSheet();
                navigation.navigate('ProfileStack', {
                  screen: 'SavedPlaces',
                });
              },
            },
          ]
        );
      } else {
        Alert.alert('오류', '저장에 실패했습니다. 다시 시도해주세요.');
      }
    } catch (error) {
      console.error('[PlaceDetailSheet] Failed to save place:', error);
      Alert.alert('오류', '저장 중 문제가 발생했습니다.');
    } finally {
      setIsSaving(false);
    }
  };

  const handleShare = () => {
    Alert.alert('알림', '공유 기능은 곧 제공됩니다.');
  };

  return (
    <Animated.View 
      style={[
        styles.container, 
        { 
          paddingBottom: insets.bottom,
          opacity: fadeAnim,
          transform: [{ translateY: slideAnim }],
        }
      ]}
    >
      <View style={styles.sheetHandle} />

      <ScrollView style={styles.content}>
        {/* 헤더 (닫기 버튼 포함) */}
        <View style={styles.header}>
          <View style={styles.headerLeft}>
          <View style={styles.iconContainer}>
              <Icon
                name={CATEGORY_ICONS[selectedPlace.category] || 'location-on'}
                size={32}
                color={selectedPlace.type === 'hazard' && selectedPlace.risk_score !== undefined ? getRiskColor(selectedPlace.risk_score) : Colors.primary}
              />
          </View>
          <View style={styles.headerContent}>
            <Text style={styles.title}>{selectedPlace.name}</Text>
            {selectedPlace.description && (
              <Text style={styles.description}>{selectedPlace.description}</Text>
            )}
            {selectedPlace.type === 'hazard' && selectedPlace.risk_score !== undefined && (
              <View style={styles.riskBadge}>
                <Text style={[styles.riskText, { color: getRiskColor(selectedPlace.risk_score) }]}>
                  위험도: {selectedPlace.risk_score}/100
                </Text>
              </View>
            )}
              {selectedPlace.latitude && selectedPlace.longitude && (
            <Text style={styles.location}>
              {selectedPlace.latitude.toFixed(4)}, {selectedPlace.longitude.toFixed(4)}
            </Text>
              )}
          </View>
          </View>
          <TouchableOpacity style={styles.closeButtonHeader} onPress={closePlaceSheet}>
            <Icon name="close" size={24} color={Colors.textSecondary} />
          </TouchableOpacity>
        </View>

        {/* 버튼들 - 가로 1줄 */}
        <View style={styles.buttonContainer}>
          <TouchableOpacity 
            style={styles.primaryButton} 
            onPress={handleRoute}
            activeOpacity={0.8}
          >
            <View style={styles.buttonContent}>
              <Icon name="route" size={18} color={Colors.textInverse} />
              <Text style={styles.primaryButtonText}>경로</Text>
            </View>
          </TouchableOpacity>
          
          <TouchableOpacity
            style={[styles.secondaryButton, isSaved && styles.savedButton]}
            onPress={handleSave}
            activeOpacity={0.8}
            disabled={isSaving}
          >
            <View style={styles.buttonContent}>
              <Icon
                name={isSaved ? "save" : "save"}
                size={18}
                color={isSaved ? Colors.success : Colors.textPrimary}
              />
              <Text style={[styles.secondaryButtonText, isSaved && styles.savedButtonText]}>
                {isSaving ? '저장 중...' : isSaved ? '저장됨' : '저장'}
              </Text>
            </View>
          </TouchableOpacity>
          
          <TouchableOpacity 
            style={styles.secondaryButton} 
            onPress={handleReport}
            activeOpacity={0.8}
          >
            <View style={styles.buttonContent}>
              <Icon name="report" size={18} color={Colors.textPrimary} />
              <Text style={styles.secondaryButtonText}>제보</Text>
            </View>
          </TouchableOpacity>
          
          <TouchableOpacity 
            style={styles.secondaryButton} 
            onPress={handleShare}
            activeOpacity={0.8}
          >
            <View style={styles.buttonContent}>
              <Icon name="share" size={18} color={Colors.textPrimary} />
              <Text style={styles.secondaryButtonText}>공유</Text>
            </View>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: Colors.surfaceElevated,
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    // 그림자 강화 (shadowLarge 적용)
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: -4 },
    shadowOpacity: 0.16,
    shadowRadius: 12,
    elevation: 8,
    maxHeight: '50%',
    zIndex: 2000, // 다른 요소들 위에 표시되도록
  },
  sheetHandle: {
    width: 40,
    height: 4,
    backgroundColor: Colors.border,
    borderRadius: 2,
    alignSelf: 'center',
    marginTop: Spacing.sm,
    marginBottom: Spacing.md,
  },
  content: {
    paddingHorizontal: Spacing.lg,
    paddingTop: Spacing.sm,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    marginBottom: Spacing.lg,
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  iconContainer: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: Colors.primary + '20',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: Spacing.md,
  },
  headerContent: {
    flex: 1,
  },
  closeButtonHeader: {
    width: 32,
    height: 32,
    justifyContent: 'center',
    alignItems: 'center',
    marginLeft: Spacing.md,
  },
  closeButtonHeaderText: {
    fontSize: 24,
    color: Colors.textSecondary,
    fontWeight: '300',
  },
  title: {
    ...Typography.h2,
    color: Colors.textPrimary,
    marginBottom: Spacing.xs,
  },
  description: {
    ...Typography.bodySmall,
    color: Colors.textSecondary,
    marginBottom: Spacing.xs,
  },
  location: {
    ...Typography.bodySmall,
    color: Colors.textTertiary,
  },
  riskBadge: {
    marginTop: Spacing.xs,
    marginBottom: Spacing.xs,
  },
  riskText: {
    ...Typography.labelSmall,
    fontWeight: '600',
  },
  buttonContainer: {
    flexDirection: 'row',
    marginBottom: Spacing.lg,
    gap: Spacing.sm,
  },
  primaryButton: {
    flex: 1,
    backgroundColor: Colors.primary,
    borderRadius: 12,
    paddingVertical: Spacing.md,
    alignItems: 'center',
    minHeight: Spacing.buttonHeight,
    justifyContent: 'center',
  },
  buttonContent: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.xs,
  },
  primaryButtonText: {
    ...Typography.buttonSmall,
    color: Colors.textInverse,
    fontWeight: '600',
  },
  secondaryButton: {
    flex: 1,
    backgroundColor: Colors.borderLight,
    borderRadius: 12,
    paddingVertical: Spacing.md,
    alignItems: 'center',
    minHeight: Spacing.buttonHeight,
    justifyContent: 'center',
  },
  secondaryButtonText: {
    ...Typography.buttonSmall,
    color: Colors.textPrimary,
    fontWeight: '500',
  },
  savedButton: {
    backgroundColor: Colors.success + '20',
    borderWidth: 1,
    borderColor: Colors.success + '40',
  },
  savedButtonText: {
    color: Colors.success,
    fontWeight: '600',
  },
});

