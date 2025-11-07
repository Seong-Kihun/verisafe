/**
 * ReportScreen.web.js - 4단계 제보 화면 (웹용 - 네이티브와 동일)
 * Phase 1-5 모든 기능 통합
 */

import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  ScrollView,
  Alert,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import { useRoute } from '@react-navigation/native';
import * as Location from 'expo-location';

import { Colors, Typography, Spacing } from '../styles';
import { reportAPI, mapAPI } from '../services/api';
import { myReportsStorage } from '../services/storage';

// 컴포넌트들
import Icon from '../components/icons/Icon';
import StepIndicator from '../components/StepIndicator';
import PhotoPicker from '../components/PhotoPicker';
import SeverityPicker from '../components/SeverityPicker';
import TimePicker from '../components/TimePicker';
import ReportPreview from '../components/ReportPreview';
import ReportSuccessModal from '../components/ReportSuccessModal';
import WebMapView from '../components/WebMapView'; // 웹용 지도

const HAZARD_TYPES = [
  { type: 'armed_conflict', label: '무력충돌', icon: 'conflict', color: '#DC2626' },
  { type: 'protest_riot', label: '시위/폭동', icon: 'protest', color: '#F59E0B' },
  { type: 'checkpoint', label: '검문소', icon: 'checkpoint', color: '#FF6B6B' },
  { type: 'road_damage', label: '도로 손상', icon: 'roadDamage', color: '#F97316' },
  { type: 'natural_disaster', label: '자연재해', icon: 'naturalDisaster', color: '#DC2626' },
  { type: 'other', label: '기타', icon: 'other', color: '#6B7280' },
];

// 조건부 질문 설정
const CONDITIONAL_QUESTIONS = {
  armed_conflict: [
    { key: 'gunfire', label: '총성/폭발음 들림?', type: 'boolean' },
    { key: 'military', label: '군대/무장단체 목격?', type: 'boolean' },
    { key: 'casualties', label: '부상자 있음?', type: 'boolean' },
  ],
  checkpoint: [
    { key: 'wait_time', label: '대기 시간 (분)', type: 'number' },
    { key: 'passable', label: '통과 가능?', type: 'boolean' },
    { key: 'documents', label: '요구 문서', type: 'text' },
  ],
  protest_riot: [
    { key: 'crowd_size', label: '인원 규모 (명)', type: 'number' },
    { key: 'violent', label: '폭력적?', type: 'boolean' },
  ],
  road_damage: [
    { key: 'severity', label: '통행 가능 여부', type: 'select', options: ['완전 차단', '주의 통행', '우회 필요'] },
  ],
};

export default function ReportScreen({ navigation }) {
  const route = useRoute();

  // 현재 단계 (1-4)
  const [currentStep, setCurrentStep] = useState(1);

  // 제보 데이터
  const [reportData, setReportData] = useState({
    hazardType: '',
    latitude: 4.8550,
    longitude: 31.5850,
    accuracy: null,
    severity: 'medium',
    reportedAt: new Date(),
    photos: [],
    description: '',
    conditionalData: {},
  });

  // UI 상태
  const [loading, setLoading] = useState(false);
  const [locationPermission, setLocationPermission] = useState(false);
  const [successModalVisible, setSuccessModalVisible] = useState(false);
  const [nearbyReports, setNearbyReports] = useState([]);

  // 지도 상태
  const [mapRegion, setMapRegion] = useState({
    latitude: 4.8550,
    longitude: 31.5850,
    latitudeDelta: 0.01,
    longitudeDelta: 0.01,
  });

  // route params에서 위치 정보 가져오기
  useEffect(() => {
    if (route.params?.location) {
      const { latitude, longitude } = route.params.location;
      updateLocation(latitude, longitude);
    }
  }, [route.params]);

  // 현재 위치 요청
  useEffect(() => {
    requestLocationPermission();
  }, []);

  // Step 3 진입 시 중복 제보 확인
  useEffect(() => {
    if (currentStep === 3 && reportData.hazardType) {
      checkNearbyReports();
    }
  }, [currentStep]);

  /**
   * 위치 권한 요청
   */
  const requestLocationPermission = async () => {
    try {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status === 'granted') {
        setLocationPermission(true);
        const location = await Location.getCurrentPositionAsync({ accuracy: Location.Accuracy.High });
        updateLocation(
          location.coords.latitude,
          location.coords.longitude,
          location.coords.accuracy
        );
      }
    } catch (error) {
      console.error('Location permission error:', error);
    }
  };

  /**
   * 위치 업데이트
   */
  const updateLocation = (lat, lng, accuracy = null) => {
    setReportData(prev => ({
      ...prev,
      latitude: lat,
      longitude: lng,
      accuracy,
    }));
    setMapRegion({
      latitude: lat,
      longitude: lng,
      latitudeDelta: 0.01,
      longitudeDelta: 0.01,
    });
  };

  /**
   * 현재 위치 사용
   */
  const handleUseCurrentLocation = async () => {
    if (!locationPermission) {
      await requestLocationPermission();
      return;
    }
    try {
      const location = await Location.getCurrentPositionAsync({ accuracy: Location.Accuracy.High });
      updateLocation(
        location.coords.latitude,
        location.coords.longitude,
        location.coords.accuracy
      );
    } catch (error) {
      Alert.alert('오류', '현재 위치를 가져올 수 없습니다.');
    }
  };

  /**
   * 지도 탭으로 위치 선택 (웹용)
   */
  const handleMapPress = (lat, lng) => {
    updateLocation(lat, lng);
  };

  /**
   * 중복 제보 확인
   */
  const checkNearbyReports = async () => {
    try {
      // 500m 반경 내 최근 제보 검색
      const response = await reportAPI.getNearby({
        latitude: reportData.latitude,
        longitude: reportData.longitude,
        radius: 0.5, // km
        hours: 24, // 최근 24시간
      });

      if (response.data?.reports && response.data.reports.length > 0) {
        setNearbyReports(response.data.reports);

        // 같은 유형의 제보가 있으면 경고
        const sameType = response.data.reports.filter(r => r.hazard_type === reportData.hazardType);
        if (sameType.length > 0) {
          Alert.alert(
            '중복 제보',
            `인근에 ${sameType.length}건의 유사한 제보가 있습니다. 계속하시겠습니까?`,
            [
              { text: '취소', style: 'cancel' },
              { text: '계속', style: 'default' },
            ]
          );
        }
      }
    } catch (error) {
      console.error('Failed to check nearby reports:', error);
    }
  };

  /**
   * 다음 단계로 이동
   */
  const goToNextStep = () => {
    // 각 단계 유효성 검사
    if (currentStep === 1 && !reportData.hazardType) {
      Alert.alert('알림', '위험 유형을 선택해주세요.');
      return;
    }

    if (currentStep < 4) {
      setCurrentStep(currentStep + 1);
    }
  };

  /**
   * 이전 단계로 이동
   */
  const goToPreviousStep = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
    } else {
      navigation.goBack();
    }
  };

  /**
   * 특정 단계로 이동 (미리보기에서 수정 시)
   */
  const goToStep = (step) => {
    setCurrentStep(step);
  };

  /**
   * 임시 저장
   */
  const saveDraft = async () => {
    try {
      await reportAPI.create({
        ...reportData,
        hazard_type: reportData.hazardType,
        is_draft: true,
      });
      Alert.alert('저장 완료', '임시 저장되었습니다.', [
        { text: '확인', onPress: () => navigation.goBack() },
      ]);
    } catch (error) {
      console.error('Failed to save draft:', error);
      Alert.alert('오류', '임시 저장에 실패했습니다.');
    }
  };

  /**
   * 제보 제출
   */
  const handleSubmit = async () => {
    setLoading(true);
    try {
      const submitData = {
        hazard_type: reportData.hazardType,
        description: reportData.description || '',
        latitude: reportData.latitude,
        longitude: reportData.longitude,
        accuracy: reportData.accuracy,
        severity: reportData.severity,
        reported_at: reportData.reportedAt.toISOString(),
        photos: JSON.stringify(reportData.photos),
        conditional_data: JSON.stringify(reportData.conditionalData),
        is_draft: false,
      };

      // 백엔드에 제보 등록
      await reportAPI.create(submitData);

      // 로컬 저장소에도 저장
      await myReportsStorage.add({
        ...submitData,
        status: 'pending',
      });

      setSuccessModalVisible(true);
    } catch (error) {
      console.error('Failed to submit report:', error);
      Alert.alert('오류', '제보 등록에 실패했습니다.');
    } finally {
      setLoading(false);
    }
  };

  /**
   * 성공 모달 닫기
   */
  const handleSuccessClose = () => {
    setSuccessModalVisible(false);
    navigation.goBack();
  };

  /**
   * 단계별 렌더링
   */
  const renderStep = () => {
    switch (currentStep) {
      case 1:
        return renderStep1(); // 위험 유형 선택
      case 2:
        return renderStep2(); // 위치 확인
      case 3:
        return renderStep3(); // 상세 정보
      case 4:
        return renderStep4(); // 미리보기
      default:
        return null;
    }
  };

  /**
   * Step 1: 위험 유형 선택
   */
  const renderStep1 = () => (
    <ScrollView style={styles.stepContainer} showsVerticalScrollIndicator={false}>
      <Text style={styles.stepTitle}>어떤 위험인가요?</Text>
      <Text style={styles.stepSubtitle}>
        해당하는 위험 유형을 선택해주세요
      </Text>

      <View style={styles.grid}>
        {HAZARD_TYPES.map((item) => {
          const isSelected = reportData.hazardType === item.type;
          return (
            <TouchableOpacity
              key={item.type}
              style={[
                styles.typeCard,
                isSelected && styles.typeCardSelected,
                isSelected && { borderColor: item.color },
              ]}
              onPress={() =>
                setReportData(prev => ({ ...prev, hazardType: item.type }))
              }
              activeOpacity={0.7}
            >
              <Icon
                name={item.icon}
                size={36}
                color={isSelected ? item.color : Colors.textSecondary}
              />
              <Text
                style={[
                  styles.typeLabel,
                  isSelected && { color: item.color, fontWeight: '700' },
                ]}
              >
                {item.label}
              </Text>
            </TouchableOpacity>
          );
        })}
      </View>
    </ScrollView>
  );

  /**
   * Step 2: 위치 확인 (웹용 - WebMapView 사용)
   */
  const renderStep2 = () => (
    <ScrollView style={styles.stepContainer} showsVerticalScrollIndicator={false}>
      <View style={styles.locationHeader}>
        <View>
          <Text style={styles.stepTitle}>어디서 발생했나요?</Text>
          <Text style={styles.stepSubtitle}>정확한 위치를 선택해주세요</Text>
        </View>
        <TouchableOpacity
          style={styles.currentLocationButton}
          onPress={handleUseCurrentLocation}
          activeOpacity={0.8}
        >
          <Icon name="myLocation" size={18} color={Colors.primary} />
          <Text style={styles.currentLocationText}>현재 위치</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.mapContainer}>
        <WebMapView
          mapRegion={mapRegion}
          onRegionChange={setMapRegion}
          onPress={handleMapPress}
          markers={[
            {
              latitude: reportData.latitude,
              longitude: reportData.longitude,
              title: '제보 위치',
            },
          ]}
          style={styles.map}
        />
        <View style={styles.mapOverlay}>
          <Text style={styles.mapHint}>지도를 탭하여 위치 선택</Text>
        </View>
      </View>

      <View style={styles.coordinateDisplay}>
        <Icon name="location" size={20} color={Colors.primary} />
        <View style={styles.coordinateTextContainer}>
          <Text style={styles.coordinateText}>
            {reportData.latitude.toFixed(5)}, {reportData.longitude.toFixed(5)}
          </Text>
          {reportData.accuracy && (
            <Text style={styles.accuracyText}>
              정확도: ±{Math.round(reportData.accuracy)}m
            </Text>
          )}
        </View>
      </View>
    </ScrollView>
  );

  /**
   * Step 3: 상세 정보
   */
  const renderStep3 = () => {
    const questions = CONDITIONAL_QUESTIONS[reportData.hazardType] || [];

    return (
      <ScrollView style={styles.stepContainer} showsVerticalScrollIndicator={false}>
        <Text style={styles.stepTitle}>상세 정보를 입력해주세요</Text>

        {/* 사진 첨부 */}
        <PhotoPicker
          photos={reportData.photos}
          onChange={(photos) => setReportData(prev => ({ ...prev, photos }))}
        />

        {/* 심각도 */}
        <SeverityPicker
          value={reportData.severity}
          onChange={(severity) => setReportData(prev => ({ ...prev, severity }))}
        />

        {/* 발생 시간 */}
        <TimePicker
          value={reportData.reportedAt}
          onChange={(reportedAt) => setReportData(prev => ({ ...prev, reportedAt }))}
        />

        {/* 조건부 질문 */}
        {questions.length > 0 && (
          <View style={styles.conditionalSection}>
            <Text style={styles.conditionalTitle}>추가 정보</Text>
            {questions.map((q) => renderConditionalQuestion(q))}
          </View>
        )}

        {/* 설명 */}
        <View style={styles.descriptionSection}>
          <Text style={styles.sectionLabel}>설명 (선택)</Text>
          <TextInput
            style={styles.textArea}
            value={reportData.description}
            onChangeText={(description) =>
              setReportData(prev => ({ ...prev, description }))
            }
            placeholder="위험 상황에 대한 상세한 설명을 입력해주세요..."
            multiline
            numberOfLines={4}
          />
        </View>
      </ScrollView>
    );
  };

  /**
   * 조건부 질문 렌더링
   */
  const renderConditionalQuestion = (question) => {
    const value = reportData.conditionalData[question.key];

    if (question.type === 'boolean') {
      return (
        <View key={question.key} style={styles.conditionalQuestion}>
          <Text style={styles.questionLabel}>{question.label}</Text>
          <View style={styles.booleanButtons}>
            <TouchableOpacity
              style={[
                styles.booleanButton,
                value === true && styles.booleanButtonSelected,
              ]}
              onPress={() =>
                setReportData(prev => ({
                  ...prev,
                  conditionalData: { ...prev.conditionalData, [question.key]: true },
                }))
              }
            >
              <Text
                style={[
                  styles.booleanButtonText,
                  value === true && styles.booleanButtonTextSelected,
                ]}
              >
                예
              </Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[
                styles.booleanButton,
                value === false && styles.booleanButtonSelected,
              ]}
              onPress={() =>
                setReportData(prev => ({
                  ...prev,
                  conditionalData: { ...prev.conditionalData, [question.key]: false },
                }))
              }
            >
              <Text
                style={[
                  styles.booleanButtonText,
                  value === false && styles.booleanButtonTextSelected,
                ]}
              >
                아니오
              </Text>
            </TouchableOpacity>
          </View>
        </View>
      );
    }

    if (question.type === 'number') {
      return (
        <View key={question.key} style={styles.conditionalQuestion}>
          <Text style={styles.questionLabel}>{question.label}</Text>
          <TextInput
            style={styles.numberInput}
            value={value?.toString() || ''}
            onChangeText={(text) =>
              setReportData(prev => ({
                ...prev,
                conditionalData: {
                  ...prev.conditionalData,
                  [question.key]: parseInt(text) || 0,
                },
              }))
            }
            keyboardType="numeric"
            placeholder="숫자 입력"
          />
        </View>
      );
    }

    if (question.type === 'text') {
      return (
        <View key={question.key} style={styles.conditionalQuestion}>
          <Text style={styles.questionLabel}>{question.label}</Text>
          <TextInput
            style={styles.textInput}
            value={value || ''}
            onChangeText={(text) =>
              setReportData(prev => ({
                ...prev,
                conditionalData: { ...prev.conditionalData, [question.key]: text },
              }))
            }
            placeholder="입력..."
          />
        </View>
      );
    }

    if (question.type === 'select') {
      return (
        <View key={question.key} style={styles.conditionalQuestion}>
          <Text style={styles.questionLabel}>{question.label}</Text>
          <View style={styles.selectButtons}>
            {question.options.map((option) => (
              <TouchableOpacity
                key={option}
                style={[
                  styles.selectButton,
                  value === option && styles.selectButtonSelected,
                ]}
                onPress={() =>
                  setReportData(prev => ({
                    ...prev,
                    conditionalData: { ...prev.conditionalData, [question.key]: option },
                  }))
                }
              >
                <Text
                  style={[
                    styles.selectButtonText,
                    value === option && styles.selectButtonTextSelected,
                  ]}
                >
                  {option}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>
      );
    }

    return null;
  };

  /**
   * Step 4: 미리보기
   */
  const renderStep4 = () => (
    <ReportPreview report={reportData} onEdit={goToStep} />
  );

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
    >
      {/* 단계 표시 */}
      <StepIndicator currentStep={currentStep} />

      {/* 단계별 컨텐츠 */}
      {renderStep()}

      {/* 하단 버튼 */}
      <View style={styles.footer}>
        {currentStep > 1 && (
          <TouchableOpacity
            style={styles.backButton}
            onPress={goToPreviousStep}
            activeOpacity={0.8}
          >
            <Icon name="chevron-left" size={24} color={Colors.textPrimary} />
            <Text style={styles.backButtonText}>이전</Text>
          </TouchableOpacity>
        )}

        {currentStep === 1 && (
          <TouchableOpacity
            style={styles.backButton}
            onPress={() => navigation.goBack()}
            activeOpacity={0.8}
          >
            <Icon name="close" size={24} color={Colors.textSecondary} />
            <Text style={styles.backButtonText}>취소</Text>
          </TouchableOpacity>
        )}

        {currentStep < 4 ? (
          <TouchableOpacity
            style={styles.nextButton}
            onPress={goToNextStep}
            activeOpacity={0.8}
          >
            <Text style={styles.nextButtonText}>다음</Text>
            <Icon name="chevron-right" size={24} color={Colors.textInverse} />
          </TouchableOpacity>
        ) : (
          <TouchableOpacity
            style={styles.submitButton}
            onPress={handleSubmit}
            disabled={loading}
            activeOpacity={0.8}
          >
            {loading ? (
              <ActivityIndicator color={Colors.textInverse} />
            ) : (
              <>
                <Icon name="check-box" size={24} color={Colors.textInverse} />
                <Text style={styles.submitButtonText}>제출하기</Text>
              </>
            )}
          </TouchableOpacity>
        )}
      </View>

      {/* 성공 모달 */}
      <ReportSuccessModal
        visible={successModalVisible}
        onClose={handleSuccessClose}
        impactCount={0}
      />
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  stepContainer: {
    flex: 1,
    paddingHorizontal: Spacing.lg,
    paddingTop: Spacing.md,
  },
  stepTitle: {
    ...Typography.h2,
    color: Colors.textPrimary,
    marginBottom: Spacing.xs,
  },
  stepSubtitle: {
    ...Typography.body,
    color: Colors.textSecondary,
    marginBottom: Spacing.lg,
  },
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: Spacing.md,
    paddingBottom: Spacing.xl,
  },
  typeCard: {
    width: '30%',
    aspectRatio: 1,
    backgroundColor: Colors.surface,
    borderRadius: 16,
    padding: Spacing.md,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 2,
    borderColor: Colors.border,
    gap: Spacing.xs,
  },
  typeCardSelected: {
    backgroundColor: Colors.surfaceElevated,
  },
  typeLabel: {
    ...Typography.caption,
    color: Colors.textSecondary,
    textAlign: 'center',
    fontWeight: '500',
  },
  locationHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: Spacing.md,
  },
  currentLocationButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: `${Colors.primary}10`,
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.sm,
    borderRadius: 12,
    gap: Spacing.xs,
  },
  currentLocationText: {
    ...Typography.caption,
    color: Colors.primary,
    fontWeight: '600',
  },
  mapContainer: {
    height: 300,
    borderRadius: 12,
    overflow: 'hidden',
    marginBottom: Spacing.md,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  map: {
    width: '100%',
    height: '100%',
  },
  mapOverlay: {
    position: 'absolute',
    top: Spacing.sm,
    left: Spacing.sm,
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.sm,
    borderRadius: 8,
  },
  mapHint: {
    ...Typography.caption,
    color: Colors.textSecondary,
  },
  coordinateDisplay: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: Colors.surface,
    padding: Spacing.md,
    borderRadius: 12,
    gap: Spacing.sm,
    marginBottom: Spacing.xl,
  },
  coordinateTextContainer: {
    flex: 1,
  },
  coordinateText: {
    ...Typography.body,
    color: Colors.textPrimary,
    fontWeight: '600',
  },
  accuracyText: {
    ...Typography.caption,
    color: Colors.textSecondary,
    marginTop: 2,
  },
  conditionalSection: {
    marginTop: Spacing.lg,
  },
  conditionalTitle: {
    ...Typography.h3,
    color: Colors.textPrimary,
    marginBottom: Spacing.md,
  },
  conditionalQuestion: {
    marginBottom: Spacing.lg,
  },
  questionLabel: {
    ...Typography.body,
    color: Colors.textPrimary,
    marginBottom: Spacing.sm,
    fontWeight: '600',
  },
  booleanButtons: {
    flexDirection: 'row',
    gap: Spacing.sm,
  },
  booleanButton: {
    flex: 1,
    paddingVertical: Spacing.md,
    backgroundColor: Colors.surface,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: Colors.border,
    alignItems: 'center',
  },
  booleanButtonSelected: {
    backgroundColor: Colors.primary,
    borderColor: Colors.primary,
  },
  booleanButtonText: {
    ...Typography.body,
    color: Colors.textSecondary,
    fontWeight: '600',
  },
  booleanButtonTextSelected: {
    color: Colors.textInverse,
  },
  numberInput: {
    backgroundColor: Colors.surface,
    borderRadius: 12,
    padding: Spacing.md,
    borderWidth: 1,
    borderColor: Colors.border,
    ...Typography.body,
    color: Colors.textPrimary,
  },
  textInput: {
    backgroundColor: Colors.surface,
    borderRadius: 12,
    padding: Spacing.md,
    borderWidth: 1,
    borderColor: Colors.border,
    ...Typography.body,
    color: Colors.textPrimary,
  },
  selectButtons: {
    gap: Spacing.sm,
  },
  selectButton: {
    paddingVertical: Spacing.md,
    paddingHorizontal: Spacing.lg,
    backgroundColor: Colors.surface,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: Colors.border,
    alignItems: 'center',
  },
  selectButtonSelected: {
    backgroundColor: Colors.primary,
    borderColor: Colors.primary,
  },
  selectButtonText: {
    ...Typography.body,
    color: Colors.textSecondary,
    fontWeight: '500',
  },
  selectButtonTextSelected: {
    color: Colors.textInverse,
    fontWeight: '600',
  },
  descriptionSection: {
    marginTop: Spacing.lg,
    marginBottom: Spacing.xl,
  },
  sectionLabel: {
    ...Typography.body,
    fontWeight: '600',
    color: Colors.textPrimary,
    marginBottom: Spacing.sm,
  },
  textArea: {
    backgroundColor: Colors.surface,
    borderRadius: 12,
    padding: Spacing.md,
    borderWidth: 1,
    borderColor: Colors.border,
    ...Typography.body,
    color: Colors.textPrimary,
    minHeight: 120,
    textAlignVertical: 'top',
  },
  footer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    padding: Spacing.lg,
    backgroundColor: Colors.surface,
    borderTopWidth: 1,
    borderTopColor: Colors.border,
    gap: Spacing.md,
  },
  backButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: Spacing.md,
    paddingHorizontal: Spacing.lg,
    backgroundColor: Colors.surfaceElevated,
    borderRadius: 12,
    gap: Spacing.xs,
  },
  backButtonText: {
    ...Typography.button,
    color: Colors.textPrimary,
  },
  nextButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: Spacing.md,
    backgroundColor: Colors.primary,
    borderRadius: 12,
    gap: Spacing.xs,
  },
  nextButtonText: {
    ...Typography.button,
    color: Colors.textInverse,
    fontWeight: '600',
  },
  submitButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: Spacing.md,
    backgroundColor: Colors.success,
    borderRadius: 12,
    gap: Spacing.xs,
  },
  submitButtonText: {
    ...Typography.button,
    color: Colors.textInverse,
    fontWeight: '600',
  },
});
