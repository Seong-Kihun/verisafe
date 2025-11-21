/**
 * 온보딩 컨텍스트
 * 온보딩 상태 관리 및 데이터 수집
 */

import React, { createContext, useContext, useState, useEffect } from 'react';
import { onboardingStorage, userCountryStorage, userProfileStorage, settingsStorage } from '../services/storage';
import { setLanguage } from '../i18n';

const OnboardingContext = createContext();

export const useOnboarding = () => {
  const context = useContext(OnboardingContext);
  if (!context) {
    throw new Error('useOnboarding must be used within OnboardingProvider');
  }
  return context;
};

export const OnboardingProvider = ({ children }) => {
  const [isOnboardingCompleted, setIsOnboardingCompleted] = useState(false);
  const [loading, setLoading] = useState(true);

  // 온보딩 데이터
  const [onboardingData, setOnboardingData] = useState({
    language: 'ko',
    country: null,
    profile: {
      name: '',
      organization: '',
    },
    emergencyContacts: [],
    permissionsGranted: {
      location: false,
      notifications: false,
      camera: false,
    },
  });

  // 초기화: 온보딩 완료 여부 확인
  useEffect(() => {
    checkOnboardingStatus();
  }, []);

  const checkOnboardingStatus = async () => {
    // 개발 중: 항상 온보딩 표시
    setIsOnboardingCompleted(false);
    setLoading(false);
    return;

    // 원래 로직 (프로덕션에서 사용 시 위 3줄 제거)
    // try {
    //   const completed = await onboardingStorage.isCompleted();
    //   setIsOnboardingCompleted(completed);
    // } catch (error) {
    //   console.error('[OnboardingContext] Failed to check onboarding status:', error);
    //   // 오류 시 온보딩 표시 (안전한 기본값)
    //   setIsOnboardingCompleted(false);
    // } finally {
    //   setLoading(false);
    // }
  };

  // 온보딩 데이터 업데이트
  const updateOnboardingData = (key, value) => {
    setOnboardingData(prev => ({
      ...prev,
      [key]: value,
    }));
  };

  // 온보딩 완료 처리
  const completeOnboarding = async () => {
    try {
      // 1. 언어 설정 저장
      await setLanguage(onboardingData.language);
      await settingsStorage.save({
        language: onboardingData.language,
        notifications: onboardingData.permissionsGranted.notifications,
        autoSync: true,
      });

      // 2. 국가 설정 저장
      if (onboardingData.country) {
        await userCountryStorage.save(onboardingData.country);
      }

      // 3. 프로필 저장
      if (onboardingData.profile.name || onboardingData.profile.organization) {
        await userProfileStorage.save({
          id: 1,
          name: onboardingData.profile.name || 'User',
          email: '',
          phone: '',
          organization: onboardingData.profile.organization || '',
          role: 'user',
        });
      }

      // 4. 긴급 연락처는 EmergencyContactsScreen에서 직접 저장됨

      // 5. 온보딩 완료 표시
      await onboardingStorage.setCompleted();
      setIsOnboardingCompleted(true);

      return true;
    } catch (error) {
      console.error('Failed to complete onboarding:', error);
      return false;
    }
  };

  // 온보딩 리셋 (개발/테스트용)
  const resetOnboarding = async () => {
    try {
      await onboardingStorage.reset();
      setIsOnboardingCompleted(false);
      setOnboardingData({
        language: 'ko',
        country: null,
        profile: {
          name: '',
          organization: '',
        },
        emergencyContacts: [],
        permissionsGranted: {
          location: false,
          notifications: false,
          camera: false,
        },
      });
      return true;
    } catch (error) {
      console.error('Failed to reset onboarding:', error);
      return false;
    }
  };

  const value = {
    isOnboardingCompleted,
    loading,
    onboardingData,
    updateOnboardingData,
    completeOnboarding,
    resetOnboarding,
  };

  return (
    <OnboardingContext.Provider value={value}>
      {children}
    </OnboardingContext.Provider>
  );
};

export default OnboardingContext;
