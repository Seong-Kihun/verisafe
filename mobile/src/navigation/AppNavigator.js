/**
 * App Navigator - 온보딩/메인 분기
 * 온보딩 완료 여부에 따라 화면 전환
 */

import React from 'react';
import { View, ActivityIndicator } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { useOnboarding } from '../contexts/OnboardingContext';
import { Colors } from '../styles';

// Navigators
import OnboardingNavigator from './OnboardingNavigator';
import TabNavigator from './TabNavigator';

export default function AppNavigator() {
  const { isOnboardingCompleted, loading } = useOnboarding();

  // 로딩 중
  if (loading) {
    return (
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: Colors.background }}>
        <ActivityIndicator size="large" color={Colors.primary} />
      </View>
    );
  }

  return (
    <NavigationContainer>
      {isOnboardingCompleted ? <TabNavigator /> : <OnboardingNavigator />}
    </NavigationContainer>
  );
}
