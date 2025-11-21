/**
 * 온보딩 네비게이터
 * 스택 기반 온보딩 플로우
 */

import React from 'react';
import { createStackNavigator } from '@react-navigation/stack';
import { Colors } from '../styles';

// 온보딩 화면들
import WelcomeScreen from '../screens/onboarding/WelcomeScreen';
import LanguageSelectScreen from '../screens/onboarding/LanguageSelectScreen';
import CountrySelectScreen from '../screens/onboarding/CountrySelectScreen';
import ProfileSetupScreen from '../screens/onboarding/ProfileSetupScreen';
import PermissionScreen from '../screens/onboarding/PermissionScreen';
import EmergencyContactScreen from '../screens/onboarding/EmergencyContactScreen';
import CompleteScreen from '../screens/onboarding/CompleteScreen';

const Stack = createStackNavigator();

export default function OnboardingNavigator() {
  return (
    <Stack.Navigator
      initialRouteName="Welcome"
      screenOptions={{
        headerShown: false,
        cardStyle: { backgroundColor: Colors.background },
        gestureEnabled: true, // 뒤로가기 제스처 활성화
      }}
    >
      <Stack.Screen
        name="Welcome"
        component={WelcomeScreen}
        options={{
          gestureEnabled: false, // 첫 화면은 뒤로가기 불가
        }}
      />
      <Stack.Screen name="LanguageSelect" component={LanguageSelectScreen} />
      <Stack.Screen name="CountrySelect" component={CountrySelectScreen} />
      <Stack.Screen name="ProfileSetup" component={ProfileSetupScreen} />
      <Stack.Screen name="Permission" component={PermissionScreen} />
      <Stack.Screen name="EmergencyContact" component={EmergencyContactScreen} />
      <Stack.Screen
        name="Complete"
        component={CompleteScreen}
        options={{
          gestureEnabled: false, // 완료 화면은 뒤로가기 불가
        }}
      />
    </Stack.Navigator>
  );
}
