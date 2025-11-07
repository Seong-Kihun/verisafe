/**
 * 내페이지탭 Stack Navigator
 */

import React from 'react';
import { createStackNavigator } from '@react-navigation/stack';
import { Colors, Typography } from '../styles';

import ProfileTabScreen from '../screens/ProfileTabScreen';
import ProfileEditScreen from '../screens/ProfileEditScreen';
import SavedPlacesScreen from '../screens/SavedPlacesScreen';
import RecentRoutesScreen from '../screens/RecentRoutesScreen';
import MyReportsScreen from '../screens/MyReportsScreen';
import SettingsScreen from '../screens/SettingsScreen';
import EmergencyContactsScreen from '../screens/EmergencyContactsScreen';
import EmergencyContactEditScreen from '../screens/EmergencyContactEditScreen';

const Stack = createStackNavigator();

export default function ProfileStack() {
  return (
    <Stack.Navigator
      screenOptions={{
        headerStyle: {
          backgroundColor: Colors.primary,
        },
        headerTintColor: Colors.textInverse,
        headerTitleStyle: {
          ...Typography.h3,
          fontWeight: '600',
        },
      }}
    >
      <Stack.Screen
        name="ProfileScreen"
        component={ProfileTabScreen}
        options={{ title: '내 페이지' }}
      />
      <Stack.Screen
        name="ProfileEdit"
        component={ProfileEditScreen}
        options={{ title: '프로필 편집' }}
      />
      <Stack.Screen
        name="SavedPlaces"
        component={SavedPlacesScreen}
        options={{ title: '즐겨찾기 장소' }}
      />
      <Stack.Screen
        name="RecentRoutes"
        component={RecentRoutesScreen}
        options={{ title: '최근 경로' }}
      />
      <Stack.Screen
        name="MyReports"
        component={MyReportsScreen}
        options={{ title: '나의 제보' }}
      />
      <Stack.Screen
        name="Settings"
        component={SettingsScreen}
        options={{ title: '설정' }}
      />
      <Stack.Screen
        name="EmergencyContacts"
        component={EmergencyContactsScreen}
        options={{ title: '긴급 연락망' }}
      />
      <Stack.Screen
        name="EmergencyContactEdit"
        component={EmergencyContactEditScreen}
        options={({ route }) => ({
          title: route.params?.contact ? '긴급 연락처 수정' : '긴급 연락처 추가'
        })}
      />
    </Stack.Navigator>
  );
}

