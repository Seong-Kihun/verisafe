/**
 * Bottom Tab Navigator
 * 4개 탭 구조: 지도, 제보, 뉴스, 내페이지
 */

import React from 'react';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { useTranslation } from 'react-i18next';
import { Colors, Typography, Spacing } from '../styles';
import Icon from '../components/icons/Icon';

// Stack Navigators
import MapStack from './MapStack';
import NewsStack from './NewsStack';
import ProfileStack from './ProfileStack';
import ReportStack from './ReportStack';

const Tab = createBottomTabNavigator();

export default function TabNavigator() {
  const { t } = useTranslation();

  return (
    <Tab.Navigator
      screenOptions={{
        headerShown: false,
        tabBarActiveTintColor: Colors.primary,
        tabBarInactiveTintColor: Colors.textSecondary,
        tabBarStyle: {
          backgroundColor: Colors.surface,
          borderTopColor: Colors.border,
          borderTopWidth: 1,
          paddingBottom: 8,
          paddingTop: 8,
          height: 60,
        },
        tabBarLabelStyle: {
          fontSize: 12,
          fontWeight: '600',
        },
      }}
    >
      <Tab.Screen
        name="MapStack"
        component={MapStack}
        options={{
          tabBarLabel: t('tabs.map'),
          tabBarIcon: ({ color, size }) => (
            <Icon name="map" size={size || 24} color={color} />
          ),
        }}
      />
      <Tab.Screen
        name="ReportStack"
        component={ReportStack}
        options={{
          tabBarLabel: t('tabs.report'),
          tabBarIcon: ({ color, size }) => (
            <Icon name="report" size={size || 24} color={color} />
          ),
        }}
      />
      <Tab.Screen
        name="NewsStack"
        component={NewsStack}
        options={{
          tabBarLabel: t('tabs.news'),
          tabBarIcon: ({ color, size }) => (
            <Icon name="article" size={size || 24} color={color} />
          ),
        }}
      />
      <Tab.Screen
        name="ProfileStack"
        component={ProfileStack}
        options={{
          tabBarLabel: t('tabs.profile'),
          tabBarIcon: ({ color, size }) => (
            <Icon name="person" size={size || 24} color={color} />
          ),
        }}
      />
    </Tab.Navigator>
  );
}

