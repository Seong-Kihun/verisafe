/**
 * 제보탭 Stack Navigator
 * 목록 - 등록/상세
 */

import React from 'react';
import { createStackNavigator } from '@react-navigation/stack';
import { Colors, Typography } from '../styles';

// Screens
import ReportListScreen from '../screens/ReportListScreen';
import ReportScreen from '../screens/ReportScreen';

const Stack = createStackNavigator();

export default function ReportStack() {
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
        // 화면 전환 애니메이션
        transitionSpec: {
          open: {
            animation: 'timing',
            config: {
              duration: 300,
              easing: require('react-native').Easing.inOut(require('react-native').Easing.ease),
            },
          },
          close: {
            animation: 'timing',
            config: {
              duration: 300,
              easing: require('react-native').Easing.inOut(require('react-native').Easing.ease),
            },
          },
        },
        cardStyleInterpolator: ({ current, next, layouts }) => {
          return {
            cardStyle: {
              transform: [
                {
                  translateX: current.progress.interpolate({
                    inputRange: [0, 1],
                    outputRange: [layouts.screen.width, 0],
                  }),
                },
              ],
              opacity: current.progress.interpolate({
                inputRange: [0, 0.5, 0.9, 1],
                outputRange: [0, 0.25, 0.7, 1],
              }),
            },
          };
        },
      }}
    >
      <Stack.Screen 
        name="ReportList" 
        component={ReportListScreen}
        options={{ title: '위험 제보' }}
      />
      <Stack.Screen 
        name="ReportCreate" 
        component={ReportScreen}
        options={{ title: '제보 등록' }}
      />
    </Stack.Navigator>
  );
}

