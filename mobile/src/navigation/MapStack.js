/**
 * 지도탭 Stack Navigator
 * MapScreen + SearchScreen (modal)
 */

import React from 'react';
import { createStackNavigator } from '@react-navigation/stack';
import { Colors, Typography } from '../styles';

import MapScreen from '../screens/MapScreen';
import SearchScreen from '../screens/SearchScreen';
import RoutePlanningScreen from '../screens/RoutePlanningScreen';
import NavigationScreen from '../screens/NavigationScreen';

const Stack = createStackNavigator();

export default function MapStack() {
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
        // 화면 전환 애니메이션 (300ms ease-in-out)
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
        name="MapMain"
        component={MapScreen}
        options={{
          title: 'VeriSafe',
          headerShown: true,
        }}
      />
      <Stack.Screen 
        name="Search" 
        component={SearchScreen}
        options={{ 
          headerShown: false,
          presentation: 'modal',
          cardStyleInterpolator: ({ current, layouts }) => ({
            cardStyle: {
              transform: [
                {
                  translateY: current.progress.interpolate({
                    inputRange: [0, 1],
                    outputRange: [layouts.screen.height, 0],
                  }),
                },
              ],
            },
          }),
        }}
      />
      <Stack.Screen
        name="RoutePlanning"
        component={RoutePlanningScreen}
        options={{
          title: '경로 찾기',
        }}
      />
      <Stack.Screen
        name="NavigationScreen"
        component={NavigationScreen}
        options={{
          headerShown: false,
          presentation: 'fullScreenModal',
        }}
      />
    </Stack.Navigator>
  );
}

