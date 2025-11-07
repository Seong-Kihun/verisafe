/**
 * 뉴스탭 Stack Navigator
 */

import React from 'react';
import { createStackNavigator } from '@react-navigation/stack';
import { Colors, Typography } from '../styles';

import NewsTabScreen from '../screens/NewsTabScreen';

const Stack = createStackNavigator();

export default function NewsStack() {
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
        name="NewsScreen" 
        component={NewsTabScreen}
        options={{ title: '뉴스' }}
      />
    </Stack.Navigator>
  );
}

