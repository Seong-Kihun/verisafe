/**
 * App Navigator - Tab 기반
 * 모든 네비게이션은 TabNavigator 내부에서 처리
 */

import React from 'react';
import { NavigationContainer } from '@react-navigation/native';

// Tab Navigator (메인)
import TabNavigator from './TabNavigator';

export default function AppNavigator() {
  return (
    <NavigationContainer>
      <TabNavigator />
    </NavigationContainer>
  );
}
