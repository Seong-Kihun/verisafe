import React from 'react';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaProvider } from 'react-native-safe-area-context';

import AppNavigator from './src/navigation/AppNavigator';
import { MapProvider } from './src/contexts/MapContext';
import { RoutePlanningProvider } from './src/contexts/RoutePlanningContext';

// i18n 다국어 지원 초기화
import './src/i18n';

export default function App() {
  return (
    <SafeAreaProvider>
      <StatusBar style="auto" />
      <MapProvider>
        <RoutePlanningProvider>
          <AppNavigator />
        </RoutePlanningProvider>
      </MapProvider>
    </SafeAreaProvider>
  );
}
