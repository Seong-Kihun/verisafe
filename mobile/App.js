import React from 'react';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaProvider } from 'react-native-safe-area-context';

import AppNavigator from './src/navigation/AppNavigator';
import { MapProvider } from './src/contexts/MapContext';
import { RoutePlanningProvider } from './src/contexts/RoutePlanningContext';
import { HazardFilterProvider } from './src/contexts/HazardFilterContext';
import { OnboardingProvider } from './src/contexts/OnboardingContext';
import { NavigationProvider } from './src/contexts/NavigationContext';

// i18n 다국어 지원 초기화
import './src/i18n';

export default function App() {
  return (
    <SafeAreaProvider>
      <StatusBar style="auto" />
      <OnboardingProvider>
        <HazardFilterProvider>
          <MapProvider>
            <RoutePlanningProvider>
              <NavigationProvider>
                <AppNavigator />
              </NavigationProvider>
            </RoutePlanningProvider>
          </MapProvider>
        </HazardFilterProvider>
      </OnboardingProvider>
    </SafeAreaProvider>
  );
}
