/**
 * FloatingActionButton.js - 경로 찾기 FAB 버튼
 * 우측 하단에 고정되어 경로 계획 화면으로 바로 이동
 */

import React from 'react';
import { TouchableOpacity, StyleSheet } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import Icon from './icons/Icon';
import { Colors, Spacing } from '../styles';

export default function FloatingActionButton({ style }) {
  const navigation = useNavigation();

  const handlePress = () => {
    navigation.navigate('RoutePlanning');
  };

  return (
    <TouchableOpacity
      style={[styles.fab, style]}
      onPress={handlePress}
      activeOpacity={0.8}
    >
      <Icon name="route" size={28} color={Colors.textInverse} />
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  fab: {
    position: 'absolute',
    right: Spacing.lg,
    bottom: Spacing.xl,
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: Colors.primary,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: Colors.shadowDark,
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.4,
    shadowRadius: 8,
    elevation: 6,
    zIndex: 1000,
  },
});
