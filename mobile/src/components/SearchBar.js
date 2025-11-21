/**
 * SearchBar - 플로팅 검색 바 컴포넌트
 * Google Maps 스타일의 지도 상단 플로팅 검색 바
 * 
 * 책임:
 * 1. 지도 상단에 항상 표시되는 검색 바
 * 2. 클릭 시 검색 모달 열기
 * 3. Step 1의 디자인 토큰 적용
 */

import React from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
} from 'react-native';
import { Colors, Typography, Spacing } from '../styles';
import Icon from './icons/Icon';

export default function SearchBar({ onPress, placeholder = '어디로 갈까요?', value }) {
  return (
    <View style={styles.container}>
      <TouchableOpacity
        style={styles.searchBar}
        onPress={onPress}
        activeOpacity={0.8}
        accessible={true}
        accessibilityRole="search"
        accessibilityLabel={value ? `선택된 장소: ${value}` : placeholder}
        accessibilityHint="두 번 탭하여 장소 검색 화면을 여세요"
      >
        <Icon name="search" size={20} color={Colors.textSecondary} />
        <View style={styles.searchContent}>
          <Text
            style={[
              styles.placeholder,
              { color: value ? Colors.textPrimary : Colors.textTertiary }
            ]}
            numberOfLines={1}
          >
            {value || placeholder}
          </Text>
        </View>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  searchBar: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    borderRadius: 12,
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.md,
    height: 56,
    // Shadow (shadowMedium 적용)
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.12,
    shadowRadius: 8,
    elevation: 4,
  },
  searchContent: {
    marginLeft: Spacing.sm,
    flex: 1,
  },
  placeholder: {
    ...Typography.input,
  },
});

