/**
 * 온보딩 - 국가 선택 화면
 * 가장 중요! 선택한 국가에 따라 뉴스/지도 중심점 변경
 */

import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  TextInput,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Colors, Typography, Spacing } from '../../styles';
import { useOnboarding } from '../../contexts/OnboardingContext';
import { COUNTRIES } from '../../constants/countries';
import Icon from '../../components/icons/Icon';

export default function CountrySelectScreen({ navigation }) {
  const { onboardingData, updateOnboardingData } = useOnboarding();
  const [selectedCountry, setSelectedCountry] = useState(onboardingData.country);
  const [searchQuery, setSearchQuery] = useState('');

  // 검색 필터링
  const filteredCountries = COUNTRIES.filter(country =>
    country.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    country.nameEn.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleSelectCountry = (country) => {
    setSelectedCountry(country);
    updateOnboardingData('country', country);
  };

  const handleNext = () => {
    if (selectedCountry) {
      navigation.navigate('ProfileSetup');
    }
  };

  const handleSkip = () => {
    // 국가 선택 건너뛰면 남수단(기본값) 설정
    updateOnboardingData('country', COUNTRIES[0]);
    navigation.navigate('ProfileSetup');
  };

  return (
    <SafeAreaView style={styles.safeArea} edges={['top', 'bottom']}>
      <View style={styles.container}>
        {/* 뒤로가기 버튼 */}
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => navigation.goBack()}
          activeOpacity={0.7}
        >
          <Icon name="arrowBack" size={24} color={Colors.textPrimary} />
        </TouchableOpacity>

      {/* 헤더 */}
      <View style={styles.header}>
        <Text style={styles.title}>활동 국가를 선택해주세요</Text>
        <Text style={styles.subtitle}>
          선택한 국가의 안전 정보와 뉴스를 우선적으로 보여드립니다
        </Text>
      </View>

      {/* 검색바 */}
      <View style={styles.searchContainer}>
        <Icon name="search" size={20} color={Colors.textTertiary} />
        <TextInput
          style={styles.searchInput}
          placeholder="국가 검색..."
          placeholderTextColor={Colors.textTertiary}
          value={searchQuery}
          onChangeText={setSearchQuery}
        />
        {searchQuery.length > 0 && (
          <TouchableOpacity onPress={() => setSearchQuery('')}>
            <Icon name="close" size={20} color={Colors.textTertiary} />
          </TouchableOpacity>
        )}
      </View>

      {/* 국가 목록 */}
      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {filteredCountries.length > 0 ? (
          filteredCountries.map((country) => (
            <TouchableOpacity
              key={country.code}
              style={[
                styles.countryOption,
                selectedCountry?.code === country.code && styles.countryOptionSelected,
              ]}
              onPress={() => handleSelectCountry(country)}
              activeOpacity={0.7}
            >
              <View style={styles.countryLeft}>
                <Text style={styles.countryFlag}>{country.flag}</Text>
                <View style={styles.countryTextContainer}>
                  <Text style={[
                    styles.countryName,
                    selectedCountry?.code === country.code && styles.countryNameSelected,
                  ]}>
                    {country.name}
                  </Text>
                  <Text style={styles.countryCity}>
                    📍 {country.center.city}
                  </Text>
                </View>
              </View>
              {selectedCountry?.code === country.code && (
                <View style={styles.checkIcon}>
                  <Icon name="check" size={24} color={Colors.primary} />
                </View>
              )}
            </TouchableOpacity>
          ))
        ) : (
          <View style={styles.emptyContainer}>
            <Icon name="search" size={48} color={Colors.textTertiary} />
            <Text style={styles.emptyText}>검색 결과가 없습니다</Text>
          </View>
        )}
      </ScrollView>

      {/* 하단 버튼 */}
      <View style={styles.footer}>
        <TouchableOpacity
          style={styles.skipButton}
          onPress={handleSkip}
          activeOpacity={0.7}
        >
          <Text style={styles.skipButtonText}>건너뛰기</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[
            styles.nextButton,
            !selectedCountry && styles.nextButtonDisabled,
          ]}
          onPress={handleNext}
          activeOpacity={0.8}
          disabled={!selectedCountry}
        >
          <Text style={styles.nextButtonText}>다음</Text>
          <Icon name="arrowForward" size={20} color={Colors.textInverse} />
        </TouchableOpacity>
      </View>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  container: {
    flex: 1,
  },
  backButton: {
    position: 'absolute',
    top: Spacing.md,
    left: Spacing.lg,
    zIndex: 10,
    padding: Spacing.sm,
  },
  header: {
    paddingHorizontal: Spacing.xl,
    paddingTop: Spacing.xxxl + Spacing.xl,
    paddingBottom: Spacing.lg,
  },
  title: {
    ...Typography.h1,
    color: Colors.textPrimary,
    marginBottom: Spacing.sm,
  },
  subtitle: {
    ...Typography.body,
    color: Colors.textSecondary,
    lineHeight: 22,
  },
  searchContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: Colors.surface,
    borderRadius: 12,
    paddingHorizontal: Spacing.lg,
    paddingVertical: Spacing.md,
    marginHorizontal: Spacing.xl,
    marginBottom: Spacing.lg,
    gap: Spacing.sm,
  },
  searchInput: {
    flex: 1,
    ...Typography.body,
    color: Colors.textPrimary,
    padding: 0,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    paddingHorizontal: Spacing.xl,
    paddingBottom: Spacing.xl,
  },
  countryOption: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: Spacing.lg,
    borderRadius: 12,
    backgroundColor: Colors.surface,
    borderWidth: 2,
    borderColor: 'transparent',
    marginBottom: Spacing.md,
  },
  countryOptionSelected: {
    backgroundColor: Colors.primary + '10',
    borderColor: Colors.primary,
  },
  countryLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  countryFlag: {
    fontSize: 36,
    marginRight: Spacing.md,
  },
  countryTextContainer: {
    flex: 1,
  },
  countryName: {
    ...Typography.bodyLarge,
    color: Colors.textPrimary,
    marginBottom: Spacing.xs / 2,
  },
  countryNameSelected: {
    color: Colors.primary,
    fontWeight: '600',
  },
  countryCity: {
    ...Typography.caption,
    color: Colors.textSecondary,
  },
  checkIcon: {
    marginLeft: Spacing.sm,
  },
  emptyContainer: {
    alignItems: 'center',
    paddingVertical: Spacing.xxxl,
  },
  emptyText: {
    ...Typography.body,
    color: Colors.textTertiary,
    marginTop: Spacing.md,
  },
  footer: {
    flexDirection: 'row',
    paddingHorizontal: Spacing.xl,
    paddingVertical: Spacing.lg,
    gap: Spacing.md,
    borderTopWidth: 1,
    borderTopColor: Colors.borderLight,
  },
  skipButton: {
    flex: 1,
    paddingVertical: Spacing.lg,
    borderRadius: 12,
    backgroundColor: Colors.surface,
    alignItems: 'center',
    justifyContent: 'center',
  },
  skipButtonText: {
    ...Typography.button,
    color: Colors.textSecondary,
  },
  nextButton: {
    flex: 2,
    flexDirection: 'row',
    paddingVertical: Spacing.lg,
    borderRadius: 12,
    backgroundColor: Colors.primary,
    alignItems: 'center',
    justifyContent: 'center',
    gap: Spacing.sm,
    shadowColor: Colors.shadowMedium,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 4,
    elevation: 2,
  },
  nextButtonDisabled: {
    backgroundColor: Colors.border,
  },
  nextButtonText: {
    ...Typography.button,
    color: Colors.textInverse,
  },
});
