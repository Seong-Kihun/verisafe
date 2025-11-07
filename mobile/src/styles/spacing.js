/**
 * VeriSafe 간격 시스템
 * 일관된 간격을 위한 8px 그리드 시스템
 */

export const Spacing = {
  // Base spacing (4px 단위)
  xs: 4,    // 4px
  sm: 8,    // 8px
  md: 12,   // 12px
  lg: 16,   // 16px
  xl: 24,   // 24px
  xxl: 32,  // 32px
  xxxl: 48, // 48px
  
  // Common use cases
  paddingScreen: 16,      // 화면 패딩
  paddingCard: 16,        // 카드 내부 패딩
  gapBetweenCards: 16,    // 카드 간 간격
  gapBetweenItems: 12,    // 항목 간 간격
  marginSection: 24,      // 섹션 간 마진
  
  // Specific components
  iconSize: 24,           // 아이콘 크기
  iconSmall: 16,          // 작은 아이콘
  iconLarge: 32,          // 큰 아이콘
  buttonHeight: 48,       // 버튼 높이
  buttonHeightSmall: 40,  // 작은 버튼 높이
  inputHeight: 48,        // 입력 필드 높이
  tabBarHeight: 56,       // 탭바 높이
  
  // Safe area
  safeAreaTop: 44,        // Safe area top (iOS notch)
  safeAreaBottom: 34,     // Safe area bottom (iOS home indicator)
};

/**
 * 간격을 객체로 반환 (상하좌우 개별 설정)
 */
export const spacingObj = {
  // Padding
  padding: (all) => ({ padding: all }),
  paddingHorizontal: (horizontal) => ({ paddingHorizontal: horizontal }),
  paddingVertical: (vertical) => ({ paddingVertical: vertical }),
  paddingTop: (top) => ({ paddingTop: top }),
  paddingBottom: (bottom) => ({ paddingBottom: bottom }),
  paddingLeft: (left) => ({ paddingLeft: left }),
  paddingRight: (right) => ({ paddingRight: right }),
  
  // Margin
  margin: (all) => ({ margin: all }),
  marginHorizontal: (horizontal) => ({ marginHorizontal: horizontal }),
  marginVertical: (vertical) => ({ marginVertical: vertical }),
  marginTop: (top) => ({ marginTop: top }),
  marginBottom: (bottom) => ({ marginBottom: bottom }),
  marginLeft: (left) => ({ marginLeft: left }),
  marginRight: (right) => ({ marginRight: right }),
  
  // Gap
  gap: (gap) => ({ gap }),
  rowGap: (gap) => ({ rowGap: gap }),
  columnGap: (gap) => ({ columnGap: gap }),
};

/**
 * Common spacing patterns
 */
export const SpacingPatterns = {
  // Screen padding
  screenPadding: {
    padding: Spacing.paddingScreen,
  },
  screenPaddingHorizontal: {
    paddingHorizontal: Spacing.paddingScreen,
  },
  screenPaddingVertical: {
    paddingVertical: Spacing.paddingScreen,
  },
  
  // Card spacing
  cardPadding: {
    padding: Spacing.paddingCard,
  },
  cardGap: {
    marginBottom: Spacing.gapBetweenCards,
  },
  
  // List spacing
  listItemGap: {
    marginBottom: Spacing.gapBetweenItems,
  },
  
  // Section spacing
  sectionMargin: {
    marginBottom: Spacing.marginSection,
  },
  
  // Button spacing
  buttonPadding: {
    paddingHorizontal: Spacing.lg,
    paddingVertical: Spacing.md,
  },
};

export default Spacing;

