/**
 * VeriSafe 전체 테마 시스템
 * 디자인 시스템 통합 및 스타일 유틸리티
 */

import { Colors, getRiskColor, getHazardColor } from './colors';
import { Typography } from './typography';
import { Spacing, SpacingPatterns } from './spacing';

export const Theme = {
  colors: Colors,
  typography: Typography,
  spacing: Spacing,
  spacingPatterns: SpacingPatterns,
  
  // Helper functions
  getRiskColor,
  getHazardColor,
};

/**
 * 공통 스타일 정의
 */
export const CommonStyles = {
  // Container styles
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  
  screenContainer: {
    flex: 1,
    backgroundColor: Colors.background,
    padding: Spacing.paddingScreen,
  },
  
  // Card styles
  card: {
    backgroundColor: Colors.surface,
    borderRadius: 16,
    padding: Spacing.paddingCard,
    marginBottom: Spacing.gapBetweenCards,
    shadowColor: Colors.shadow,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 3,
  },
  
  cardElevated: {
    backgroundColor: Colors.surfaceElevated,
    borderRadius: 16,
    padding: Spacing.paddingCard,
    marginBottom: Spacing.gapBetweenCards,
    shadowColor: Colors.shadowDark,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.15,
    shadowRadius: 12,
    elevation: 5,
  },
  
  // Button styles
  button: {
    borderRadius: 24,
    paddingVertical: Spacing.md,
    paddingHorizontal: Spacing.xl,
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: Spacing.buttonHeight,
  },
  
  buttonPrimary: {
    borderRadius: 24,
    paddingVertical: Spacing.md,
    paddingHorizontal: Spacing.xl,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: Colors.primary,
    minHeight: Spacing.buttonHeight,
  },
  
  buttonSecondary: {
    borderRadius: 24,
    paddingVertical: Spacing.md,
    paddingHorizontal: Spacing.xl,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: Colors.accent,
    minHeight: Spacing.buttonHeight,
  },
  
  buttonDanger: {
    borderRadius: 24,
    paddingVertical: Spacing.md,
    paddingHorizontal: Spacing.xl,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: Colors.danger,
    minHeight: Spacing.buttonHeight,
  },
  
  buttonOutline: {
    borderRadius: 24,
    paddingVertical: Spacing.md,
    paddingHorizontal: Spacing.xl,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 2,
    borderColor: Colors.primary,
    backgroundColor: 'transparent',
    minHeight: Spacing.buttonHeight,
  },
  
  // Input styles
  input: {
    borderRadius: 16,
    borderWidth: 1,
    borderColor: Colors.border,
    padding: Spacing.md,
    fontSize: 16,
    backgroundColor: Colors.surface,
    minHeight: Spacing.inputHeight,
  },
  
  inputFocused: {
    borderColor: Colors.primary,
    borderWidth: 2,
  },
  
  inputError: {
    borderColor: Colors.danger,
    borderWidth: 2,
  },
  
  // Shadow styles - 개선된 그림자 계층
  shadowSmall: {
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.08,  // Colors.shadowSmall 사용
    shadowRadius: 4,
    elevation: 2,
  },
  
  shadowMedium: {
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.12,  // Colors.shadowMedium 사용
    shadowRadius: 8,
    elevation: 3,
  },
  
  shadowLarge: {
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.16,  // Colors.shadowLarge 사용
    shadowRadius: 12,
    elevation: 5,
  },
  
  // Border radius
  radiusSmall: { borderRadius: 8 },
  radiusMedium: { borderRadius: 12 },
  radiusLarge: { borderRadius: 16 },
  radiusXLarge: { borderRadius: 24 },
  radiusRound: { borderRadius: 999 },
  
  // Flex helpers
  row: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  
  rowBetween: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  
  rowStart: {
    flexDirection: 'row',
    alignItems: 'flex-start',
  },
  
  column: {
    flexDirection: 'column',
  },
  
  center: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  
  // Spacing helpers
  ...SpacingPatterns,
};

/**
 * 테마 기본 내보내기
 */
export default Theme;

