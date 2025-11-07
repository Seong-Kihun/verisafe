/**
 * VeriSafe 타이포그래피 시스템
 * 가독성 최우선, 네비게이션 앱에 최적화
 */

export const Typography = {
  // Display (큰 제목)
  display: {
    fontSize: 32,
    fontWeight: '700',      // Bold 명확화
    lineHeight: 40,         // 1.25
    letterSpacing: -0.5,
  },
  
  // Headings
  h1: {
    fontSize: 24,
    fontWeight: '700',      // Bold 명확화
    lineHeight: 32,         // 1.33
    letterSpacing: -0.3,
  },
  h2: {
    fontSize: 20,
    fontWeight: '600',
    lineHeight: 28,
    letterSpacing: -0.2,
  },
  h3: {
    fontSize: 18,
    fontWeight: '600',
    lineHeight: 26,
    letterSpacing: -0.1,
  },
  
  // Body
  bodyLarge: {
    fontSize: 16,
    fontWeight: '400',
    lineHeight: 24,
    letterSpacing: 0,
  },
  body: {
    fontSize: 16,
    fontWeight: '400',
    lineHeight: 24,
    letterSpacing: 0,
  },
  bodyMedium: {
    fontSize: 15,
    fontWeight: '400',
    lineHeight: 22,
    letterSpacing: 0,
  },
  bodySmall: {
    fontSize: 14,
    fontWeight: '400',
    lineHeight: 20,
    letterSpacing: 0,
  },
  
  // Caption
  caption: {
    fontSize: 14,
    fontWeight: '400',
    lineHeight: 20,
    letterSpacing: 0.1,
  },
  captionSmall: {
    fontSize: 12,
    fontWeight: '400',
    lineHeight: 16,
    letterSpacing: 0.2,
  },
  
  // Labels
  label: {
    fontSize: 14,
    fontWeight: '600',
    lineHeight: 20,
    letterSpacing: 0.1,
  },
  labelSmall: {
    fontSize: 12,
    fontWeight: '600',
    lineHeight: 16,
    letterSpacing: 0.2,
  },
  
  // Button
  button: {
    fontSize: 16,
    fontWeight: '600',
    lineHeight: 24,
    letterSpacing: 0.2,
  },
  buttonSmall: {
    fontSize: 14,
    fontWeight: '600',
    lineHeight: 20,
    letterSpacing: 0.1,
  },
  
  // Input
  input: {
    fontSize: 16,
    fontWeight: '400',
    lineHeight: 24,
    letterSpacing: 0,
  },
  placeholder: {
    fontSize: 16,
    fontWeight: '400',
    lineHeight: 24,
    letterSpacing: 0,
    opacity: 0.5,
  },
  
  // Overline (작은 라벨)
  overline: {
    fontSize: 11,
    fontWeight: '600',
    lineHeight: 16,
    letterSpacing: 1,
    textTransform: 'uppercase',
  },
};

/**
 * 폰트 패밀리 (나중에 커스텀 폰트 추가 가능)
 */
export const FontFamily = {
  regular: 'System', // iOS: San Francisco, Android: Roboto
  medium: 'System',
  semibold: 'System',
  bold: 'System',
};

/**
 * 타이포그래피 스타일 적용 헬퍼
 */
export const applyTypography = (variant) => {
  return {
    ...Typography[variant],
    fontFamily: FontFamily.regular,
  };
};

/**
 * 행간이 적용된 Text 컴포넌트 스타일
 */
export const getTextStyle = (variant, color) => {
  return {
    ...Typography[variant],
    color,
  };
};

export default Typography;

