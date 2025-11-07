# VeriSafe 간격 시스템 가이드라인

**작성일**: 2025-11-04  
**기준**: 8px 그리드 시스템

---

## 📏 기본 원칙

모든 간격은 **8px의 배수**를 사용합니다:
- 4px (xs) - 매우 작은 간격
- 8px (sm) - 작은 간격
- 12px (md) - 중간 간격
- 16px (lg) - 큰 간격
- 24px (xl) - 매우 큰 간격
- 32px (xxl) - 섹션 간격
- 48px (xxxl) - 화면 간격

---

## 🎯 컴포넌트별 간격 가이드라인

### 카드 (Card)
- **내부 패딩**: `Spacing.lg` (16px)
- **카드 간 간격**: `Spacing.lg` (16px)
- **둥근 모서리**: `borderRadius: 16` (16px)

### 버튼 (Button)
- **내부 패딩**: 
  - Vertical: `Spacing.md` (12px)
  - Horizontal: `Spacing.lg` (16px)
- **버튼 높이**: `Spacing.buttonHeight` (48px)
- **버튼 간 간격**: `Spacing.sm` (8px)

### 입력 필드 (Input)
- **내부 패딩**: `Spacing.md` (12px)
- **입력 필드 높이**: `Spacing.inputHeight` (48px)
- **라벨과 입력 필드 간격**: `Spacing.sm` (8px)

### 섹션 (Section)
- **섹션 간 마진**: `Spacing.xl` (24px)
- **섹션 내부 패딩**: `Spacing.lg` (16px)

### 리스트 (List)
- **항목 간 간격**: `Spacing.md` (12px)
- **리스트 내부 패딩**: `Spacing.lg` (16px)

### 화면 (Screen)
- **화면 패딩**: `Spacing.lg` (16px)
- **상단 여백**: Safe area + `Spacing.md` (12px)

---

## ✅ 확인 사항

각 컴포넌트 수정 시:
1. Spacing 토큰을 사용하는가?
2. 8px 그리드에 맞는가?
3. 일관된 간격을 사용하는가?

---

**참고**: `mobile/src/styles/spacing.js`에 모든 값이 정의되어 있습니다.

