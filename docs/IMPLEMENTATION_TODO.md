# VeriSafe 개선 작업 TODO 리스트

**작성일**: 2025-11-04  
**기준 문서**: `COMPREHENSIVE_IMPROVEMENT_PLAN.md`

---

## 📋 전체 구조 파악

### 현재 파일 구조
```
mobile/src/
├── components/          # 재사용 가능한 컴포넌트
│   ├── LocationInput.js
│   ├── PlaceDetailSheet.js       ⚠️ Step 2.3, 3.3 대상
│   ├── RouteCard.js              ⚠️ Step 2.3, 3.3 대상
│   ├── RouteHazardBriefing.js    ⚠️ Step 2.3 대상
│   ├── TransportationModeSelector.js  ⚠️ Step 3.3 대상
│   └── WebMapView.js
├── contexts/           # 전역 상태 관리
│   ├── MapContext.js
│   └── RoutePlanningContext.js
├── navigation/         # 네비게이션 설정
│   ├── TabNavigator.js            ⚠️ Step 3.3, 5.3 대상
│   └── MapStack.js
├── screens/            # 화면 컴포넌트
│   ├── MapScreen.native.js        ⚠️ Step 2.2, 4.3 대상
│   ├── MapScreen.web.js           ⚠️ Step 2.2, 4.3 대상
│   ├── RoutePlanningScreen.js     ⚠️ Step 4.2 대상
│   ├── ReportScreen.js            ⚠️ Step 4.4 대상
│   └── SearchScreen.js            (모달, 유지)
└── styles/             # 디자인 시스템
    ├── colors.js                  ⚠️ Step 1.1 수정
    ├── typography.js              ⚠️ Step 1.2 수정
    ├── spacing.js                 ⚠️ Step 1.3 검토
    └── theme.js                   (통합 파일)
```

### 새로 생성할 파일 (중복 방지)
- ✅ `mobile/src/components/SearchBar.js` - 플로팅 검색 바 (새로 생성)
- ✅ `mobile/src/components/icons/Icon.js` - 통합 아이콘 컴포넌트 (새로 생성)
- ✅ `mobile/src/components/RouteComparison.js` - 경로 비교 UI (새로 생성)

### 주의사항
- ❌ `SearchScreen.js`는 그대로 유지 (모달 화면)
- ❌ `fakeSearchBar`는 `SearchBar.js`로 대체
- ❌ 이모지는 점진적으로 아이콘으로 교체 (한 번에 모두 바꾸지 않음)

---

## 🎯 Step 1: 디자인 토큰 기반 구축

**예상 시간**: 2-3시간  
**목표**: 모든 컴포넌트의 기반이 되는 디자인 토큰 확정

### 1.1 색상 시스템 재정의

**파일**: `mobile/src/styles/colors.js`

**작업 내용**:
- [ ] Primary 색상 변경: `#0066CC` → `#0047AB`
- [ ] Primary 계층 추가: `primaryLight: '#0066CC'`, `primaryDark: '#003380'`
- [ ] Background 계층 재정의:
  - `background: '#FFFFFF'` (순수 흰색)
  - `surface: '#F8F9FA'` (기존 background)
  - `surfaceElevated: '#FFFFFF'` (카드/시트)
- [ ] Text 색상 강화: `textPrimary: '#0F172A'` (더 진하게)
- [ ] Shadow 계층 추가:
  - `shadowSmall: 'rgba(0, 0, 0, 0.08)'`
  - `shadowMedium: 'rgba(0, 0, 0, 0.12)'`
  - `shadowLarge: 'rgba(0, 0, 0, 0.16)'`
- [ ] 위험도 색상 개선 (더 명확한 구분):
  - `riskVeryLow: '#10B981'` (0-2)
  - `riskLow: '#84CC16'` (3-4)
  - `riskMedium: '#F59E0B'` (5-7)
  - `riskHigh: '#EF4444'` (8-10)

**검증**:
- 기존 코드와 호환되는지 확인
- 모든 컴포넌트가 자동으로 새 색상 사용

---

### 1.2 타이포그래피 강화

**파일**: `mobile/src/styles/typography.js`

**작업 내용**:
- [ ] Display: `fontWeight: 'bold'` → `'700'` (명확화)
- [ ] H1: `fontWeight: 'bold'` → `'700'`
- [ ] H2-H3: `fontWeight: '600'` 유지 (Semibold)
- [ ] Body: `fontWeight: '400'` 유지 (Regular)
- [ ] 행간 확인: 모든 텍스트가 lineHeight 1.5 이상인지 확인
- [ ] 색상 대비: WCAG AA 기준 (4.5:1) 확인

**검증**:
- 기존 텍스트 스타일이 깨지지 않는지 확인
- 가독성 개선 확인

---

### 1.3 간격 시스템 통일

**파일**: `mobile/src/styles/spacing.js`

**작업 내용**:
- [ ] 현재 Spacing 값 검토 (이미 8px 그리드 기반인지 확인)
- [ ] 컴포넌트별 간격 가이드라인 작성:
  - 카드 내부 패딩: 16px
  - 카드 간 간격: 16px
  - 섹션 간 마진: 24px
  - 버튼 내부 패딩: 12px (vertical), 16px (horizontal)
- [ ] theme.js의 CommonStyles 확인 및 업데이트

**검증**:
- 모든 컴포넌트가 Spacing 토큰을 사용하는지 확인
- 일관성 있는 간격 적용 확인

---

### 1.4 디자인 토큰 적용 테스트

**작업 내용**:
- [ ] 앱 실행하여 색상 변경 확인
- [ ] 타이포그래피 변경 확인
- [ ] 기존 컴포넌트가 정상 작동하는지 확인
- [ ] 시각적 개선 확인

**체크포인트**: ✅ 모든 컴포넌트가 새로운 토큰을 사용할 준비 완료

---

## 🚀 Step 2: 핵심 UI 컴포넌트 개선

**예상 시간**: 4-5시간  
**목표**: 가장 많이 보이는 UI 개선

### 2.1 플로팅 검색 바 컴포넌트 생성

**파일**: `mobile/src/components/SearchBar.js` (새로 생성)

**작업 내용**:
- [ ] 컴포넌트 구조 설계:
  - 플로팅 카드 (지도 상단)
  - 클릭 시 검색 모달 확장
  - 자동완성 결과 드롭다운 (옵션)
- [ ] Step 1의 색상/타이포 적용
- [ ] 스타일:
  - 반투명 배경 (`rgba(255, 255, 255, 0.95)`)
  - 그림자 (`shadowMedium`)
  - 둥근 모서리 (12px)
- [ ] Props 설계:
  - `onPress`: 검색 모달 열기
  - `placeholder`: 검색어 입력 안내
  - `value`: 현재 검색어 (선택사항)

**주의사항**:
- SearchScreen 모달은 그대로 유지
- SearchBar는 단순히 모달을 여는 트리거 역할

---

### 2.2 MapScreen에 플로팅 검색 바 통합

**파일**: 
- `mobile/src/screens/MapScreen.native.js`
- `mobile/src/screens/MapScreen.web.js`

**작업 내용**:
- [ ] `fakeSearchBar` 제거 (현재 177-185줄, 222-229줄)
- [ ] `SearchBar` 컴포넌트 import
- [ ] `SearchBar` 컴포넌트 추가 (지도 상단, 플로팅)
- [ ] `onPress` 핸들러: `navigation.navigate('Search')`
- [ ] 위치 조정: `top: insets.top + Spacing.md`

**주의사항**:
- Category Pills 위치 조정 필요할 수 있음
- 기존 검색 기능은 그대로 유지

---

### 2.3 카드/시트 디자인 개선

**파일**:
- `mobile/src/components/PlaceDetailSheet.js`
- `mobile/src/components/RouteCard.js`
- `mobile/src/components/RouteHazardBriefing.js`

**작업 내용** (각 파일별):

#### PlaceDetailSheet.js
- [ ] 그림자 강화: `shadowMedium` → `shadowLarge`
- [ ] 여백 증가: `padding: Spacing.lg` (16px)
- [ ] 둥근 모서리: `borderRadius: 24` → 확인 (이미 24px인지 확인)
- [ ] 버튼 스타일 개선 (Step 2.4와 함께)

#### RouteCard.js
- [ ] 그림자 추가: `shadowMedium` 적용
- [ ] 여백 증가: `padding: Spacing.md` → `Spacing.lg`
- [ ] 둥근 모서리: `borderRadius: 12` → `16`
- [ ] 선택 상태 스타일 개선

#### RouteHazardBriefing.js
- [ ] 그림자 강화: `shadowLarge` 적용
- [ ] 여백 확인 및 조정
- [ ] 둥근 모서리: `borderTopLeftRadius: 24` 유지

**검증**:
- 모든 카드가 일관된 스타일 사용
- 깊이감이 명확하게 보임

---

### 2.4 버튼 스타일 개선

**파일**:
- `mobile/src/components/PlaceDetailSheet.js`
- `mobile/src/components/RouteCard.js`
- 기타 버튼 사용 컴포넌트

**작업 내용**:
- [ ] Primary 버튼 스타일 명확화:
  - 배경색: `Colors.primary`
  - 텍스트 색상: `Colors.textInverse`
  - 높이: `48px` (Spacing.buttonHeight)
- [ ] Secondary 버튼 스타일:
  - 배경색: `Colors.borderLight`
  - 텍스트 색상: `Colors.textPrimary`
  - 높이: `48px`
- [ ] 터치 피드백 추가:
  - `activeOpacity={0.7}` (기존)
  - Scale 애니메이션 (Step 5.1에서 추가)

**대상 컴포넌트**:
1. PlaceDetailSheet.js - 4개 버튼 (경로, 저장, 제보, 공유)
2. RouteCard.js - 선택 버튼
3. TransportationModeSelector.js - 이동 수단 버튼
4. 기타 주요 버튼

**검증**:
- 버튼 스타일 일관성
- 터치 피드백 작동

---

## 🎨 Step 3: 아이콘 시스템 도입

**예상 시간**: 3-4시간  
**목표**: 이모지 → 아이콘 교체

### 3.1 아이콘 라이브러리 설치

**작업 내용**:
- [ ] `cd mobile`
- [ ] `npm install @expo/vector-icons`
- [ ] 설치 확인

---

### 3.2 통합 아이콘 컴포넌트 생성

**파일**: `mobile/src/components/icons/Icon.js` (새로 생성)

**작업 내용**:
- [ ] 디렉토리 생성: `mobile/src/components/icons/`
- [ ] Icon 컴포넌트 생성:
  ```javascript
  import { MaterialIcons } from '@expo/vector-icons';
  
  export default function Icon({ name, size, color, ...props }) {
    return <MaterialIcons name={name} size={size} color={color} {...props} />;
  }
  ```
- [ ] 아이콘 매핑 상수 정의:
  - Navigation: route, location-on, search, map
  - Hazard: warning, security, groups, dangerous
  - Transportation: directions-car, directions-walk, directions-bike
  - Action: bookmark, share, report, close
- [ ] 헬퍼 함수: `getIconName(type)` (이모지 → 아이콘 이름)

**주의사항**:
- `@expo/vector-icons`는 Expo에 기본 포함되어 있을 수 있음
- 없으면 설치 필요

---

### 3.3 이모지 → 아이콘 교체

**우선순위별 작업**:

#### 3.3.1 PlaceDetailSheet.js
- [ ] 버튼 아이콘 교체:
  - 🗺️ → `MaterialIcons.route` 또는 `directions`
  - ⭐ → `MaterialIcons.bookmark`
  - ⚠️ → `MaterialIcons.warning`
  - 📤 → `MaterialIcons.share`
- [ ] 카테고리 아이콘 교체 (CATEGORY_ICONS)
- [ ] Icon 컴포넌트 import 및 사용

#### 3.3.2 RouteCard.js
- [ ] 경로 타입 아이콘 교체:
  - 🛡️ → `MaterialIcons.shield` 또는 `security`
  - ⚡ → `MaterialIcons.flash-on` 또는 `bolt`
  - 📍 → `MaterialIcons.place`
- [ ] 이동 수단 아이콘 교체 (이미 TransportationModeSelector에서 처리)
- [ ] 상세 정보 아이콘:
  - ⏱️ → `MaterialIcons.access-time`
  - 📍 → `MaterialIcons.place`
  - ⚠️ → `MaterialIcons.warning`

#### 3.3.3 TransportationModeSelector.js
- [ ] 이동 수단 아이콘 교체:
  - 🚗 → `MaterialIcons.directions-car`
  - 🚶 → `MaterialIcons.directions-walk`
  - 🚴 → `MaterialIcons.directions-bike`

#### 3.3.4 TabNavigator.js
- [ ] 탭 아이콘 추가:
  - 지도: `MaterialIcons.map`
  - 제보: `MaterialIcons.report`
  - 뉴스: `MaterialIcons.newspaper` 또는 `article`
  - 내페이지: `MaterialIcons.person`
- [ ] `tabBarIcon` 옵션에 Icon 컴포넌트 사용

#### 3.3.5 MapScreen.*.js
- [ ] 내 위치 버튼: 📍 → `MaterialIcons.my-location`
- [ ] Go 버튼: 🚗 → `MaterialIcons.directions-car` 또는 `navigation`
- [ ] 카테고리 필터 아이콘 (CATEGORIES)

**검증**:
- 모든 아이콘이 제대로 표시되는지
- 크기와 색상이 일관된지
- 이모지가 남아있지 않은지

---

## 🗺️ Step 4: 핵심 기능 UX 개선

**예상 시간**: 6-8시간  
**목표**: 기능적 가치 향상

### 4.1 경로 비교 UI 컴포넌트 생성

**파일**: `mobile/src/components/RouteComparison.js` (새로 생성)

**작업 내용**:
- [ ] 컴포넌트 구조 설계:
  - 탭으로 Safe/Fast/Alternative 전환
  - 각 경로 정보 카드
  - 위험도/시간/거리 바 그래프
- [ ] Props:
  - `routes`: 경로 배열
  - `selectedRoute`: 선택된 경로
  - `onSelect`: 경로 선택 핸들러
- [ ] 바 그래프 시각화:
  - 위험도: 0-10 스케일
  - 시간: 상대적 비교
  - 거리: 상대적 비교
- [ ] Step 1-3의 디자인 토큰 적용

**주의사항**:
- RouteCard.js와 중복되지 않도록
- RouteComparison은 여러 경로를 한 화면에서 비교
- RouteCard는 단일 경로 카드

---

### 4.2 RoutePlanningScreen에 경로 비교 UI 통합

**파일**: `mobile/src/screens/RoutePlanningScreen.js`

**작업 내용**:
- [ ] RouteComparison 컴포넌트 import
- [ ] 기존 RouteCard 리스트를 RouteComparison으로 교체 (옵션)
  - 또는 두 가지 모두 제공 (토글 가능)
- [ ] 경로 선택 시 지도에 표시 (이미 구현됨)
- [ ] Step 1-3의 디자인 토큰 적용

**주의사항**:
- 기존 기능 유지
- 사용자가 선택할 수 있도록 옵션 제공 고려

---

### 4.3 지도 인터랙션 강화

**파일**:
- `mobile/src/screens/MapScreen.native.js`
- `mobile/src/screens/MapScreen.web.js`

**작업 내용**:

#### 더블 탭 줌
- [ ] `onDoublePress` 핸들러 추가 (react-native-maps)
- [ ] WebMapView에도 동일 기능 추가 (react-leaflet)
- [ ] 줌 레벨 계산 (현재 줌 + 1)

#### 롱 프레스 장소 선택
- [ ] `onLongPress` 핸들러 추가
- [ ] 좌표에서 장소 정보 조회 (기존 API 활용)
- [ ] PlaceDetailSheet 열기 또는 제보 화면으로 이동 옵션

#### 경로 선택 시 자동 포커스
- [ ] 이미 구현됨 (useEffect로 확인)
- [ ] 애니메이션 개선 (Step 5.1)

**주의사항**:
- Web과 Native 모두 동일한 UX 제공
- 제스처 충돌 방지

---

### 4.4 제보 플로우 개선

**파일**: `mobile/src/screens/ReportScreen.js`

**작업 내용**:
- [ ] 지도 컴포넌트 통합:
  - Web: WebMapView 사용
  - Native: MapView 사용 (작은 미니맵)
- [ ] 지도에서 위치 선택:
  - 탭/클릭으로 위치 선택
  - 마커로 선택 위치 표시
- [ ] 현재 위치 자동 감지:
  - `expo-location` 활용
  - "현재 위치 사용" 버튼 추가
- [ ] 주소 자동완성:
  - 기존 검색 API 활용
  - 위치 선택 시 주소 자동 입력

**주의사항**:
- 지도 컴포넌트는 작은 크기로 (전체 화면 아님)
- 텍스트 입력도 여전히 가능하도록

---

## ✨ Step 5: 마무리 및 세부 개선

**예상 시간**: 4-5시간  
**목표**: 완성도 향상

### 5.1 애니메이션 추가

**작업 내용**:

#### 화면 전환 애니메이션
- [ ] React Navigation transition 설정
- [ ] `MapStack.js`, `ReportStack.js` 등에 애니메이션 추가
- [ ] 300ms ease-in-out

#### 카드/시트 등장 애니메이션
- [ ] PlaceDetailSheet: Fade in + Slide up
- [ ] RouteHazardBriefing: Fade in + Slide up
- [ ] RouteCard: Fade in (리스트에 추가될 때)

#### 버튼 터치 피드백
- [ ] Scale 애니메이션 (0.95 → 1.0)
- [ ] `Animated` API 또는 `react-native-reanimated` 사용
- [ ] 100ms duration

#### 로딩 스켈레톤 UI
- [ ] 경로 계산 중 스켈레톤 UI
- [ ] 검색 결과 로딩 중 스켈레톤

**대상 컴포넌트**:
- Step 2에서 개선한 모든 컴포넌트
- Step 4에서 개선한 기능

---

### 5.2 로딩/에러 상태 개선

**작업 내용**:
- [ ] 스켈레톤 UI 컴포넌트 생성 (선택사항)
- [ ] 에러 메시지 개선:
  - 친화적인 문구
  - 재시도 버튼 제공
- [ ] 빈 상태 (Empty State) 디자인:
  - 검색 결과 없음
  - 경로 없음
  - 제보 없음
- [ ] 성공 피드백:
  - Toast 메시지 (선택사항)
  - 또는 기존 Alert 개선

**대상 화면**:
- SearchScreen
- RoutePlanningScreen
- ReportListScreen

---

### 5.3 네비게이션 최종 개선

**파일**: `mobile/src/navigation/TabNavigator.js`

**작업 내용**:
- [ ] 탭 아이콘 추가 (Step 3.3.4 완료 후)
- [ ] 현재 탭 강조:
  - `tabBarActiveTintColor` 확인
  - 아이콘 크기 조정 (선택 시)
- [ ] 알림 배지 (선택사항):
  - 제보 대기 개수
  - 새로운 뉴스

**검증**:
- 모든 탭에 아이콘 표시
- 현재 탭이 명확하게 구분됨

---

### 5.4 전체 일관성 검토

**작업 내용**:
- [ ] 시각적 일관성 검토:
  - 모든 컴포넌트가 동일한 디자인 토큰 사용
  - 색상, 간격, 타이포 일관성
- [ ] 기능 정상 작동 확인:
  - 각 화면별 기능 테스트
  - 경로 계산, 제보, 검색 등
- [ ] 성능 확인:
  - 애니메이션 성능
  - 렌더링 성능
  - 메모리 사용량
- [ ] 접근성 확인:
  - 터치 영역 크기 (최소 44x44px)
  - 색상 대비 (WCAG AA)
  - 폰트 크기

**체크리스트**:
- [ ] Step 1: 디자인 토큰 적용 완료
- [ ] Step 2: 핵심 UI 컴포넌트 개선 완료
- [ ] Step 3: 아이콘 시스템 도입 완료
- [ ] Step 4: 핵심 기능 UX 개선 완료
- [ ] Step 5: 마무리 작업 완료

---

## 📝 진행 상황 추적

### 완료 기준
각 Step은 다음이 완료되면 완료로 간주:
1. ✅ 기능 테스트 통과
2. ✅ 시각적 검토 완료
3. ✅ 코드 리뷰 완료
4. ✅ 다음 Step 준비 완료

### 주의사항
- **중복 파일 생성 방지**: 새 파일 생성 전 기존 파일 확인
- **기능 유지**: 디자인 개선 시 기존 기능은 그대로 유지
- **점진적 적용**: 한 번에 모든 것을 바꾸지 않고 단계적으로
- **테스트**: 각 Step 완료 후 전체 앱 테스트

---

## 🔄 파일 변경 추적

### 새로 생성할 파일
1. `mobile/src/components/SearchBar.js` - Step 2.1
2. `mobile/src/components/icons/Icon.js` - Step 3.2
3. `mobile/src/components/RouteComparison.js` - Step 4.1

### 수정할 파일
1. `mobile/src/styles/colors.js` - Step 1.1
2. `mobile/src/styles/typography.js` - Step 1.2
3. `mobile/src/styles/spacing.js` - Step 1.3 (검토)
4. `mobile/src/screens/MapScreen.native.js` - Step 2.2, 4.3
5. `mobile/src/screens/MapScreen.web.js` - Step 2.2, 4.3
6. `mobile/src/components/PlaceDetailSheet.js` - Step 2.3, 2.4, 3.3
7. `mobile/src/components/RouteCard.js` - Step 2.3, 2.4, 3.3
8. `mobile/src/components/RouteHazardBriefing.js` - Step 2.3
9. `mobile/src/components/TransportationModeSelector.js` - Step 2.4, 3.3
10. `mobile/src/navigation/TabNavigator.js` - Step 3.3, 5.3
11. `mobile/src/screens/RoutePlanningScreen.js` - Step 4.2
12. `mobile/src/screens/ReportScreen.js` - Step 4.4

### 삭제할 요소
- `MapScreen.*.js`의 `fakeSearchBar` (SearchBar 컴포넌트로 대체)

---

**다음 단계**: Step 1부터 순차적으로 진행 시작

