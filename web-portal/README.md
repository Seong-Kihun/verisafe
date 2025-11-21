# VeriSafe Web Portal

매퍼와 검수자를 위한 VeriSafe 웹 포털입니다.

## 기능

### 매퍼 (Mapper)
- **지도 편집**: 위성지도를 보고 건물, 도로 등 객체를 그려 정보 추가
- **내 기여**: 제출한 지리 정보와 검수 상태 확인
- Leaflet.draw를 사용한 직관적인 지도 편집 도구

### 검수자 (Reviewer / Admin)
- **검수 대기열**: 제출된 지리 정보 검토 및 승인/거부
- **검수 대시보드**: 검수 통계와 현황 모니터링
- 출처별 필터링 (AI 탐지 vs 매퍼 제출)

## 기술 스택

- **Next.js 16** - React 프레임워크 (App Router)
- **TypeScript** - 타입 안정성
- **Tailwind CSS** - 스타일링
- **Leaflet + Leaflet.draw** - 지도 및 드로잉 도구
- **React Query** - 서버 상태 관리
- **Zustand** - 클라이언트 상태 관리 (인증)
- **Axios** - HTTP 클라이언트

## 프로젝트 구조

```
web-portal/
├── app/                     # Next.js App Router
│   ├── (portal)/           # 인증 필요 페이지 (레이아웃 공유)
│   │   ├── mapper/         # 매퍼 페이지
│   │   │   ├── page.tsx              # 지도 편집기
│   │   │   └── contributions/        # 내 기여
│   │   └── reviewer/       # 검수자 페이지
│   │       ├── page.tsx              # 검수 대기열
│   │       └── dashboard/            # 대시보드
│   ├── login/              # 로그인
│   ├── layout.tsx          # 루트 레이아웃
│   ├── page.tsx            # 홈 (리다이렉트)
│   └── globals.css         # 글로벌 스타일
├── components/             # 공통 컴포넌트
│   ├── layout/
│   │   ├── Sidebar.tsx
│   │   └── Header.tsx
│   └── map/
│       └── MapEditor.tsx   # Leaflet 지도 편집기
├── lib/                    # 유틸리티 및 API
│   ├── api/
│   │   ├── client.ts       # Axios 클라이언트
│   │   ├── auth.ts         # 인증 API
│   │   ├── mapper.ts       # 매퍼 API
│   │   └── reviewer.ts     # 검수자 API
│   ├── stores/
│   │   └── auth-store.ts   # 인증 상태 (Zustand)
│   └── providers/
│       └── react-query-provider.tsx
└── types/
    └── index.ts            # TypeScript 타입 정의
```

## 시작하기

### 1. 의존성 설치

```bash
cd web-portal
npm install
```

### 2. 환경 변수 설정

`.env.local` 파일이 이미 생성되어 있습니다:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_APP_NAME=VeriSafe Mapper Portal
```

프로덕션 환경에서는 실제 API URL로 변경하세요.

### 3. 개발 서버 실행

```bash
npm run dev
```

브라우저에서 http://localhost:3000 을 열어주세요.

### 4. 로그인

백엔드에서 생성한 매퍼 또는 관리자 계정으로 로그인하세요.

- **매퍼 계정**: 지도 편집 + 내 기여 메뉴
- **관리자 계정**: 매퍼 메뉴 + 검수 메뉴

## 빌드 및 배포

### 프로덕션 빌드

```bash
npm run build
npm start
```

### 타입 체크

```bash
npm run type-check
```

### 린트

```bash
npm run lint
```

## 백엔드 통합

이 웹 포털은 기존 VeriSafe FastAPI 백엔드와 통합되어 있습니다.

### 필요한 백엔드 엔드포인트

**인증**
- `POST /api/auth/login` - 로그인
- `POST /api/auth/register` - 회원가입

**매퍼**
- `POST /api/mapper/features` - 지리 정보 생성
- `PUT /api/mapper/features/{id}` - 수정
- `DELETE /api/mapper/features/{id}` - 삭제
- `GET /api/mapper/my-contributions` - 내 기여 목록
- `GET /api/mapper/my-summary` - 통계

**검수자**
- `GET /api/review/pending` - 검수 대기열
- `POST /api/review/{id}/start-review` - 검수 시작
- `POST /api/review/{id}/approve` - 승인
- `POST /api/review/{id}/reject` - 거부
- `GET /api/review/dashboard` - 대시보드 통계
- `GET /api/review/area` - 지도 영역 조회

### CORS 설정

백엔드 `config.py`의 `allowed_origins`에 웹 포털 URL 추가:

```python
allowed_origins: str = "...,http://localhost:3000,http://127.0.0.1:3000"
```

## 주요 기능 상세

### 지도 편집기 (MapEditor)

- **위성 이미지 기본 레이어**: ArcGIS World_Imagery
- **OpenStreetMap 레이블 레이어**: 반투명 오버레이
- **드로잉 도구**:
  - Marker (마커) → Point
  - Polyline (선) → Line
  - Polygon (다각형) → Polygon
  - Rectangle (사각형) → Polygon
- **편집 기능**: 그린 객체 수정 및 삭제
- **GeoJSON 변환**: 자동으로 백엔드 호환 형식으로 변환

### 역할 기반 접근 제어

- **일반 유저 (user)**: 웹 포털 접근 불가 (모바일 앱만 사용)
- **매퍼 (mapper)**: 지도 편집, 내 기여 메뉴
- **관리자 (admin)**: 모든 메뉴 (매퍼 + 검수자)

## 모바일 앱과의 연동

모바일 앱의 프로필 탭에서 매퍼/관리자는 "매핑 포털" 메뉴를 통해 웹 포털에 접근할 수 있습니다.

위치: `mobile/src/screens/ProfileTabScreen.js`

## 트러블슈팅

### Leaflet 아이콘이 표시되지 않음

Leaflet의 기본 아이콘 경로 설정이 필요합니다. `MapEditor.tsx`에 이미 설정되어 있습니다:

```typescript
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
});
```

### CORS 에러

백엔드에서 웹 포털 URL을 CORS에 허용했는지 확인하세요.

### 인증 에러

1. 백엔드가 실행 중인지 확인
2. `.env.local`의 `NEXT_PUBLIC_API_URL` 확인
3. 브라우저 개발자 도구에서 Network 탭 확인

## 라이선스

MIT License

## 기여

VeriSafe Team
