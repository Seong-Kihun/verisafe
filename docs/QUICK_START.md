# VeriSafe 빠른 시작 가이드

## 🚀 실행 순서

### 0단계: Docker 환경 설정 (최초 1회만)

**Phase 1 변경사항**: PostgreSQL + PostGIS + Redis가 Docker로 실행됩니다.

**새 터미널 창 1개 열기**

```powershell
# 1. 프로젝트 루트로 이동
cd C:\Users\ki040\verisafe

# 2. Docker Compose로 PostgreSQL + PostGIS + Redis 시작
docker compose up -d

# 3. 컨테이너 상태 확인
docker compose ps
```

**정상 실행 시 출력 예시**:
```
NAME                STATUS          PORTS
verisafe-postgres   Up (healthy)    0.0.0.0:5432->5432/tcp
verisafe-redis      Up (healthy)    0.0.0.0:6379->6379/tcp
verisafe-pgadmin    Up              0.0.0.0:5050->80/tcp
```

**로그 확인**:
```powershell
# PostgreSQL 로그
docker compose logs postgres

# Redis 로그
docker compose logs redis
```

**주요 서비스 포트**:
- PostgreSQL: `localhost:5432`
- Redis: `localhost:6379`
- pgAdmin (선택): `http://localhost:5050` (Email: admin@verisafe.com, Password: admin2025)

---

### 1단계: 백엔드 환경 설정

**새 터미널 창 1개 열기**

```powershell
# 1. 백엔드 디렉토리로 이동
cd C:\Users\ki040\verisafe\backend

# 2. 가상환경 활성화
.\venv\Scripts\activate

# 3. .env 파일 생성 (최초 1회만)
# backend/.env 파일이 없으면 생성 필요
# 아래 내용을 .env 파일에 복사:
```

**.env 파일 내용** (backend/.env 파일 생성):
```env
# Application
APP_NAME=VeriSafe API
VERSION=1.0.0
DEBUG=True

# Database (PostgreSQL + PostGIS)
DATABASE_TYPE=postgresql
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_USER=verisafe_user
DATABASE_PASSWORD=verisafe_pass_2025
DATABASE_NAME=verisafe_db

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=verisafe_redis_2025
REDIS_DB=0
REDIS_CACHE_TTL=300

# JWT
SECRET_KEY=your-secret-key-change-this-in-production-2025
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# CORS
ALLOWED_ORIGINS=http://localhost:8081,http://192.168.45.177:8081

# File Storage
UPLOAD_DIR=./uploads
MAX_UPLOAD_SIZE=10485760

# External APIs (Phase 2에서 사용)
ACLED_API_KEY=
GDACS_API_URL=https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH
RELIEFWEB_API_URL=https://api.reliefweb.int/v1
```

```powershell
# 4. 패키지 설치 (최초 1회만 또는 requirements.txt 변경 시)
pip install -r requirements.txt

# 5. 서버 실행
.\venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**정상 실행 시 출력 예시**:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
[Main] 서버 시작 - 초기화 시작...
[Main] 데이터베이스 테이블 초기화...
[Database] 테이블 생성 완료
[Main] Redis 연결 초기화...
[RedisManager] 연결 성공: localhost:6379
[GraphManager] OSMnx로 주바 도로 데이터 다운로드 중...
[GraphManager] OSM 데이터 로드 성공: 1234개 노드, 2345개 엣지
[Main] GraphManager + HazardScorer 초기화 완료
```

**백엔드 서버가 `http://localhost:8000`에서 실행됩니다.**

**참고**: 
- 데이터베이스 테이블은 서버 시작 시 자동으로 생성됩니다
- Redis 캐싱이 자동으로 활성화됩니다
- 기존 SQLite 데이터가 있다면 `migrate_to_postgres.py`로 마이그레이션 가능

---

### 2단계: 모바일 앱 실행

**새 터미널 창 1개 열기**

```powershell
# 1. 모바일 디렉토리로 이동
cd C:\Users\ki040\verisafe\mobile

# 2. 의존성이 설치되어 있지 않다면 실행
npm install

# 3. Expo 서버 실행
npx expo start
```

**정상 실행 시**:
- QR 코드가 터미널에 표시됨
- 브라우저가 자동으로 열림 (http://localhost:8081)

---

### 3단계: IP 주소 업데이트

**중요**: 모바일 기기가 백엔드에 접근하려면 PC의 IP 주소를 설정해야 합니다.

**현재 PC IP**: `192.168.45.177`

**파일 수정 필요**: `mobile/src/services/api.js`

```javascript
// 6번째 줄 수정
const API_BASE_URL = __DEV__ 
  ? 'http://192.168.45.177:8000'  // PC IP 주소로 변경
  : 'https://api.verisafe.com';
```

**만약 IP가 바뀌었다면**:
```powershell
# 현재 IP 확인
ipconfig | Select-String -Pattern "IPv4"
```

---

### 4단계: 모바일 기기에서 실행

#### 옵션 A: Expo Go 사용 (권장)

1. **스마트폰에 Expo Go 앱 설치**
   - [iOS App Store](https://apps.apple.com/app/expo-go/id982107779)
   - [Google Play Store](https://play.google.com/store/apps/details?id=host.exp.exponent)

2. **QR 코드 스캔**
   - 터미널에 표시된 QR 코드를 Expo Go로 스캔
   - 또는 터미널에서 `a` 입력 (Android) / `i` 입력 (iOS)

3. **같은 WiFi 네트워크 확인**
   - PC와 스마트폰이 **같은 WiFi**에 연결되어 있어야 함

#### 옵션 B: 에뮬레이터 사용

```powershell
# Android 에뮬레이터 열기
npx expo start --android

# iOS 시뮬레이터 열기 (Mac만 가능)
npx expo start --ios
```

---

## ✅ 테스트 방법

### 백엔드 테스트

**새 터미널 창 1개 열기**

```powershell
# 1. 헬스 체크 (PostgreSQL + Redis 상태 확인)
Invoke-WebRequest -Uri http://localhost:8000/health | Select-Object -ExpandProperty Content

# 2. Redis 캐시 통계 확인
Invoke-WebRequest -Uri http://localhost:8000/api/route/cache/stats | Select-Object -ExpandProperty Content

# 3. 경로 계산 테스트 (첫 번째 요청 - 캐시 미스)
$body = @{
    start = @{lat=4.8670; lng=31.5880}
    end = @{lat=4.8500; lng=31.6000}
    preference = 'safe'
    transportation_mode = 'car'
} | ConvertTo-Json

$response1 = Invoke-WebRequest -Uri http://localhost:8000/api/route/calculate `
  -Method POST `
  -ContentType 'application/json' `
  -Body $body
$result1 = $response1.Content | ConvertFrom-Json
Write-Host "첫 번째 요청 (캐시 미스): $($result1.calculation_time_ms)ms"

# 4. 동일한 경로 계산 테스트 (두 번째 요청 - 캐시 히트)
$response2 = Invoke-WebRequest -Uri http://localhost:8000/api/route/calculate `
  -Method POST `
  -ContentType 'application/json' `
  -Body $body
$result2 = $response2.Content | ConvertFrom-Json
Write-Host "두 번째 요청 (캐시 히트): $($result2.calculation_time_ms)ms"
Write-Host "성능 개선: 약 $([math]::Round(($result1.calculation_time_ms - $result2.calculation_time_ms) / $result1.calculation_time_ms * 100, 1))% 빠름"
```

### 모바일 앱 테스트

1. **앱 실행 후 홈 화면 확인**
   - 로딩이 완료되면 "위험 정보", "랜드마크" 카운트 표시
   
2. **"경로 찾기" 버튼 클릭**
   - 안전 경로와 빠른 경로 비교 결과 표시

3. **"위험 제보하기" 버튼 클릭**
   - 제보 화면 열림
   - 위험 유형 선택 후 제보 등록

---

## 🐛 문제 해결

### 문제 1: Docker 컨테이너가 시작 안 됨

**증상**: `docker compose up -d` 실행 시 오류

**해결**:
1. Docker Desktop이 실행 중인지 확인
2. 포트 충돌 확인 (5432, 6379, 5050 포트 사용 중인지 확인)
3. 기존 컨테이너 정리:
   ```powershell
   docker compose down
   docker compose up -d
   ```

### 문제 2: PostgreSQL 연결 실패

**증상**: `[Database] 연결 실패: ...` 또는 `could not connect to server`

**해결**:
1. Docker 컨테이너 상태 확인:
   ```powershell
   docker compose ps
   ```
2. PostgreSQL 로그 확인:
   ```powershell
   docker compose logs postgres
   ```
3. 컨테이너 재시작:
   ```powershell
   docker compose restart postgres
   ```

### 문제 3: Redis 연결 실패

**증상**: `[RedisManager] 연결 실패: ...`

**해결**:
1. Redis 컨테이너 상태 확인:
   ```powershell
   docker compose ps redis
   ```
2. Redis 연결 테스트:
   ```powershell
   docker compose exec redis redis-cli -a verisafe_redis_2025 ping
   ```
   - 응답: `PONG` (정상)
3. .env 파일의 REDIS_PASSWORD 확인

### 문제 4: 백엔드 서버가 시작 안 됨

**증상**: `ModuleNotFoundError: No module named 'osmnx'` 또는 `psycopg2`

**해결**:
```powershell
cd backend
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 문제 5: 모바일 기기가 백엔드에 연결 안 됨

**증상**: `Network request failed` 또는 `Connection refused`

**해결**:
1. PC 방화벽 설정 확인
2. PC와 스마트폰이 같은 WiFi인지 확인
3. PC IP 주소가 `mobile/src/services/api.js`에 올바르게 설정되어 있는지 확인

**방화벽 설정 (Windows)**:
```powershell
New-NetFirewallRule -DisplayName "VeriSafe Backend" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
```

### 문제 6: OSM 데이터 다운로드 실패

**증상**: `[GraphManager] OSM 로드 실패: ...`

**해결**:
- 인터넷 연결 확인
- 자동으로 더미 그래프로 폴백됨 (정상 동작)

### 문제 7: Expo가 열리지 않음

**증상**: `Cannot find module 'expo'`

**해결**:
```powershell
cd mobile
rm -r node_modules
npm install
npx expo install --fix
```

---

## 🏢 회사 보안 WiFi 환경에서 개발하기

회사 보안 WiFi는 기기 간 통신을 제한할 수 있어, 일반적인 로컬 네트워크 연결이 안 될 수 있습니다. **코드 변경 없이** 사용할 수 있는 방법들:

### 방법 1: 웹 브라우저로 테스트 (가장 빠름) ⭐

**장점**: 즉시 사용 가능, 네트워크 제한 없음, 빠른 개발 사이클

```powershell
# 모바일 디렉토리에서
cd C:\Users\ki040\verisafe\mobile
npx expo start --web
```

**브라우저에서 `http://localhost:8081` 자동으로 열림**

**참고**: 
- UI/UX 테스트에 최적
- 지도 기능은 웹에서도 동작 (일부 모바일 전용 기능 제외)
- 백엔드는 `localhost:8000`에서 정상 작동

---

### 방법 2: 에뮬레이터 사용 (안드로이드 권장)

**장점**: 실제 모바일 환경과 유사, 네트워크 제한 없음

```powershell
# Android 에뮬레이터 실행 (Android Studio 필요)
npx expo start --android

# 또는 에뮬레이터가 이미 실행 중이면
npx expo start
# 터미널에서 'a' 입력
```

**참고**: 
- Android Studio 설치 필요 (한 번만)
- 에뮬레이터는 PC에서 실행되므로 `localhost` 접근 가능
- 실제 기기와 동일한 테스트 환경

---

### 방법 3: Expo Tunnel 모드 (모바일 기기 사용 시)

**장점**: 실제 기기에서 테스트 가능, 네트워크 제한 우회

```powershell
# Tunnel 모드로 실행 (인터넷 연결 필요)
npx expo start --tunnel
```

**작동 방식**:
- Expo가 자동으로 터널링 서비스 사용
- QR 코드가 생성되면 스마트폰으로 스캔
- 인터넷을 통해 접근하므로 회사 WiFi 제한 우회

**참고**: 
- 무료 버전은 느릴 수 있음
- 인터넷 연결 필요
- QR 코드 스캔 후 앱 실행

---

### 방법 4: 스마트폰 핫스팟 사용 (가장 안정적) ⭐⭐

**장점**: 실제 기기에서 테스트, 네트워크 제한 없음, 안정적

**절차**:
1. **스마트폰 핫스팟 켜기**
   - 설정 → 네트워크 → 핫스팟 활성화
   
2. **PC를 핫스팟에 연결**
   - PC의 WiFi에서 스마트폰 핫스팟 선택

3. **PC IP 주소 확인**
   ```powershell
   ipconfig | Select-String -Pattern "IPv4"
   ```
   - 예: `192.168.43.123` (핫스팟 IP)

4. **`mobile/src/services/api.js`에서 IP 주소 업데이트**
   ```javascript
   const API_BASE_URL = __DEV__ 
     ? 'http://192.168.43.123:8000'  // 핫스팟에서 받은 PC IP
     : 'https://api.verisafe.com';
   ```

5. **정상 실행**
   ```powershell
   # 백엔드 실행
   cd backend
   .\venv\Scripts\activate
   .\venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   
   # 모바일 앱 실행
   cd mobile
   npx expo start
   ```

**참고**: 
- PC와 스마트폰이 같은 네트워크 (핫스팟)
- 데이터 사용량 주의 (백엔드는 로컬이므로 대용량 사용 없음)

---

### 방법 5: USB 디버깅 (Android 전용)

**장점**: 네트워크 없이도 연결 가능, 빠름

**절차**:
1. **Android 기기 USB 디버깅 활성화**
   - 설정 → 개발자 옵션 → USB 디버깅 활성화

2. **USB로 PC에 연결**

3. **Expo 실행**
   ```powershell
   npx expo start --android
   ```
   - 자동으로 USB 연결된 기기 감지

**참고**: 
- Android 전용
- USB 케이블 필요
- ADB 드라이버 자동 설치됨

---

## 📋 환경별 권장 방법

| 환경 | 권장 방법 | 이유 |
|------|----------|------|
| **집 (일반 WiFi)** | 기존 방법 (로컬 네트워크) | 가장 빠르고 간단 |
| **회사 (보안 WiFi)** | 웹 브라우저 + 에뮬레이터 | 네트워크 제한 없음 |
| **회사 (모바일 테스트 필요)** | 핫스팟 또는 Tunnel | 실제 기기 테스트 가능 |
| **빠른 UI 확인** | 웹 브라우저 | 즉시 확인 가능 |

---

## 💡 환경 전환 팁

### 집 ↔ 회사 전환 시

1. **IP 주소 확인만 하면 됨**
   ```powershell
   # 현재 IP 확인
   ipconfig | Select-String -Pattern "IPv4"
   ```

2. **`mobile/src/services/api.js`에서 IP만 변경**
   - 집: `192.168.45.177` (예시)
   - 회사: 핫스팟 IP 또는 `localhost` (웹/에뮬레이터)

3. **코드는 그대로 두고 환경만 바꾸면 됨**

---

## 📊 서버 상태 확인

### Docker 컨테이너 상태
```powershell
# 모든 컨테이너 상태 확인
docker compose ps

# 특정 컨테이너 로그 확인
docker compose logs postgres
docker compose logs redis
```

### 백엔드 API 확인
- Health: http://localhost:8000/health (PostgreSQL + Redis 상태 확인)
- Root: http://localhost:8000/ (API 정보)
- API 문서: http://localhost:8000/docs (Swagger UI)
- 대체 문서: http://localhost:8000/redoc
- Redis 캐시 통계: http://localhost:8000/api/route/cache/stats

### 데이터베이스 관리
- pgAdmin: http://localhost:5050
  - Email: `admin@verisafe.com`
  - Password: `admin2025`
- PostgreSQL 직접 접속:
  ```powershell
  docker compose exec postgres psql -U verisafe_user -d verisafe_db
  ```

### 모바일 앱 상태
- Metro Bundler: http://localhost:8081
- Expo DevTools: 터미널에서 `m` 입력

---

## 🎯 다음 단계

프로젝트가 정상 실행되면:

1. **지도 시각화 추가**
   - 실제 경로 polyline 표시
   - 위험 마커 표시

2. **이미지 업로드 구현**
   - 제보 사진 첨부 기능

3. **관리자 승인 기능**
   - 제보 검증 UI

4. **UI/UX 개선**
   - KOICA 블루 테마 적용
   - 애니메이션 추가

---

## 💡 팁

- 백엔드와 프론트엔드는 **별도의 터미널**에서 실행해야 함
- Docker 컨테이너는 백그라운드에서 계속 실행됨 (종료하려면 `docker compose down`)
- 코드 변경 시 자동으로 재시작됨 (hot reload)
- `Ctrl+C`로 서버 종료 (Docker는 계속 실행)
- 백엔드 로그에서 `[GraphManager]`, `[HazardScorer]`, `[RedisManager]` 로그 확인 가능
- PostgreSQL 관리: pgAdmin (http://localhost:5050) 또는 `docker compose exec postgres psql -U verisafe_user -d verisafe_db`
- Redis 캐시 확인: `docker compose exec redis redis-cli -a verisafe_redis_2025`

---

**문제가 있으면 언제든지 물어보세요! 🚀**

