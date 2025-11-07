# 📡 외부 데이터 수집 시스템 사용 가이드

## ✅ Phase 1 & 2 완료 항목

### 1. 자동 데이터 수집 시스템

**서버 시작 시 자동 실행됩니다!**

```bash
# 서버 시작
cd backend
call venv\Scripts\activate.bat
uvicorn app.main:app --reload
```

**자동으로 실행되는 작업:**
1. 서버 시작 시 즉시 외부 데이터 수집 (1회)
2. 24시간마다 자동 재수집
3. GraphManager 자동 업데이트

### 2. 데이터 소스

현재 6개 외부 API에서 데이터 수집:

| 소스 | 설명 | 수집 주기 | API 키 필요 | Phase |
|------|------|-----------|-------------|-------|
| **ACLED** | 분쟁/폭력 사건 | 7일간 | 선택사항 | 1 |
| **GDACS** | 자연재해 | 30일간 | 불필요 | 1 |
| **ReliefWeb** | 인도적 보고서 | 7일간 | 불필요 | 1 |
| **Twitter/X** | 소셜 미디어 모니터링 | 24시간 | 필수 | 2 |
| **NewsAPI** | 뉴스 기사 분석 | 3일간 | 필수 | 2 |
| **Sentinel Hub** | 위성 이미지 분석 | 7일간 | 필수 | 2 |

---

## 🎯 대시보드 접근 방법

### 방법 1: 웹 브라우저에서 직접 API 호출

```bash
# 통계 조회
GET http://localhost:8000/api/data/dashboard/stats

# 수동 데이터 수집
POST http://localhost:8000/api/data/dashboard/trigger-collection

# 최근 수집 데이터
GET http://localhost:8000/api/data/dashboard/recent-hazards?limit=20
```

### 방법 2: 모바일 앱 대시보드

**파일 추가됨:** `mobile/src/screens/DataDashboardScreen.js`

**네비게이션 추가 방법:**

`mobile/src/navigation/AppNavigator.js`에 추가:

```javascript
import DataDashboardScreen from '../screens/DataDashboardScreen';

// 스택에 추가
<Stack.Screen
  name="DataDashboard"
  component={DataDashboardScreen}
  options={{ title: '데이터 수집 현황' }}
/>
```

**탭 네비게이터에 추가** (권장):

```javascript
// 프로필 탭 또는 설정 탭에 버튼 추가
<TouchableOpacity
  onPress={() => navigation.navigate('DataDashboard')}
>
  <Text>데이터 수집 현황</Text>
</TouchableOpacity>
```

---

## 🔧 API 키 설정

### Phase 1 APIs (선택사항)

#### ACLED API 키 (무료)

API 키가 없으면 더미 데이터가 생성됩니다.

**발급 방법:**
1. https://acleddata.com/ 접속
2. 회원가입
3. API Access 신청
4. 키 발급 (무료)

**설정:**

`backend/.env` 파일에 추가:

```env
ACLED_API_KEY=your_api_key_here
```

### Phase 2 APIs (선택사항)

#### Twitter/X API Bearer Token

**발급 방법:**
1. https://developer.twitter.com/ 접속
2. Developer Portal 가입
3. App 생성
4. Bearer Token 발급

**설정:**
```env
TWITTER_BEARER_TOKEN=your_bearer_token_here
```

**주의:** Twitter API는 무료 티어 제한이 있습니다 (월 500,000 트윗 조회).

#### NewsAPI 키 (무료)

**발급 방법:**
1. https://newsapi.org/ 접속
2. Get API Key 클릭
3. 무료 계정 생성
4. API 키 복사

**설정:**
```env
NEWS_API_KEY=your_news_api_key_here
```

**주의:** 무료 계정은 하루 100개 요청으로 제한됩니다.

#### Sentinel Hub API

**발급 방법:**
1. https://www.sentinel-hub.com/ 접속
2. 무료 Trial 계정 생성
3. OAuth Client 생성
4. Client ID와 Secret 복사

**설정:**
```env
SENTINEL_CLIENT_ID=your_client_id_here
SENTINEL_CLIENT_SECRET=your_client_secret_here
```

**주의:** Trial 계정은 월 1,000 Processing Units (PU) 제한이 있습니다.

### API 키 없이 사용하기

**Phase 2 API 키가 없어도 시스템은 정상 작동합니다!**

- API 키가 없으면 각 collector가 자동으로 더미 데이터를 생성합니다.
- 더미 데이터로 시스템 테스트 및 개발이 가능합니다.
- 실제 프로덕션 환경에서는 API 키 설정을 권장합니다.

---

## 📊 수집된 데이터 확인

### 1. 데이터베이스에서 직접 확인

```sql
-- PostgreSQL
SELECT
  source,
  hazard_type,
  COUNT(*) as count,
  MAX(created_at) as last_updated
FROM hazards
GROUP BY source, hazard_type
ORDER BY source;
```

### 2. API로 확인

```bash
# 최근 수집 데이터
curl http://localhost:8000/api/data/dashboard/recent-hazards?limit=10
```

### 3. 지도에서 확인

앱 실행 → 지도 화면 → 위험 정보 마커 확인

---

## 🚨 트러블슈팅

### 문제 1: 데이터가 수집되지 않음

**원인:** 백엔드가 시작되지 않았거나 초기화 오류

**해결:**
```bash
# 백엔드 로그 확인
cd backend
tail -f logs/app.log

# 다음 메시지 확인:
# [Main] 외부 데이터 스케줄러 초기화 중...
# [Main] 초기 외부 데이터 수집 시작...
# [DataCollectorScheduler] 데이터 수집 시작
```

### 문제 2: API 호출 오류

**원인:** CORS 또는 네트워크 오류

**해결:**
```python
# backend/app/config.py 확인
allowed_origins: str = "http://localhost:8081,http://192.168.45.177:8081"
```

### 문제 3: 더미 데이터만 생성됨

**정상입니다!** API 키가 없으면 더미 데이터로 작동합니다.

**실제 데이터를 원하면:**
- ACLED API 키 발급 (위 참조)
- GDACS, ReliefWeb은 키 불필요 (자동 작동)

---

## 📈 데이터 수집 로직

### ACLED (분쟁 데이터)

```python
# 이벤트 타입별 위험도
Battles: 80-100점
Violence against civilians: 75-95점
Riots: 50-70점
Protests: 30-60점

# 위험도 계산
base_risk = 80  # Battles
fatalities = 5
risk_score = min(100, base_risk + (fatalities * 5))  # 105 → 100
```

### GDACS (자연재해)

```python
# Alert Level 기반 위험도
Red: 90점
Orange: 70점
Green: 40점

# 재난 타입별 지속 기간
지진: 24시간
홍수: 168시간 (7일)
사이클론: 72시간 (3일)
```

### ReliefWeb (인도적 보고서)

```python
# 키워드 기반 위험 감지
"conflict", "violence" → conflict (60점)
"flood", "flooding" → flood (50점)
"disease", "epidemic" → other (40점)
```

---

## ✅ Phase 2 완료 항목

### 1. Twitter/X API 통합

**실시간 소셜 미디어 모니터링**

- 위험 관련 키워드 자동 감지 (conflict, protest, flood 등)
- 최근 24시간 트윗 수집
- 리트윗/좋아요 수로 신뢰도 계산
- 지리적 위치 정보 추출

**위험 유형 감지:**
- Conflict (분쟁)
- Protest (시위)
- Natural Disaster (자연재해)
- Checkpoint (검문소)
- Emergency (긴급 상황)

### 2. NewsAPI 통합

**뉴스 기사 기반 위험 정보**

- 남수단 관련 최근 3일 뉴스 수집
- 키워드 분석으로 위험 유형 자동 감지
- 신뢰할 수 있는 언론사 우선 (BBC, Reuters, AP 등)
- 기사 제목 + 내용 분석

**신뢰도 향상:**
- 신뢰할 수 있는 언론사 +10점 가산
- 다수 키워드 매칭 시 위험도 증가

### 3. Sentinel Hub 위성 이미지 분석

**환경 재해 자동 감지**

- Sentinel-2 위성 데이터 활용
- 주요 도시 4곳 모니터링 (Juba, Malakal, Wau, Bentiu)

**감지 기능:**
- **NDWI (홍수 감지)**: 물의 존재 여부 분석
- **NBR (화재 감지)**: 화재 발생 및 확산 감지
- **NDVI (가뭄 감지)**: 식생 변화로 가뭄 예측

**장점:**
- 객관적 데이터 (위성 기반)
- 넓은 지역 커버
- 신뢰도 높음 (verified=True)

## 🎯 다음 단계 (Phase 3)

Phase 3 구현을 원하시면 `EXTERNAL_DATA_UPGRADE_PLAN.md` 참조:

1. **NLP 분석기** - BERT/Transformers 기반 텍스트 분석
2. **LSTM 예측 모델** - 시계열 데이터로 위험 예측
3. **크라우드소싱 시스템** - 사용자 제보 신뢰도 평가

---

## 💡 팁

### 데이터 수집 주기 변경

`backend/app/main.py`:

```python
# 24시간 → 6시간으로 변경
asyncio.create_task(data_scheduler.start_scheduler(interval_hours=6))
```

### 특정 소스만 수집

`backend/app/services/external_data/data_collector_scheduler.py`:

```python
async def collect_all_data(self, db: Session):
    # ACLED만 수집
    acled_count = await self.acled.collect_recent_events(db)

    # GDACS, ReliefWeb 주석 처리
    # gdacs_count = await self.gdacs.collect_recent_disasters(db)
    # reliefweb_count = await self.reliefweb.collect_recent_reports(db)
```

### 수집 기간 변경

```python
# 7일 → 14일
await self.acled.collect_recent_events(db, days=14)
```

---

## 📞 지원

문제가 발생하면:

1. 백엔드 로그 확인
2. API 테스트: http://localhost:8000/docs
3. 데이터베이스 직접 조회
4. GitHub Issues 등록

---

## ✅ Phase 3 완료 항목

### 1. NLP 텍스트 분석기

**자연어 처리 기반 텍스트 분석**

- Transformers/BERT 기반 감정 분석
- 위험도 자동 평가 (0-100점)
- 키워드 추출 및 위험 유형 추론
- 긴급성 평가 (immediate, recent, upcoming, general)
- 규칙 기반 폴백 (Transformers 없어도 작동)

**주요 기능:**
- 감정 분석 (긍정/부정/중립)
- 위험 키워드 가중치 분석 (very_high: x3, high: x2, medium: x1)
- 5가지 위험 유형 자동 감지 (conflict, protest, natural_disaster, checkpoint, other)

### 2. LSTM 시계열 예측

**과거 패턴 학습으로 미래 위험 예측**

- 향후 1-14일 위험 발생 예측
- 이상 징후 실시간 감지 (Anomaly Detection)
- 위험 핫스팟 지리적 식별
- 통계적 패턴 분석

**예측 종류:**
- **미래 위험 예측**: 과거 발생 패턴 기반 (확률, 위치, 유형)
- **이상 탐지**: 위험도 급증, 빈도 급증, 새로운 유형 출현
- **핫스팟 식별**: 지리적 클러스터링으로 위험 집중 지역 파악

### 3. 크라우드소싱 신뢰도 평가

**사용자 제보 자동 검증 시스템**

- 신뢰도 점수 계산 (0-100점)
- 5가지 요소 종합 평가
- 스팸/허위 정보 필터링
- 사용자 평판 관리

**신뢰도 평가 요소:**
- 사용자 평판 (30%)
- 데이터 일관성 (25%)
- 교차 검증 (20%)
- 시의성 (15%)
- 완전성 (10%)

---

**작성일:** 2025-11-05
**최종 업데이트:** 2025-11-05
**버전:** 3.0.0 (Phase 1, 2, 3 완료)
