# VeriSafe SOS 긴급 기능 - 테스트 가이드

## 개요

VeriSafe의 SOS 긴급 기능은 위험 상황에서 사용자가 긴급 연락처에게 자동으로 도움을 요청할 수 있는 핵심 안전 기능입니다.

## 기능 설명

### 주요 기능
1. **긴급 연락처 관리** (최대 5명)
   - 이름, 전화번호, 이메일, 관계 설정
   - 우선순위 자동 지정 (등록 순서)
   - 위치 공유 설정

2. **SOS 발동**
   - 5초 카운트다운 (자동 발송)
   - 즉시 발송 옵션
   - 취소 가능

3. **자동 알림 전송**
   - SMS로 긴급 연락처에게 위치 정보 전송
   - Google Maps 링크 포함
   - 백엔드 서버에 SOS 이벤트 기록

4. **가장 가까운 안전 대피처 표시**
   - SOS 발동 시 자동 계산
   - 거리 정보 제공

5. **오프라인 모드 지원**
   - 서버 연결 실패 시에도 SMS 발송 가능
   - Fallback으로 직접 전화 걸기 옵션

## 아키텍처

### Backend (FastAPI)

**모델 (app/models/sos_event.py)**
```python
class SOSEvent:
    - id: 이벤트 고유 ID
    - user_id: 사용자 ID
    - latitude, longitude: 발생 위치
    - message: 긴급 메시지
    - status: active/cancelled/resolved
    - created_at, updated_at, resolved_at
```

**API 엔드포인트 (app/routes/emergency.py)**
- `POST /api/emergency/sos` - SOS 발동
- `POST /api/emergency/sos/{id}/cancel` - SOS 취소
- `GET /api/emergency/sos/user/{user_id}` - 사용자 SOS 이력 조회
- `GET /api/emergency/sos/{id}` - SOS 상세 조회

### Mobile (React Native)

**주요 컴포넌트**
- `SOSConfirmModal` - SOS 확인 모달 (5초 카운트다운)
- `EmergencyContactsScreen` - 긴급 연락처 목록
- `EmergencyContactEditScreen` - 연락처 추가/편집
- `MapScreen` - SOS 버튼 및 로직

**서비스**
- `services/sms.js` - SMS 발송 (expo-sms)
- `services/api.js` - Backend API 통신
- `services/storage.js` - 로컬 저장소 (AsyncStorage)

## 테스트 시나리오

### 1. 긴급 연락처 등록

**테스트 단계:**
1. 앱 실행 → 프로필 탭 → "긴급 연락망"
2. "+" 버튼 클릭
3. 연락처 정보 입력:
   - 이름: "테스트 연락처"
   - 전화번호: "+211-XXX-XXX-XXX" 또는 로컬 번호
   - 관계: 선택 (가족/친구/동료/기타)
   - 위치 공유: ON
4. "추가" 버튼 클릭

**예상 결과:**
- 연락처가 우선순위 1로 추가됨
- 목록에 표시됨

**검증:**
```javascript
// 개발자 도구 콘솔에서
import { emergencyContactsStorage } from './services/storage';
const contacts = await emergencyContactsStorage.getAll();
console.log(contacts);
```

### 2. SOS 발동 (정상 시나리오)

**사전 조건:**
- 최소 1명의 긴급 연락처 등록
- 위치 권한 허용
- 백엔드 서버 실행 중
- SMS 기능 사용 가능

**테스트 단계:**
1. 지도 화면에서 우측 하단 "SOS" 버튼 클릭
2. 모달 확인:
   - 5초 카운트다운 표시
   - 긴급 연락처 수 표시
   - 현재 위치 정보 표시
3. "즉시 발송" 또는 카운트다운 대기

**예상 결과:**
- SMS 전송 화면 표시 (expo-sms)
- 백엔드에 SOS 이벤트 저장
- 성공 알림:
  ```
  🆘 SOS 발송 완료
  긴급 알림이 전송되었습니다.
  ✅ 1명에게 SMS 전송 완료

  📍 가장 가까운 대피처:
  [대피처 이름]
  거리: XXXm
  ```

**검증:**
```bash
# Backend 로그 확인
# [SOS] User 1 activated SOS at (4.8594, 31.5713)
# [SOS] SOS event X created. Nearest haven: ...

# SMS 내용 확인
🆘 긴급 SOS 알림 🆘

[사용자]님이 긴급 상황에 처했습니다.

📅 시간: 2025-11-08 오후 3:30
📍 위치:
위도: 4.859400
경도: 31.571300

🗺️ 지도에서 위치 확인:
https://www.google.com/maps?q=4.8594,31.5713

즉시 확인하고 연락해주세요!

- VeriSafe 긴급 알림 시스템
```

### 3. SOS 취소

**테스트 단계:**
1. SOS 모달 표시 상태에서
2. "취소" 버튼 클릭

**예상 결과:**
- 카운트다운 중지
- 모달 닫힘
- SMS 전송 안 됨
- 백엔드 이벤트 생성 안 됨

### 4. 오프라인 모드 (서버 연결 실패)

**사전 조건:**
- 백엔드 서버 중지 또는 네트워크 차단

**테스트 단계:**
1. SOS 버튼 클릭 → 즉시 발송

**예상 결과:**
- SMS는 정상 발송
- 성공 알림에 경고 표시:
  ```
  🆘 SOS 발송 완료
  긴급 알림이 전송되었습니다.
  ✅ 1명에게 SMS 전송 완료
  ⚠️ 서버 연결 실패 (오프라인 모드)
  ```

**검증:**
- 콘솔 로그에 `[SOS] Backend failed, continuing with SMS` 표시

### 5. SMS 실패 시 Fallback

**사전 조건:**
- SMS 사용 불가 환경 (시뮬레이터 등)

**테스트 단계:**
1. SOS 버튼 클릭 → 즉시 발송

**예상 결과:**
- 전화 걸기 옵션 제공:
  ```
  SMS 사용 불가
  이 기기에서는 SMS를 사용할 수 없습니다.
  [연락처 이름]([전화번호])에게 직접 전화하시겠습니까?

  [취소] [전화 걸기]
  ```

### 6. SOS 이력 조회 (API 테스트)

**테스트 방법:**
```bash
# 사용자 SOS 이력 조회
curl -X GET "http://localhost:8000/api/emergency/sos/user/1"

# 특정 상태 필터
curl -X GET "http://localhost:8000/api/emergency/sos/user/1?status=active"

# SOS 상세 조회
curl -X GET "http://localhost:8000/api/emergency/sos/1"
```

**예상 응답:**
```json
{
  "total": 1,
  "events": [
    {
      "id": 1,
      "user_id": 1,
      "latitude": 4.8594,
      "longitude": 31.5713,
      "message": "긴급 SOS 요청",
      "status": "active",
      "created_at": "2025-11-08T15:30:00Z",
      "updated_at": "2025-11-08T15:30:00Z",
      "resolved_at": null
    }
  ]
}
```

### 7. SOS 취소 API (Backend)

**테스트 방법:**
```bash
curl -X POST "http://localhost:8000/api/emergency/sos/1/cancel" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1}'
```

**예상 응답:**
```json
{
  "success": true,
  "message": "SOS cancelled successfully"
}
```

**검증:**
- `resolved_at` 필드가 현재 시간으로 업데이트됨
- `status`가 'cancelled'로 변경됨

## 에러 시나리오

### 1. 긴급 연락처 없이 SOS 발동
**예상:** "등록된 긴급 연락처가 없습니다" 알림

### 2. 위치 정보 없이 SOS 발동
**예상:** "현재 위치를 확인할 수 없습니다" 알림

### 3. 최대 5명 초과 등록 시도
**예상:** "긴급 연락처는 최대 5명까지만 등록할 수 있습니다" 알림

### 4. 잘못된 전화번호 형식
**예상:** "올바른 전화번호 형식을 입력해주세요" 알림

## 데이터베이스 확인

```sql
-- SOS 이벤트 조회
SELECT * FROM sos_events ORDER BY created_at DESC LIMIT 10;

-- 활성 SOS 이벤트
SELECT * FROM sos_events WHERE status = 'active';

-- 사용자별 SOS 통계
SELECT user_id, COUNT(*) as total_sos,
       COUNT(CASE WHEN status = 'active' THEN 1 END) as active_count
FROM sos_events
GROUP BY user_id;
```

## 보안 고려사항

1. **사용자 인증**
   - 현재: user_id를 클라이언트에서 전송
   - 향후: JWT 토큰 기반 인증 추가 필요

2. **개인정보 보호**
   - 긴급 연락처는 로컬 저장소에만 보관
   - 서버에는 SOS 이벤트만 기록

3. **위치 정보**
   - 사용자 동의 후에만 수집
   - SOS 발동 시에만 전송

## 알려진 제약사항

1. **SMS 전송**
   - iOS 시뮬레이터에서는 작동 안 함 (실제 기기 필요)
   - Android 에뮬레이터에서는 제한적

2. **백그라운드 실행**
   - 현재 앱이 활성 상태일 때만 SOS 가능
   - 향후: 백그라운드 위치 추적 + 자동 SOS 고려

3. **네트워크 의존성**
   - 가장 가까운 대피처 정보는 서버 연결 필요
   - SMS는 오프라인에서도 작동

## 개선 사항 (향후)

1. **Push Notification** 추가
   - SOS 발동 시 서버에서 관리자에게 알림

2. **WebSocket 실시간 통신**
   - SOS 상태 실시간 업데이트

3. **위치 추적**
   - SOS 활성화 중 위치 계속 업데이트

4. **SOS 이력 UI**
   - 모바일 앱에서 SOS 이력 확인 화면 추가

5. **자동 해제**
   - 일정 시간 경과 후 자동으로 resolved 상태로 변경

## 문의

문제 발생 시 로그 확인:
- Backend: `backend/logs/`
- Mobile: React Native 개발자 도구 콘솔

---

**마지막 업데이트:** 2025-11-08
**테스트 완료 버전:** v1.0.0
