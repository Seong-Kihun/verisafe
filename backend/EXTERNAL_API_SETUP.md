# 외부 API 설정 가이드

VeriSafe는 다양한 외부 데이터 소스에서 위험 정보를 수집합니다.

## 📊 지원하는 데이터 소스

### 1. ACLED (Armed Conflict Location & Event Data)
- **데이터**: 분쟁, 시위, 폭력 사건
- **커버리지**: 전 세계 (남수단 포함)
- **API 키**: **필수**

#### ACLED API 키 발급 방법:
1. https://acleddata.com/ 방문
2. 우측 상단 "Access Data" 클릭
3. "Register for Access" 선택
4. 이메일 주소로 가입 (무료)
5. API 키 발급 (이메일로 전송됨)

#### 설정 방법:
```bash
# .env 파일에 추가
ACLED_API_KEY=your_api_key_here
```

---

### 2. GDACS (Global Disaster Alert and Coordination System)
- **데이터**: 자연재해 (지진, 홍수, 산사태)
- **커버리지**: 전 세계
- **API 키**: **불필요** (공개 API)

#### 특징:
- API 키 없이 사용 가능
- XML/RSS 형식으로 데이터 제공
- South Sudan에 데이터가 없으면 자동으로 주변 국가(Uganda, Kenya, Ethiopia, Sudan) 검색

---

### 3. ReliefWeb (Humanitarian Information Service)
- **데이터**: 인도적 지원 보고서
- **커버리지**: 전 세계
- **API 키**: **불필요**

---

## 🚀 데이터 수집 실행

### 자동 수집 (스케줄러)
서버 시작 시 자동으로 6시간마다 데이터 수집:
```bash
cd backend
python -m uvicorn app.main:app --reload
```

### 수동 수집 (테스트/디버깅)
```bash
# Python 스크립트로 직접 실행
cd backend
python -m app.services.external_data.test_collectors
```

### API 엔드포인트로 수집
```bash
# POST 요청으로 즉시 수집
curl -X POST http://localhost:8000/api/external-data/collect

# 수집 상태 확인
curl http://localhost:8000/api/external-data/status
```

---

## 📈 수집 통계 확인

### 데이터베이스에서 확인:
```bash
cd backend
python -c "from app.database import get_db; from app.models.hazard import Hazard; from sqlalchemy import func; db = next(get_db()); sources = db.query(Hazard.source, func.count(Hazard.id)).group_by(Hazard.source).all(); print('Data sources:'); [print(f'  {s[0]}: {s[1]} records') for s in sources]"
```

### API로 확인:
```bash
curl http://localhost:8000/api/external-data/status
```

예상 출력:
```json
{
  "status": "success",
  "data_sources": {
    "acled": {
      "name": "ACLED (Armed Conflict Location & Event Data)",
      "count": 45,
      "last_updated": "2025-01-06T10:30:00",
      "description": "분쟁 및 폭력 사건 데이터"
    },
    "gdacs": {
      "name": "GDACS (Global Disaster Alert and Coordination System)",
      "count": 12,
      "last_updated": "2025-01-06T10:30:00",
      "description": "재난 및 자연재해 데이터"
    }
  },
  "total_hazards": 57
}
```

---

## ⚠️ 트러블슈팅

### API 키가 없는 경우
- ACLED은 더미 데이터를 자동 생성합니다
- 프로덕션 환경에서는 반드시 실제 API 키 설정 필요

### 데이터가 수집되지 않는 경우
1. **네트워크 확인**: 외부 API 접근 가능한지 확인
2. **API 키 확인**: .env 파일에 올바른 API 키가 있는지 확인
3. **로그 확인**: 백엔드 콘솔에서 에러 메시지 확인
4. **테스트 스크립트 실행**:
   ```bash
   cd backend
   python test_external_api.py
   ```

### South Sudan에 데이터가 없는 경우
- GDACS는 자동으로 주변 국가 검색
- 검색 결과가 없으면 더미 데이터 생성
- 더미 데이터는 `verified=False`로 표시됨

---

## 🎯 개선사항 (2025-01-06)

### GDACS Collector
✅ Status 204 (No Content) 처리 추가
✅ 주변 국가 검색 기능 추가 (Uganda, Kenya, Ethiopia, Sudan)
✅ 더미 데이터 품질 개선 (랜덤 좌표, 시간)

### ACLED Collector
✅ 더미 데이터 다양성 개선
✅ 현실적인 이벤트 타입 및 위치
✅ 랜덤 날짜 생성 (최근 7일)

### 공통
✅ 더미 데이터에 `verified=False` 플래그 추가
✅ 오류 처리 강화
✅ 로깅 개선

---

## 📝 참고 자료

- ACLED API 문서: https://acleddata.com/knowledge-base/api-user-guide/
- GDACS API 문서: https://www.gdacs.org/About/dataintegration.aspx
- ReliefWeb API 문서: https://apidoc.reliefweb.int/

---

## 🔒 보안 주의사항

- .env 파일을 **절대** Git에 커밋하지 마세요
- API 키를 코드에 하드코딩하지 마세요
- 프로덕션 환경에서는 환경 변수로 설정하세요
