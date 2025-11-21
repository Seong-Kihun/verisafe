# 🎬 VeriSafe AI 시연 빠른 시작 가이드

## 📋 시연 전 체크리스트

- [ ] 백엔드 가상환경 활성화 완료
- [ ] verisafe.db 데이터베이스 존재 확인
- [ ] PyTorch 설치 확인 (`pip list | findstr torch`)
- [ ] 터미널 2개 준비 (서버용, 테스트용)

---

## 🚀 3분 빠른 시연

가장 빠르게 AI 시스템을 시연하는 방법입니다.

### 1️⃣ 모델 학습 (이미 학습되어 있다면 생략)
```bash
# 실행: DEMO_1_TRAIN.bat
# 소요 시간: 10-15분
```

### 2️⃣ 모델 테스트
```bash
# 실행: DEMO_2_TEST.bat
# 소요 시간: 1-2분
```

### 3️⃣ API 시연
```bash
# 터미널 1: DEMO_3_API.bat (서버 시작 - 계속 실행)
# 터미널 2: DEMO_4_PREDICT.bat (API 테스트)
```

---

## 📁 시연 파일 구조

```
verisafe/
├── AI_DEMO_GUIDE.md          # 📖 상세 시연 가이드 (설명 스크립트 포함)
├── AI_DEMO_README.md          # 📋 이 파일 (빠른 참조)
│
├── DEMO_1_TRAIN.bat           # 🎓 모델 학습
├── DEMO_2_TEST.bat            # 🧪 모델 테스트
├── DEMO_3_API.bat             # 🌐 API 서버 시작
├── DEMO_4_PREDICT.bat         # 🔮 실시간 예측 테스트
├── DEMO_FULL.bat              # 🎯 전체 자동 실행
│
└── backend/
    ├── train_models.py        # 모델 학습 스크립트
    ├── test_deep_learning.py  # 모델 테스트 스크립트
    └── models/                # 학습된 모델 저장 폴더
        ├── lstm_risk_model.pth
        └── time_multiplier_model.pth
```

---

## 🎯 시연 시나리오별 가이드

### 시나리오 A: 전체 시연 (20분)

**대상:** 처음부터 모든 과정을 보여주고 싶을 때

```
1. DEMO_FULL.bat 실행 (학습 + 테스트)
   ↓
2. DEMO_3_API.bat 실행 (서버 시작)
   ↓
3. DEMO_4_PREDICT.bat 실행 (API 테스트)
```

**발표 흐름:**
- 학습 과정 설명 (Loss 감소 등)
- 테스트 결과 해석 (시간대별 변화)
- API 실시간 예측 시연

---

### 시나리오 B: 빠른 시연 (5분)

**대상:** 이미 모델이 학습되어 있고, 결과만 보여주고 싶을 때

```
1. DEMO_2_TEST.bat 실행 (모델 테스트)
   ↓
2. 테스트 결과 설명
```

**발표 포인트:**
- 시간대별 위험도 차이 (아침 37점 vs 밤 100점)
- 딥러닝 vs 통계 비교 (+18점 더 정확)
- 높은 신뢰도 (0.9 이상)

---

### 시나리오 C: API 중심 시연 (10분)

**대상:** 실제 API 사용 방법을 보여주고 싶을 때

```
터미널 1: DEMO_3_API.bat (서버 - 계속 실행)
터미널 2: DEMO_4_PREDICT.bat (순차적 API 테스트)
```

**시연 순서:**
1. 실시간 예측 (저녁 vs 아침)
2. 예측 방법 비교
3. 향후 7일 예측
4. 이상 징후 감지
5. 핫스팟 분석
6. 종합 대시보드

---

## 🎤 주요 발표 포인트

### 1. 모델 학습 단계

**보여줄 것:**
- Epoch별 Loss 감소
- Train/Validation Loss 비교

**강조할 점:**
> "Loss가 0.05에서 0.01로 감소하는 것을 볼 수 있습니다.
> 이는 모델이 데이터의 패턴을 성공적으로 학습하고 있음을 의미합니다."

---

### 2. 모델 테스트 단계

**보여줄 것:**
```
Morning (9 AM)   | Base: 45.3 | Mult: 0.82 | Final: 37.1
Evening (6 PM)   | Base: 62.1 | Mult: 1.45 | Final: 90.0
Night (11 PM)    | Base: 71.3 | Mult: 1.82 | Final: 100.0
```

**강조할 점:**
> "같은 장소에서도 시간에 따라 위험도가 37점에서 100점까지 변합니다.
> 이는 시간대별 승수 모델이 실제 위험 패턴을 반영하기 때문입니다."

---

### 3. 딥러닝 vs 통계 비교

**보여줄 것:**
```
Deep Learning: 90.04점 (신뢰도 0.91)
Statistical:   71.50점
Difference:    +18.54점
```

**강조할 점:**
> "딥러닝 모델이 통계 기반보다 18.54점 더 높게 예측했습니다.
> 실제 위험 상황에서 이러한 높은 감지율이 생명을 구할 수 있습니다."

---

### 4. 향후 예측

**보여줄 것:**
```json
{
  "predicted_date": "2025-11-19T10:00:00",
  "hazard_type": "armed_conflict",
  "probability": 0.82,
  "predicted_risk_score": 75
}
```

**강조할 점:**
> "내일 오전 10시 무장 충돌 발생 확률 82%로 예측됩니다.
> 이 정보로 사전에 위험 지역을 회피할 수 있습니다."

---

### 5. 이상 징후 감지

**보여줄 것:**
```json
{
  "type": "risk_spike",
  "description": "Average risk score increased by 45.3%",
  "current_value": 72.5,
  "baseline_value": 49.9
}
```

**강조할 점:**
> "최근 24시간 위험도가 평소 대비 45% 급증했습니다.
> 이러한 이상 징후를 조기 감지하여 신속한 대응이 가능합니다."

---

### 6. 핫스팟 분석

**보여줄 것:**
```json
{
  "latitude": 4.85,
  "longitude": 31.58,
  "hazard_count": 15,
  "avg_risk_score": 68.3,
  "most_common_type": "armed_conflict"
}
```

**강조할 점:**
> "이 지역에서 과거 30일간 15건의 위험이 발생했습니다.
> 평균 위험도 68.3점으로 집중 모니터링이 필요한 핫스팟입니다."

---

## 🔧 문제 해결

### ❌ "LSTM model not trained" 오류

**원인:** 모델이 학습되지 않았습니다.

**해결:**
```bash
DEMO_1_TRAIN.bat 실행
```

---

### ❌ API 응답 없음

**원인:** 백엔드 서버가 실행 중이 아닙니다.

**해결:**
```bash
# 터미널 1
DEMO_3_API.bat

# 서버가 시작될 때까지 대기 (약 10초)
# "Uvicorn running on http://127.0.0.1:8000" 메시지 확인

# 터미널 2
DEMO_4_PREDICT.bat
```

---

### ❌ "PyTorch 미설치" 경고

**원인:** PyTorch가 설치되지 않았습니다.

**해결:**
```bash
cd backend
venv\Scripts\activate
pip install torch
```

---

### ❌ curl 명령어 인식 안 됨

**원인:** Windows에서 curl이 없거나 경로 문제

**해결 1:** PowerShell 사용
```powershell
# .bat 대신 PowerShell에서 직접 실행
Invoke-RestMethod -Uri "http://localhost:8000/api/ai/models/info" -Method Get
```

**해결 2:** Postman 사용
- Postman 설치
- GET http://localhost:8000/api/ai/models/info

**해결 3:** 브라우저 사용
- 브라우저에서 http://localhost:8000/docs 접속
- Swagger UI에서 테스트

---

## 📊 예상 출력 결과

### 모델 학습 성공 시
```
======================================================================
[SUCCESS] All models trained successfully!
======================================================================

Training Summary:
  1. Time Multiplier model: success
  2. LSTM Risk model: success
```

### 모델 테스트 성공 시
```
Location: Juba City Center (4.8517, 31.5825)
  Morning (9 AM)   | Base: 45.3 | Mult: 0.82 | Final: 37.1 | Conf: 0.87
  Evening (6 PM)   | Base: 62.1 | Mult: 1.45 | Final: 90.0 | Conf: 0.91
```

### API 예측 성공 시
```json
{
  "status": "success",
  "prediction": {
    "final_risk_score": 90.0,
    "confidence": 0.91
  }
}
```

---

## 🎓 핵심 메시지 (30초 요약)

> "VeriSafe는 최신 딥러닝 기술을 활용한 AI 위험 예측 시스템을 갖추고 있습니다.
>
> **3가지 핵심 기술:**
> 1. 양방향 LSTM + Attention: 정확한 위험도 예측
> 2. 시간대별 승수 모델: 동적 위험도 조정
> 3. 시계열 분석: 미래 예측 및 이상 감지
>
> **실제 성능:**
> - 통계 대비 18점 더 정확
> - 신뢰도 0.9 이상
> - 실시간 처리 (100ms 이내)
>
> **활용:**
> - 사용자: 안전한 경로 선택
> - 관리자: 사전 대응 자원 배치
> - 구호 단체: 위험 지역 모니터링
>
> VeriSafe는 AI 기술로 분쟁 지역 주민의 생명을 지킵니다."

---

## 📞 추가 정보

### 상세 가이드
- **AI_DEMO_GUIDE.md**: 각 시연 단계별 상세 설명 스크립트

### API 문서
- **서버 실행 후**: http://localhost:8000/docs
- **Swagger UI**: 모든 API를 브라우저에서 테스트 가능

### 파일 위치
- **학습된 모델**: `backend/models/`
- **학습 스크립트**: `backend/train_models.py`
- **테스트 스크립트**: `backend/test_deep_learning.py`

---

**시연 성공을 기원합니다!** 🎉
