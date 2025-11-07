# VeriSafe - 안전한 경로 추천 앱

> KOICA 사업 수행인력 및 구호활동가를 위한 안전 경로 네비게이션

---

## 🚀 빠른 시작 (3단계)

### 처음 실행 시 (한 번만)

1. **STEP1_database.bat** 실행
   - Docker 컨테이너 시작 (PostgreSQL, Redis, PgAdmin)
   - 30초 대기

2. **STEP2_install.bat** 실행
   - Python 가상환경 생성
   - 패키지 설치 (5-10분 소요, 커피 타임 ☕)
   - Python 3.10 이상 필요

3. **STEP3_setup.bat** 실행
   - .env 파일 자동 생성 (비밀번호 자동 설정)
   - 데이터베이스 테이블 생성
   - 1분 소요

### 서버 시작

- **START.bat** 실행 (자동으로 백엔드와 프론트엔드 시작)

**접속 URL:**
- API 문서: http://localhost:8000/docs
- 프론트엔드: http://localhost:8081
- 데이터베이스 관리 (PgAdmin): http://localhost:5050
  - Email: admin@verisafe.com
  - Password: admin2025

---

## 📦 프로젝트 구조

```
verisafe/
├── backend/          # FastAPI 백엔드
│   ├── app/          # 애플리케이션 코드
│   │   ├── models/   # 데이터베이스 모델
│   │   ├── routes/   # API 엔드포인트
│   │   ├── services/ # 비즈니스 로직
│   │   └── utils/    # 유틸리티
│   ├── migrations/   # 데이터베이스 마이그레이션
│   └── .env          # 환경 변수 (자동 생성됨)
├── mobile/           # React Native 앱
└── docker-compose.yml
```

---

## ✅ 구현 완료

### 백엔드 (Phase 1-5)
- ✅ PostgreSQL + PostGIS (공간 데이터베이스)
- ✅ Redis 캐싱 (99% 성능 향상)
- ✅ 외부 데이터 수집 (ACLED, GDACS, ReliefWeb)
- ✅ AI 위험도 예측 (LSTM)
- ✅ 위성사진 분석
- ✅ 사용자 제보 시스템
- ✅ 포인트 & 인센티브
- ✅ 관리자 대시보드

### 보안 & 성능
- ✅ bcrypt 비밀번호 해싱
- ✅ JWT 인증
- ✅ PostGIS 공간 인덱스 (99.5% 속도 향상)
- ✅ N+1 쿼리 최적화

### 모바일 앱
- ✅ 4탭 구조 (지도/제보/뉴스/내정보)
- ✅ 경로 계산 및 표시
- ✅ 위험 정보 시각화
- ✅ GPS 내 위치

---

## 📚 상세 문서

- **READ_ME_FIRST.txt** - 시작 가이드
- **STATUS.md** - 프로젝트 현황
- **backend/OPTIMIZATION_REPORT.md** - 최적화 상세 보고서
- **backend/OPTIMIZATION_SUMMARY.md** - 최적화 요약

---

## 🛠 기술 스택

**백엔드**
- FastAPI, SQLAlchemy, PostGIS
- Redis, JWT, bcrypt
- PyTorch, OpenCV

**모바일**
- React Native, Expo
- React Navigation
- Axios

**인프라**
- Docker, PostgreSQL, Redis

---

## 📊 프로젝트 상태

- 개발 진행률: **100%**
- 보안 강화: **완료**
- 성능 최적화: **완료**
- 프로덕션 준비: **90%**

---

**문의사항은 이슈로 남겨주세요!**
