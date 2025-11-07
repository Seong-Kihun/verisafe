# VeriSafe 테스트/시연 모드 가이드

## 개요
테스트 및 시연을 위해 보안 검증을 완화했습니다.
개발 환경에서는 기본 설정으로 바로 실행 가능합니다!

---

## 🚀 빠른 시작

### 백엔드 실행
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m app.main
```

**보안 설정**: ✅ 기본값 사용 가능
- DEBUG=True이므로 경고만 표시되고 서버는 정상 시작됩니다
- 별도 .env 파일 생성 **불필요**

### 모바일 앱 실행
```bash
cd mobile
npm install
npm start
```

**API 설정**: ✅ 기본 IP 사용
- `192.168.45.177:8000`으로 설정되어 있음
- 경고 메시지 제거됨

---

## 📝 현재 설정 (테스트 모드)

### 보안 검증
| 환경 | 동작 | 설명 |
|-----|------|------|
| **개발 (DEBUG=True)** | ⚠️ 경고만 | 기본 비밀번호 사용해도 서버 시작 |
| **프로덕션 (DEBUG=False)** | ❌ 에러 | 기본 비밀번호 사용 시 서버 시작 차단 |

### 기본 설정값
```python
# backend/app/config.py
DEBUG = True  # 개발 모드
SECRET_KEY = "CHANGE-ME-IN-PRODUCTION"
DATABASE_PASSWORD = "verisafe_pass_2025"
REDIS_PASSWORD = "verisafe_redis_2025"
```

### API 기본 URL
```javascript
// mobile/src/services/api.js
// 웹: http://localhost:8000
// 모바일: http://192.168.45.177:8000
```

---

## 🎯 테스트/시연 시 주의사항

### 서버 시작 시 메시지
```
⚠️  보안 경고 (테스트/개발용): SECRET_KEY가 기본값입니다, DATABASE_PASSWORD가 기본값입니다, REDIS_PASSWORD가 기본값입니다
   프로덕션 배포 시 반드시 .env 파일에서 변경하세요!
```

**→ 이 경고는 무시하고 계속 진행하셔도 됩니다!**

### 네트워크 설정
PC의 IP 주소가 `192.168.45.177`이 아닌 경우:
1. PC의 실제 IP 확인: `ipconfig` (Windows) 또는 `ifconfig` (Mac/Linux)
2. `mobile/src/services/api.js` 26번째 줄 수정
3. 또는 `.env` 파일에 `EXPO_PUBLIC_API_URL=http://YOUR_IP:8000` 설정

---

## 🔒 프로덕션 배포 시

### 1. 보안 설정 필수 변경
```bash
# backend/.env 파일 생성
DEBUG=False
SECRET_KEY=your-secret-key-min-32-chars
DATABASE_PASSWORD=your-secure-password
REDIS_PASSWORD=your-secure-password
```

### 2. 환경 변수 설정
```bash
# mobile/.env 파일 생성
EXPO_PUBLIC_API_URL=https://your-production-api.com
```

### 3. 보안 검증 동작
- `DEBUG=False`로 설정하면 기본 비밀번호 사용 시 서버가 시작되지 않습니다
- 반드시 모든 비밀번호를 변경해야 합니다

---

## 💡 유용한 팁

### 개발 중 변경된 IP 주소 적용
PC IP가 자주 바뀌는 경우:
```bash
# mobile/.env 파일 생성 (우선순위 높음)
EXPO_PUBLIC_API_URL=http://192.168.1.100:8000
```

### 로깅 레벨 조정
```python
# backend/app/utils/logger.py
# 개발 중 디버그 로그 보고 싶을 때
level=logging.DEBUG
```

### CORS 설정 추가
```python
# backend/app/config.py
allowed_origins: str = "http://localhost:8081,http://192.168.45.177:8081,http://YOUR_IP:8081"
```

---

## 📞 문제 해결

### 1. 서버가 시작되지 않음
- PostgreSQL 실행 확인: `pg_ctl status`
- Redis 실행 확인: `redis-cli ping`

### 2. 모바일 앱이 API에 연결 안됨
- 백엔드 서버 실행 확인: `http://localhost:8000/docs`
- IP 주소 확인: PC와 모바일이 같은 네트워크에 있어야 함
- 방화벽 확인: 8000번 포트 허용

### 3. "Port 8081 is being used" 오류
```bash
# 포트 사용 프로세스 확인
netstat -ano | findstr :8081

# 프로세스 종료 (PID는 위 명령어 결과 참고)
taskkill /F /PID [PID]
```

---

## ✅ 현재 상태

- ✅ 개발/테스트 환경: 기본 설정으로 바로 실행 가능
- ✅ 보안 경고: 무시 가능 (경고만 표시)
- ✅ 모바일 앱: 경고 메시지 제거됨
- ✅ 프로덕션 보호: DEBUG=False 시 자동 차단

**편하게 테스트하세요!** 🎉
