@echo off
chcp 65001 >nul
cls

echo ================================================
echo   VeriSafe - 초기 설정 (한 번만 실행)
echo ================================================
echo.
echo 이 스크립트는 다음 작업을 수행합니다:
echo   1. Docker 데이터베이스 시작
echo   2. Python 패키지 설치
echo   3. 환경 설정 및 데이터베이스 초기화
echo.
echo 소요 시간: 약 10분
echo.
pause
cls

cd /d "%~dp0"

REM ============================================
REM STEP 1: Docker 데이터베이스 시작
REM ============================================
echo.
echo ================================================
echo   [1/3] Docker 데이터베이스 시작
echo ================================================
echo.

echo Docker 확인 중...
docker --version >nul 2>&1
if errorlevel 1 (
    echo [오류] Docker가 설치되지 않았습니다!
    echo Docker Desktop을 설치해주세요:
    echo https://www.docker.com/products/docker-desktop/
    echo.
    pause
    exit /b 1
)

echo Docker 발견! 데이터베이스 시작 중...
docker-compose up -d

if errorlevel 1 (
    echo.
    echo [경고] Docker 컨테이너 시작에 문제가 있을 수 있습니다.
    echo 계속 진행합니다...
    echo.
)

echo.
echo 데이터베이스 준비 대기 중 (30초)...
timeout /t 30 /nobreak >nul

echo.
echo [완료] 데이터베이스 시작됨!
echo   - PostgreSQL: localhost:5432
echo   - Redis: localhost:6379
echo   - PgAdmin: http://localhost:5050
echo.
pause
cls

REM ============================================
REM STEP 2: Python 패키지 설치
REM ============================================
echo.
echo ================================================
echo   [2/3] Python 패키지 설치
echo   (5-10분 소요, 잠시만 기다려주세요...)
echo ================================================
echo.

cd backend

echo Python 버전 확인 중...
python --version
if errorlevel 1 (
    echo [오류] Python이 설치되지 않았습니다!
    echo Python 3.10 이상을 설치해주세요:
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)

echo.
echo Python 버전 체크 (3.10 이상 필요)...
python -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
    echo [경고] Python 3.10 이상을 권장합니다!
    echo 현재 버전에서는 호환성 문제가 발생할 수 있습니다.
    echo.
    pause
)

echo.
echo 가상환경 생성 중...
if not exist venv (
    python -m venv venv
    echo 가상환경 생성 완료!
) else (
    echo 가상환경이 이미 존재합니다.
)

echo.
echo 가상환경 활성화 중...
call venv\Scripts\activate.bat

echo.
echo 패키지 설치 중 (5-10분 소요)...
echo ☕ 커피 한 잔 하고 오세요!
echo.
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [오류] 패키지 설치 실패!
    echo requirements.txt를 확인해주세요.
    pause
    exit /b 1
)

echo.
echo [완료] Python 패키지 설치 완료!
echo.
pause
cls

REM ============================================
REM STEP 3: 환경 설정 및 데이터베이스 초기화
REM ============================================
echo.
echo ================================================
echo   [3/3] 환경 설정 및 데이터베이스 초기화
echo ================================================
echo.

echo .env 파일 확인 중...
if not exist .env (
    echo .env 파일 생성 중...
    (
        echo # VeriSafe Environment Configuration - Auto-generated
        echo # Generated on: %date% %time%
        echo.
        echo # Application
        echo APP_NAME=VeriSafe API
        echo DEBUG=true
        echo.
        echo # Database ^(PostgreSQL^)
        echo DATABASE_TYPE=postgresql
        echo DATABASE_HOST=localhost
        echo DATABASE_PORT=5432
        echo DATABASE_USER=verisafe_user
        echo DATABASE_PASSWORD=verisafe_pass_2025
        echo DATABASE_NAME=verisafe_db
        echo.
        echo # Redis
        echo REDIS_HOST=localhost
        echo REDIS_PORT=6379
        echo REDIS_PASSWORD=verisafe_redis_2025
        echo REDIS_DB=0
        echo REDIS_CACHE_TTL=300
        echo.
        echo # JWT Authentication
        echo SECRET_KEY=dev_secret_key_change_in_production_2025
        echo ALGORITHM=HS256
        echo ACCESS_TOKEN_EXPIRE_MINUTES=1440
        echo.
        echo # CORS
        echo ALLOWED_ORIGINS=http://localhost:8081,http://192.168.104.30:8081
        echo.
        echo # File Upload
        echo UPLOAD_DIR=./uploads
        echo MAX_UPLOAD_SIZE=10485760
        echo.
        echo # External APIs ^(Optional^)
        echo ACLED_API_KEY=
        echo GDACS_API_URL=https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH
        echo RELIEFWEB_API_URL=https://api.reliefweb.int/v1
        echo TWITTER_BEARER_TOKEN=
        echo NEWS_API_KEY=
        echo SENTINEL_CLIENT_ID=
        echo SENTINEL_CLIENT_SECRET=
    ) > .env
    echo .env 파일 생성 완료!
) else (
    echo .env 파일이 이미 존재합니다.
)

echo.
echo 가상환경 활성화 중...
call venv\Scripts\activate.bat

echo.
echo 데이터베이스 테이블 생성 중...
python init_db.py

if errorlevel 1 (
    echo.
    echo [경고] 데이터베이스 초기화에 문제가 있을 수 있습니다.
    echo 하지만 계속 진행합니다...
    echo.
)

cd ..

echo.
echo ================================================
echo   ✅ 초기 설정 완료!
echo ================================================
echo.
echo 다음 단계:
echo   1. START.bat 실행하여 서버 시작
echo.
echo 서비스 정보:
echo   - API 문서: http://localhost:8000/docs
echo   - 프론트엔드: http://localhost:8081
echo   - 데이터베이스 관리: http://localhost:5050
echo     Email: admin@verisafe.com
echo     Password: admin2025
echo.
echo ================================================
echo.

pause
