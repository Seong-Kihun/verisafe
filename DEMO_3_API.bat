@echo off
chcp 65001 > nul
echo ============================================================
echo VeriSafe API 서버 시작
echo ============================================================
echo.
echo [설명]
echo 백엔드 API 서버를 시작합니다.
echo 서버가 실행되면 다른 터미널에서 API 테스트를 할 수 있습니다.
echo.
echo 서버 주소: http://localhost:8000
echo API 문서: http://localhost:8000/docs
echo.
echo [중요] 이 창은 닫지 마세요!
echo        서버를 중지하려면 Ctrl+C를 누르세요.
echo.
pause

cd backend

echo.
echo [1단계] 가상환경 활성화...
call venv\Scripts\activate

echo.
echo [2단계] API 서버 시작...
echo.
echo 서버가 시작되면:
echo   1. 새 터미널을 열어서
echo   2. DEMO_4_PREDICT.bat 실행 (실시간 예측 테스트)
echo   또는
echo   브라우저에서 http://localhost:8000/docs 접속
echo.
uvicorn app.main:app --reload
