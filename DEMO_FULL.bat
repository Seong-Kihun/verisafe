@echo off
chcp 65001 > nul
echo ============================================================
echo VeriSafe AI 전체 시연 (자동 실행)
echo ============================================================
echo.
echo [설명]
echo 이 스크립트는 AI 시스템 전체 시연을 자동으로 실행합니다:
echo.
echo   1단계: 모델 학습 (10-15분)
echo   2단계: 모델 테스트 (1-2분)
echo   3단계: 모델 정보 확인
echo.
echo 총 예상 시간: 15-20분
echo.
echo [주의]
echo   - API 서버 테스트는 별도로 DEMO_3_API.bat를 실행하세요
echo   - 이 스크립트는 학습과 테스트만 수행합니다
echo.
pause

cd backend

echo.
echo [준비] 가상환경 활성화...
call venv\Scripts\activate

echo.
echo.
echo ============================================================
echo 1단계: AI 모델 학습
echo ============================================================
echo.
echo [시작] 딥러닝 모델 학습을 시작합니다...
echo.
python train_models.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [오류] 모델 학습 중 오류가 발생했습니다.
    pause
    exit /b 1
)

echo.
echo [완료] 모델 학습이 완료되었습니다!
echo.
pause

echo.
echo.
echo ============================================================
echo 2단계: 학습된 모델 테스트
echo ============================================================
echo.
echo [시작] 다양한 시나리오에서 모델을 테스트합니다...
echo.
python test_deep_learning.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [오류] 모델 테스트 중 오류가 발생했습니다.
    pause
    exit /b 1
)

echo.
echo [완료] 모델 테스트가 완료되었습니다!
echo.
pause

echo.
echo.
echo ============================================================
echo [성공] 전체 시연이 완료되었습니다!
echo ============================================================
echo.
echo 학습된 모델:
echo   - LSTM 위험도 예측 모델: backend\models\lstm_risk_model.pth
echo   - 시간대별 승수 모델: backend\models\time_multiplier_model.pth
echo.
echo 다음 단계:
echo.
echo   [API 서버 테스트]
echo   1. 새 터미널에서 DEMO_3_API.bat 실행 (서버 시작)
echo   2. 또 다른 터미널에서 DEMO_4_PREDICT.bat 실행 (API 테스트)
echo.
echo   [웹 브라우저 테스트]
echo   1. DEMO_3_API.bat로 서버 시작
echo   2. 브라우저에서 http://localhost:8000/docs 접속
echo   3. Swagger UI에서 API 직접 테스트
echo.
pause
