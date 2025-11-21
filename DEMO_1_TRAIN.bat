@echo off
chcp 65001 > nul
echo ============================================================
echo VeriSafe AI 모델 학습 시연
echo ============================================================
echo.
echo [설명]
echo 이 스크립트는 두 개의 딥러닝 모델을 학습시킵니다:
echo   1. 시간대별 승수 예측 모델 (1-2분 소요)
echo   2. LSTM 위험도 예측 모델 (5-10분 소요)
echo.
echo 예상 소요 시간: 10-15분
echo.
pause

cd backend

echo.
echo [1단계] 가상환경 활성화...
call venv\Scripts\activate

echo.
echo [2단계] 모델 학습 시작...
python train_models.py

echo.
echo ============================================================
echo [완료] 모델 학습이 완료되었습니다!
echo ============================================================
echo.
echo 다음 단계:
echo   - DEMO_2_TEST.bat: 학습된 모델 테스트
echo   - DEMO_3_API.bat: API 서버 시작 후 실시간 예측 테스트
echo.
pause
