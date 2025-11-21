@echo off
chcp 65001 > nul
echo ============================================================
echo VeriSafe AI 모델 테스트 시연
echo ============================================================
echo.
echo [설명]
echo 학습된 모델의 성능을 다양한 시나리오에서 테스트합니다:
echo   - 주바 3개 지역 (시내, 북부, 남부)
echo   - 3개 시간대 (아침 9시, 저녁 6시, 밤 11시)
echo   - 딥러닝 vs 통계 방법 비교
echo.
echo 예상 소요 시간: 1-2분
echo.
pause

cd backend

echo.
echo [1단계] 가상환경 활성화...
call venv\Scripts\activate

echo.
echo [2단계] 모델 테스트 시작...
python test_deep_learning.py

echo.
echo ============================================================
echo [완료] 모델 테스트가 완료되었습니다!
echo ============================================================
echo.
echo 다음 단계:
echo   - DEMO_3_API.bat: API 서버 시작
echo   - 그 후 DEMO_4_PREDICT.bat: 실시간 예측 테스트
echo.
pause
