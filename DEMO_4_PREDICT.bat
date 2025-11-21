@echo off
chcp 65001 > nul
echo ============================================================
echo VeriSafe AI 실시간 예측 시연
echo ============================================================
echo.
echo [주의] 이 스크립트를 실행하기 전에:
echo   1. DEMO_3_API.bat로 서버를 먼저 시작해야 합니다
echo   2. 서버가 http://localhost:8000 에서 실행 중이어야 합니다
echo.
pause

echo.
echo ============================================================
echo 테스트 1: 저녁 6시 위험도 예측 (위험도 높음)
echo ============================================================
echo.
echo [요청] 주바 시내 중심가, 저녁 6시
curl -X POST "http://localhost:8000/api/ai/predict/deep-learning" ^
  -H "Content-Type: application/json" ^
  -d "{\"latitude\": 4.8517, \"longitude\": 31.5825, \"timestamp\": \"2025-11-18T18:00:00Z\"}"

echo.
echo.
pause

echo.
echo ============================================================
echo 테스트 2: 아침 9시 위험도 예측 (위험도 낮음)
echo ============================================================
echo.
echo [요청] 같은 장소, 아침 9시
curl -X POST "http://localhost:8000/api/ai/predict/deep-learning" ^
  -H "Content-Type: application/json" ^
  -d "{\"latitude\": 4.8517, \"longitude\": 31.5825, \"timestamp\": \"2025-11-18T09:00:00Z\"}"

echo.
echo.
pause

echo.
echo ============================================================
echo 테스트 3: 예측 방법 비교 (딥러닝 vs 통계)
echo ============================================================
echo.
curl "http://localhost:8000/api/ai/predict/compare?latitude=4.8517&longitude=31.5825"

echo.
echo.
pause

echo.
echo ============================================================
echo 테스트 4: 향후 7일 위험 예측
echo ============================================================
echo.
curl "http://localhost:8000/api/ai/predict-future?days_ahead=7"

echo.
echo.
pause

echo.
echo ============================================================
echo 테스트 5: 이상 징후 감지 (최근 24시간)
echo ============================================================
echo.
curl "http://localhost:8000/api/ai/detect-anomalies?hours=24"

echo.
echo.
pause

echo.
echo ============================================================
echo 테스트 6: 위험 핫스팟 예측
echo ============================================================
echo.
curl "http://localhost:8000/api/ai/predict-hotspots?grid_size_km=10"

echo.
echo.
pause

echo.
echo ============================================================
echo 테스트 7: 종합 AI 대시보드
echo ============================================================
echo.
curl "http://localhost:8000/api/ai/analytics/overview"

echo.
echo.
pause

echo.
echo ============================================================
echo 테스트 8: 모델 정보 조회
echo ============================================================
echo.
curl "http://localhost:8000/api/ai/models/info"

echo.
echo.
echo ============================================================
echo [완료] 모든 API 테스트가 완료되었습니다!
echo ============================================================
echo.
echo 더 많은 API 테스트를 하려면:
echo   브라우저에서 http://localhost:8000/docs 를 열어
echo   Swagger UI에서 직접 테스트할 수 있습니다.
echo.
pause
