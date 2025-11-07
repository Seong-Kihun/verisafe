"""딥러닝 모델 테스트 스크립트"""
import sys
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 Python path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from app.services.ai.improved_risk_predictor import improved_risk_predictor
from app.services.ai.time_multiplier_nn import time_multiplier_predictor


def test_models():
    """학습된 모델 테스트"""
    print("\n" + "="*70)
    print("Deep Learning Models Test")
    print("="*70 + "\n")

    # 테스트 데이터 (주바 지역)
    test_locations = [
        {"lat": 4.8517, "lon": 31.5825, "name": "Juba City Center"},
        {"lat": 4.86, "lon": 31.60, "name": "Juba North"},
        {"lat": 4.83, "lon": 31.57, "name": "Juba South"},
    ]

    # 테스트 시간
    test_times = [
        {"hour": 9, "desc": "Morning (9 AM)"},
        {"hour": 18, "desc": "Evening (6 PM)"},
        {"hour": 23, "desc": "Night (11 PM)"},
    ]

    print("1. Model Status")
    print("-" * 70)
    print(f"LSTM Risk Predictor trained: {improved_risk_predictor.is_trained}")
    print(f"Time Multiplier trained: {time_multiplier_predictor.is_trained}\n")

    if not improved_risk_predictor.is_trained:
        print("[WARNING] LSTM model not trained. Run train_models.py first.")
        return

    if not time_multiplier_predictor.is_trained:
        print("[WARNING] Time Multiplier model not trained. Run train_models.py first.")
        return

    print("2. Prediction Tests")
    print("-" * 70)

    for location in test_locations:
        print(f"\nLocation: {location['name']} ({location['lat']}, {location['lon']})")
        print("  " + "-" * 66)

        for time_info in test_times:
            # 현재 날짜에 시간만 변경
            test_time = datetime.now().replace(hour=time_info['hour'], minute=0, second=0)

            # 딥러닝 예측
            base_risk, confidence, method = improved_risk_predictor.predict_risk(
                latitude=location['lat'],
                longitude=location['lon'],
                timestamp=test_time
            )

            # 시간 승수
            multiplier, mult_method = time_multiplier_predictor.get_multiplier(
                timestamp=test_time,
                use_hybrid=True
            )

            # 최종 위험도
            final_risk = base_risk * multiplier

            print(f"  {time_info['desc']:20} | "
                  f"Base: {base_risk:5.1f} | "
                  f"Mult: {multiplier:4.2f} | "
                  f"Final: {final_risk:5.1f} | "
                  f"Conf: {confidence:.2f}")

    print("\n3. Method Comparison (Juba Center, 6 PM)")
    print("-" * 70)

    test_time = datetime.now().replace(hour=18, minute=0)
    lat, lon = 4.8517, 31.5825

    # 딥러닝
    dl_risk, dl_conf, dl_method = improved_risk_predictor.predict_risk(
        lat, lon, test_time
    )
    dl_mult, _ = time_multiplier_predictor.get_multiplier(test_time, False)

    # 규칙 기반
    rule_risk = improved_risk_predictor._rule_based_predict(lat, lon, test_time)
    stat_mult = time_multiplier_predictor.statistical_calculator.get_multiplier(test_time)

    print(f"\nDeep Learning:")
    print(f"  Risk: {dl_risk:.2f}, Multiplier: {dl_mult:.2f}, Final: {dl_risk * dl_mult:.2f}")
    print(f"  Confidence: {dl_conf:.2f}, Method: {dl_method}")

    print(f"\nRule-based:")
    print(f"  Risk: {rule_risk:.2f}, Multiplier: {stat_mult:.2f}, Final: {rule_risk * stat_mult:.2f}")

    print(f"\nDifference:")
    print(f"  Risk: {dl_risk - rule_risk:+.2f}")
    print(f"  Final: {(dl_risk * dl_mult) - (rule_risk * stat_mult):+.2f}")

    print("\n" + "="*70)
    print("[SUCCESS] All tests completed!")
    print("="*70)

    print("\nNext steps:")
    print("  - Start backend: uvicorn app.main:app --reload")
    print("  - API endpoint: POST /api/ai/predict/deep-learning")
    print("  - Compare endpoint: GET /api/ai/predict/compare")


if __name__ == "__main__":
    try:
        test_models()
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\nPress Ctrl+C to exit...")
