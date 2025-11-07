"""딥러닝 모델 학습 스크립트"""
import asyncio
import sys
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from app.database import SessionLocal
from app.services.ai.improved_risk_predictor import improved_risk_predictor
from app.services.ai.time_multiplier_nn import time_multiplier_predictor


async def train_all_models():
    """모든 딥러닝 모델 학습"""
    db = SessionLocal()

    try:
        print("\n" + "="*70)
        print("VeriSafe 딥러닝 모델 학습 시작")
        print("="*70 + "\n")

        # 1. 시간대별 승수 모델 학습 (빠름 ~1-2분)
        print("\n[1/2] 시간대별 승수 예측 모델 학습...")
        print("-" * 70)

        time_result = await time_multiplier_predictor.train_model(
            db=db,
            epochs=50,
            batch_size=32,
            learning_rate=0.01
        )

        print(f"\n[OK] Time Multiplier model training completed")
        print(f"  - Final loss: {time_result.get('best_val_loss', 'N/A')}")
        print(f"  - Model path: {time_result.get('model_path', 'N/A')}")

        # 2. LSTM 위험도 예측 모델 학습 (느림 ~5-10분)
        print("\n[2/2] LSTM Risk Prediction model training...")
        print("-" * 70)

        risk_result = await improved_risk_predictor.train_model(
            db=db,
            epochs=100,
            batch_size=32,
            learning_rate=0.001
        )

        print(f"\n[OK] LSTM Risk Prediction model training completed")
        print(f"  - Final train loss: {risk_result.get('final_train_loss', 'N/A'):.4f}")
        print(f"  - Final val loss: {risk_result.get('final_val_loss', 'N/A'):.4f}")
        print(f"  - Training samples: {risk_result.get('training_samples', 'N/A')}")
        print(f"  - Model path: {risk_result.get('model_path', 'N/A')}")

        print("\n" + "="*70)
        print("[SUCCESS] All models trained successfully!")
        print("="*70 + "\n")

        print("Training Summary:")
        print(f"  1. Time Multiplier model: {time_result.get('status', 'N/A')}")
        print(f"  2. LSTM Risk model: {risk_result.get('status', 'N/A')}")

        print("\nNext steps:")
        print("  - Start backend: uvicorn app.main:app --reload")
        print("  - Test prediction: POST /api/ai/predict/deep-learning")
        print("  - Compare methods: GET /api/ai/predict/compare")

        return {
            "time_multiplier": time_result,
            "risk_predictor": risk_result
        }

    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        db.close()


if __name__ == "__main__":
    print("\n=== VeriSafe Deep Learning Model Training ===\n")
    print("This script trains two models:")
    print("  1. Time Multiplier Prediction (Neural Network)")
    print("  2. LSTM Risk Prediction (Bidirectional LSTM + Attention)")
    print("\nEstimated time: 10-15 minutes")
    print("="*70 + "\n")

    # 비동기 실행
    results = asyncio.run(train_all_models())

    if results:
        print("\n[SUCCESS] Training completed successfully!")
    else:
        print("\n[ERROR] Training failed.")

    print("\nPress Ctrl+C to exit...")
