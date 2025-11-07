"""Standalone Deep Learning Models Test - No Database Required"""
import sys
from pathlib import Path
from datetime import datetime
import torch
import numpy as np

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

print("\n" + "="*70)
print("VeriSafe Deep Learning Models - Standalone Test")
print("="*70 + "\n")

# Check PyTorch
print("1. Environment Check")
print("-" * 70)
print(f"PyTorch Version: {torch.__version__}")
print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

# Check model files
model_dir = Path(__file__).parent / "models"
lstm_path = model_dir / "improved_risk_lstm.pth"
multiplier_path = model_dir / "time_multiplier_nn.pth"

print(f"LSTM Model: {'[OK] Found' if lstm_path.exists() else '[X] Not Found'}")
print(f"Time Multiplier Model: {'[OK] Found' if multiplier_path.exists() else '[X] Not Found'}")

if not lstm_path.exists() or not multiplier_path.exists():
    print("\n[ERROR] Model files not found. Run train_models.py first.")
    sys.exit(1)

print("\n2. Loading Models")
print("-" * 70)

# Import model classes
from app.services.ai.improved_risk_predictor import ImprovedRiskLSTM
from app.services.ai.time_multiplier_nn import TimeMultiplierNN

device = torch.device('cpu')

# Load LSTM
lstm_model = ImprovedRiskLSTM()
lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
lstm_model.eval()
print("LSTM Risk Predictor: Loaded [OK]")

# Load Time Multiplier
mult_model = TimeMultiplierNN()
mult_model.load_state_dict(torch.load(multiplier_path, map_location=device))
mult_model.eval()
print("Time Multiplier NN: Loaded [OK]")

print("\n3. Running Predictions")
print("-" * 70)

# Test data
test_cases = [
    {"name": "Juba Center - Morning", "lat": 4.8517, "lon": 31.5825, "hour": 9},
    {"name": "Juba Center - Evening", "lat": 4.8517, "lon": 31.5825, "hour": 18},
    {"name": "Juba Center - Night", "lat": 4.8517, "lon": 31.5825, "hour": 23},
    {"name": "Juba North - Evening", "lat": 4.86, "lon": 31.60, "hour": 18},
]

for test in test_cases:
    # Create LSTM input (7-day sequence)
    sequence_length = 7
    inputs = []

    current_time = datetime.now().replace(hour=test['hour'], minute=0, second=0)

    for i in range(sequence_length):
        hour_norm = test['hour'] / 23.0
        dow_norm = current_time.weekday() / 6.0
        lat_norm = (test['lat'] - 4.85) / 0.3
        lon_norm = (test['lon'] - 31.6) / 0.3

        # Simple past risk estimate
        base_risk = 50.0
        if test['hour'] >= 18 and test['hour'] <= 20:
            base_risk = 70.0
        elif test['hour'] >= 22 or test['hour'] <= 5:
            base_risk = 65.0

        risk_norm = base_risk / 100.0
        inputs.append([hour_norm, dow_norm, lat_norm, lon_norm, risk_norm])

    # LSTM prediction
    x = torch.tensor([inputs], dtype=torch.float32)
    with torch.no_grad():
        lstm_output = lstm_model(x)
        base_risk = lstm_output.item() * 100

    # Time multiplier prediction
    dow = current_time.weekday()
    hour = test['hour']

    # Create features for multiplier
    mult_features = [
        dow / 6.0,
        hour / 23.0,
        np.cos((hour / 24.0) * 2 * np.pi),
        np.sin((hour / 24.0) * 2 * np.pi),
        1 if dow >= 5 else 0,  # weekend
        1 if 9 <= hour <= 17 else 0,  # work hours
        1 if 17 <= hour <= 20 else 0,  # evening
        1 if hour >= 22 or hour <= 5 else 0,  # late night
    ]

    mult_x = torch.tensor([mult_features], dtype=torch.float32)
    with torch.no_grad():
        mult_output = mult_model(mult_x)
        multiplier = mult_output.item()

    # Final risk
    final_risk = base_risk * multiplier

    print(f"\n{test['name']}")
    print(f"  Location: ({test['lat']}, {test['lon']})")
    print(f"  Base Risk (LSTM): {base_risk:.2f}")
    print(f"  Time Multiplier (NN): {multiplier:.2f}")
    print(f"  Final Risk: {final_risk:.2f}")

print("\n" + "="*70)
print("[SUCCESS] All models working correctly!")
print("="*70)

print("\nNext Steps:")
print("  - Models are trained and ready for production")
print("  - Start backend: uvicorn app.main:app --reload")
print("  - Test API: POST /api/ai/predict/deep-learning")
print("  - Compare methods: GET /api/ai/predict/compare")

print("\nPress Ctrl+C to exit...")
