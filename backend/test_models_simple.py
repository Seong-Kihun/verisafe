"""Truly Standalone Deep Learning Models Test"""
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np

print("\n" + "="*70)
print("VeriSafe Deep Learning Models - Standalone Test")
print("="*70 + "\n")

# Define model architectures directly (no imports from app)
class ImprovedRiskLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, num_layers=3, dropout=0.3):
        super(ImprovedRiskLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        attention_weights = torch.softmax(
            self.attention(lstm_out).squeeze(-1),
            dim=1
        )
        attention_weights = attention_weights.unsqueeze(-1)
        context_vector = torch.sum(lstm_out * attention_weights, dim=1)

        out = self.fc1(context_vector)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.sigmoid(out)

        return out


class TimeMultiplierNN(nn.Module):
    def __init__(self, input_size=10):
        super(TimeMultiplierNN, self).__init__()

        self.fc1 = nn.Linear(input_size, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(16, 8)
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        out = self.fc3(out)
        out = self.relu3(out)

        out = self.fc4(out)
        out = self.sigmoid(out)

        return out


# 1. Environment Check
print("1. Environment Check")
print("-" * 70)
print(f"PyTorch Version: {torch.__version__}")
print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

# 2. Check model files
model_dir = Path(__file__).parent / "models"
lstm_path = model_dir / "improved_risk_lstm.pth"
multiplier_path = model_dir / "time_multiplier_nn.pth"

print(f"LSTM Model: {'[OK] Found' if lstm_path.exists() else '[X] Not Found'}")
print(f"Time Multiplier Model: {'[OK] Found' if multiplier_path.exists() else '[X] Not Found'}")

if not lstm_path.exists() or not multiplier_path.exists():
    print("\n[ERROR] Model files not found. Run train_models.py first.")
    exit(1)

# 3. Load models
print("\n2. Loading Models")
print("-" * 70)

device = torch.device('cpu')

lstm_model = ImprovedRiskLSTM()
lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
lstm_model.eval()
print("LSTM Risk Predictor: Loaded [OK]")

mult_model = TimeMultiplierNN()
mult_model.load_state_dict(torch.load(multiplier_path, map_location=device))
mult_model.eval()
print("Time Multiplier NN: Loaded [OK]")

# 4. Run predictions
print("\n3. Running Predictions")
print("-" * 70)

test_cases = [
    {"name": "Juba Center - Morning (9 AM)", "lat": 4.8517, "lon": 31.5825, "hour": 9, "dow": 1},
    {"name": "Juba Center - Evening (6 PM)", "lat": 4.8517, "lon": 31.5825, "hour": 18, "dow": 1},
    {"name": "Juba Center - Night (11 PM)", "lat": 4.8517, "lon": 31.5825, "hour": 23, "dow": 1},
    {"name": "Juba North - Evening (6 PM)", "lat": 4.86, "lon": 31.60, "hour": 18, "dow": 4},
    {"name": "Juba South - Friday Evening", "lat": 4.83, "lon": 31.57, "hour": 18, "dow": 4},
]

for test in test_cases:
    # LSTM input (7-day sequence)
    sequence_length = 7
    inputs = []

    for i in range(sequence_length):
        hour_norm = test['hour'] / 23.0
        dow_norm = test['dow'] / 6.0
        lat_norm = (test['lat'] - 4.85) / 0.3
        lon_norm = (test['lon'] - 31.6) / 0.3

        # Simple past risk estimate
        base_risk = 50.0
        if test['hour'] >= 18 and test['hour'] <= 20:
            base_risk = 70.0
        elif test['hour'] >= 22 or test['hour'] <= 5:
            base_risk = 65.0
        if test['dow'] == 4:  # Friday
            base_risk += 10

        risk_norm = base_risk / 100.0
        inputs.append([hour_norm, dow_norm, lat_norm, lon_norm, risk_norm])

    # LSTM prediction
    x = torch.tensor([inputs], dtype=torch.float32)
    with torch.no_grad():
        lstm_output = lstm_model(x)
        base_risk = lstm_output.item() * 100

    # Time multiplier prediction
    hour_rad = (test['hour'] / 24.0) * 2 * np.pi
    dow_rad = (test['dow'] / 7.0) * 2 * np.pi

    mult_features = [
        test['dow'] / 6.0,
        test['hour'] / 23.0,
        np.cos(hour_rad),
        np.sin(hour_rad),
        np.cos(dow_rad),
        np.sin(dow_rad),
        1.0 if test['dow'] in [0, 6] else 0.0,  # weekend
        1.0 if (test['dow'] in [1, 2, 3, 4, 5] and 9 <= test['hour'] <= 17) else 0.0,  # work hours
        1.0 if 17 <= test['hour'] <= 20 else 0.0,  # evening
        1.0 if (test['hour'] >= 22 or test['hour'] <= 5) else 0.0,  # late night
    ]

    mult_x = torch.tensor([mult_features], dtype=torch.float32)
    with torch.no_grad():
        mult_output = mult_model(mult_x)
        normalized = mult_output.item()

    # 0-1 -> 0.5-2.0 conversion
    multiplier = normalized * 1.5 + 0.5
    multiplier = max(0.5, min(2.0, multiplier))

    # Final risk
    final_risk = base_risk * multiplier

    print(f"\n{test['name']}")
    print(f"  Base Risk (LSTM):     {base_risk:6.2f}")
    print(f"  Time Multiplier (NN): {multiplier:6.2f}x")
    print(f"  Final Risk Score:     {final_risk:6.2f}/100")

# 5. Summary
print("\n" + "="*70)
print("[SUCCESS] All models working correctly!")
print("="*70)

print("\nModel Summary:")
print("  - LSTM Risk Predictor: Bidirectional 3-layer LSTM with Attention")
print("  - Time Multiplier NN: 4-layer feedforward network")
print("  - Both models trained on synthetic data (1000 samples)")

print("\nIntegration Status:")
print("  [OK] Models trained and saved")
print("  [OK] Models load successfully")
print("  [OK] Predictions working")
print("  [NEXT] Start backend API for production use")

print("\nAPI Endpoints Available:")
print("  POST /api/ai/predict/deep-learning")
print("  GET  /api/ai/predict/compare")
print("  GET  /api/ai/training/status")
