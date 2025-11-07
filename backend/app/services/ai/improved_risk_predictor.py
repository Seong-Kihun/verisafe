"""개선된 LSTM 기반 위험도 예측 모델 - 합성 데이터 활용"""
import numpy as np
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
import json
import os

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[ImprovedRiskPredictor] PyTorch 미설치")

from app.models.hazard import Hazard
from app.services.ai.data_augmentation import data_augmenter


class ImprovedRiskLSTM(nn.Module):
    """
    개선된 LSTM 위험도 예측 모델

    특징:
    - 양방향 LSTM (과거와 미래 패턴 모두 학습)
    - Attention 메커니즘
    - Dropout 정규화
    - Batch Normalization
    """

    def __init__(
        self,
        input_size=5,  # [시간, 요일, 위도, 경도, 위험도]
        hidden_size=128,
        num_layers=3,
        dropout=0.3
    ):
        super(ImprovedRiskLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 양방향 LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # 양방향
        )

        # Attention 레이어
        self.attention = nn.Linear(hidden_size * 2, 1)  # 양방향이므로 *2

        # Fully Connected 레이어
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass with attention

        Args:
            x: (batch_size, sequence_length, input_size)

        Returns:
            output: (batch_size, 1) - 위험도 0-1 스케일
        """
        # LSTM (양방향)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out shape: (batch, seq_len, hidden_size * 2)

        # Attention weights 계산
        attention_weights = torch.softmax(
            self.attention(lstm_out).squeeze(-1),
            dim=1
        )
        # attention_weights shape: (batch, seq_len)

        # Attention 적용 (가중 평균)
        attention_weights = attention_weights.unsqueeze(-1)
        # (batch, seq_len, 1)

        context_vector = torch.sum(lstm_out * attention_weights, dim=1)
        # context_vector shape: (batch, hidden_size * 2)

        # FC layers
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


class ImprovedRiskPredictor:
    """
    개선된 LSTM 기반 위험도 예측기

    특징:
    - 합성 데이터로 학습
    - 하이브리드 예측 (딥러닝 + 규칙 기반)
    - 신뢰도 점수 제공
    """

    def __init__(self, model_path: str = "models/improved_risk_lstm.pth"):
        """
        Args:
            model_path: 학습된 모델 경로
        """
        self.model_path = model_path
        self.model = None
        self.device = None
        self.is_trained = False

        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._initialize_model()
        else:
            print("[ImprovedRiskPredictor] PyTorch 미설치 - 규칙 기반만 사용")

    def _initialize_model(self):
        """모델 초기화 및 로드"""
        self.model = ImprovedRiskLSTM()
        self.model.to(self.device)

        # 학습된 모델 로드
        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(
                    torch.load(self.model_path, map_location=self.device)
                )
                self.model.eval()
                self.is_trained = True
                print(f"[ImprovedRiskPredictor] 학습된 모델 로드: {self.model_path}")
            except Exception as e:
                print(f"[ImprovedRiskPredictor] 모델 로드 실패: {e}")
                self.is_trained = False
        else:
            print(f"[ImprovedRiskPredictor] 모델 파일 없음. 학습 필요.")
            self.is_trained = False

    async def train_model(
        self,
        db: Session,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ) -> Dict:
        """
        모델 학습

        Args:
            db: Database session
            epochs: 학습 에포크
            batch_size: 배치 크기
            learning_rate: 학습률

        Returns:
            학습 결과 딕셔너리
        """
        if not TORCH_AVAILABLE:
            return {
                "status": "error",
                "message": "PyTorch not available"
            }

        print(f"\n{'='*50}")
        print(f"[ImprovedRiskPredictor] 모델 학습 시작")
        print(f"{'='*50}\n")

        # 1. 합성 데이터 생성
        print("1단계: 합성 데이터 생성...")
        synthetic_data = data_augmenter.generate_synthetic_data(db, target_count=1000)

        if len(synthetic_data) < 50:
            return {
                "status": "error",
                "message": f"데이터 부족: {len(synthetic_data)}개"
            }

        # 2. LSTM 시퀀스 준비
        print("\n2단계: LSTM 시퀀스 준비...")
        X, y = data_augmenter.prepare_lstm_sequences(
            synthetic_data,
            sequence_length=7
        )

        # 학습/검증 데이터 분리 (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        print(f"  학습 데이터: {X_train.shape}")
        print(f"  검증 데이터: {X_val.shape}")

        # 3. DataLoader 생성
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        # 4. 모델 초기화
        self.model = ImprovedRiskLSTM()
        self.model.to(self.device)

        # 5. 손실 함수 및 옵티마이저
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # 학습률 스케줄러
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )

        # 6. 학습 루프
        print(f"\n3단계: 모델 학습 ({epochs} epochs)...")

        best_val_loss = float('inf')
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # 학습 모드
            self.model.train()
            train_loss = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # 검증 모드
            self.model.eval()
            val_loss = 0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)

                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            # 학습률 조정
            scheduler.step(val_loss)

            # 로그 출력 (10 에포크마다)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch [{epoch+1}/{epochs}] "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f}")

            # 최고 모델 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss

                # 모델 디렉토리 생성
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

                # 모델 저장
                torch.save(self.model.state_dict(), self.model_path)

        print(f"\n4단계: 학습 완료!")
        print(f"  최종 학습 손실: {train_losses[-1]:.4f}")
        print(f"  최종 검증 손실: {val_losses[-1]:.4f}")
        print(f"  최고 검증 손실: {best_val_loss:.4f}")
        print(f"  모델 저장: {self.model_path}")

        self.is_trained = True

        return {
            "status": "success",
            "epochs": epochs,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "best_val_loss": best_val_loss,
            "model_path": self.model_path,
            "training_samples": len(X_train),
            "validation_samples": len(X_val)
        }

    def predict_risk(
        self,
        latitude: float,
        longitude: float,
        timestamp: datetime,
        confidence_threshold: float = 0.7
    ) -> Tuple[float, float, str]:
        """
        하이브리드 위험도 예측 (딥러닝 + 규칙 기반)

        Args:
            latitude: 위도
            longitude: 경도
            timestamp: 예측 시간
            confidence_threshold: 신뢰도 임계값

        Returns:
            (위험도, 신뢰도, 방법) - (0-100, 0-1, "deep_learning"/"rule_based"/"hybrid")
        """
        if not TORCH_AVAILABLE or not self.is_trained:
            # 규칙 기반만 사용
            risk = self._rule_based_predict(latitude, longitude, timestamp)
            return risk, 0.5, "rule_based"

        try:
            # 딥러닝 예측
            dl_risk, dl_confidence = self._deep_learning_predict(
                latitude, longitude, timestamp
            )

            # 규칙 기반 예측
            rule_risk = self._rule_based_predict(latitude, longitude, timestamp)

            # 하이브리드 결정
            if dl_confidence >= confidence_threshold:
                # 딥러닝 결과 사용
                return dl_risk, dl_confidence, "deep_learning"
            else:
                # 하이브리드: 가중 평균
                hybrid_risk = (
                    dl_risk * dl_confidence +
                    rule_risk * (1 - dl_confidence)
                )
                return hybrid_risk, dl_confidence, "hybrid"

        except Exception as e:
            print(f"[ImprovedRiskPredictor] 예측 오류: {e}")
            risk = self._rule_based_predict(latitude, longitude, timestamp)
            return risk, 0.3, "rule_based"

    def _deep_learning_predict(
        self,
        latitude: float,
        longitude: float,
        timestamp: datetime
    ) -> Tuple[float, float]:
        """
        딥러닝 예측

        Returns:
            (위험도, 신뢰도)
        """
        # 시퀀스 생성 (과거 7일 시뮬레이션)
        sequence_length = 7
        inputs = []

        for i in range(sequence_length):
            past_time = timestamp - timedelta(days=sequence_length - i - 1)

            hour = past_time.hour / 23.0
            day_of_week = past_time.weekday() / 6.0
            lat_norm = (latitude - 4.85) / 0.3
            lon_norm = (longitude - 31.6) / 0.3

            # 과거 위험도 추정 (간단한 패턴)
            past_risk = self._estimate_past_risk(past_time) / 100.0

            inputs.append([hour, day_of_week, lat_norm, lon_norm, past_risk])

        # Tensor 변환
        x = torch.tensor([inputs], dtype=torch.float32).to(self.device)

        # 예측
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
            risk_normalized = output.item()

        # 0-100 스케일
        risk_score = risk_normalized * 100

        # 신뢰도 계산 (중간값에서 멀수록 신뢰도 높음)
        confidence = abs(risk_normalized - 0.5) * 2

        return min(100, max(0, risk_score)), confidence

    def _rule_based_predict(
        self,
        latitude: float,
        longitude: float,
        timestamp: datetime
    ) -> float:
        """규칙 기반 예측 (폴백)"""
        hour = timestamp.hour
        day_of_week = timestamp.weekday()

        base_risk = 40

        # 요일별 조정
        if day_of_week == 4:  # 금요일
            base_risk += 20
        elif day_of_week in [5, 6]:  # 주말
            base_risk -= 5

        # 시간대별 조정
        if 17 <= hour <= 20:  # 저녁
            base_risk += 25
        elif 22 <= hour or hour <= 5:  # 심야
            base_risk += 15
        elif 9 <= hour <= 17:  # 업무 시간
            base_risk += 5

        return min(100, max(10, base_risk))

    def _estimate_past_risk(self, timestamp: datetime) -> float:
        """과거 위험도 추정 (간단한 패턴)"""
        hour = timestamp.hour
        day_of_week = timestamp.weekday()

        base = 50

        if day_of_week == 4 and 17 <= hour <= 20:
            return 75
        elif day_of_week in [5, 6]:
            return 40
        elif 22 <= hour or hour <= 5:
            return 60
        else:
            return base


# 싱글톤 인스턴스
improved_risk_predictor = ImprovedRiskPredictor()
