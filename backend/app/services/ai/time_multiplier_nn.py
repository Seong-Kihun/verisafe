"""시간대별 위험도 승수 예측 Neural Network"""
import numpy as np
from typing import Dict, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
import os

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[TimeMultiplierNN] PyTorch 미설치")

from app.services.ai.time_multiplier import TimeMultiplierCalculator


class TimeMultiplierNetwork(nn.Module):
    """
    시간대별 승수 예측 신경망

    입력: [요일(0-6), 시간(0-23), 최근 7일 평균 위험도]
    출력: 승수 (0.5-2.0)
    """

    def __init__(self, input_size=10):
        super(TimeMultiplierNetwork, self).__init__()

        # 간단한 Feedforward Network
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
        """
        Forward pass

        Args:
            x: (batch_size, input_size)

        Returns:
            output: (batch_size, 1) - 승수 0-1 스케일 (실제 0.5-2.0으로 변환 필요)
        """
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
        out = self.sigmoid(out)  # 0-1 스케일

        return out


class TimeMultiplierPredictor:
    """
    딥러닝 기반 시간대별 승수 예측기

    특징:
    - 요일, 시간, 최근 패턴을 학습
    - 하이브리드 (딥러닝 + 통계 기반)
    """

    def __init__(self, model_path: str = "models/time_multiplier_nn.pth"):
        """
        Args:
            model_path: 학습된 모델 경로
        """
        self.model_path = model_path
        self.model = None
        self.device = None
        self.is_trained = False

        # 통계 기반 계산기 (폴백용)
        self.statistical_calculator = TimeMultiplierCalculator()

        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._initialize_model()
        else:
            print("[TimeMultiplierPredictor] PyTorch 미설치 - 통계 기반만 사용")

    def _initialize_model(self):
        """모델 초기화 및 로드"""
        self.model = TimeMultiplierNetwork()
        self.model.to(self.device)

        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(
                    torch.load(self.model_path, map_location=self.device)
                )
                self.model.eval()
                self.is_trained = True
                print(f"[TimeMultiplierPredictor] 모델 로드: {self.model_path}")
            except Exception as e:
                print(f"[TimeMultiplierPredictor] 모델 로드 실패: {e}")
                self.is_trained = False
        else:
            print(f"[TimeMultiplierPredictor] 모델 없음. 학습 필요.")
            self.is_trained = False

    async def train_model(
        self,
        db: Session,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.01
    ) -> Dict:
        """
        모델 학습

        Args:
            db: Database session
            epochs: 학습 에포크
            batch_size: 배치 크기
            learning_rate: 학습률

        Returns:
            학습 결과
        """
        if not TORCH_AVAILABLE:
            return {
                "status": "error",
                "message": "PyTorch not available"
            }

        print(f"\n{'='*50}")
        print(f"[TimeMultiplierPredictor] 시간대별 승수 모델 학습")
        print(f"{'='*50}\n")

        # 1. 통계 기반 승수 계산
        print("1단계: 통계 기반 승수 계산...")
        await self.statistical_calculator.calculate_and_cache(db, days_back=30)

        # 2. 학습 데이터 생성
        print("\n2단계: 학습 데이터 생성...")
        X, y = self._prepare_training_data()

        print(f"  학습 데이터: {X.shape}")

        # 학습/검증 분리
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # DataLoader
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 3. 모델 초기화
        self.model = TimeMultiplierNetwork()
        self.model.to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # 4. 학습 루프
        print(f"\n3단계: 모델 학습 ({epochs} epochs)...")

        best_val_loss = float('inf')

        for epoch in range(epochs):
            # 학습
            self.model.train()
            train_loss = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # 검증
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

            # 로그
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch [{epoch+1}/{epochs}] "
                      f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")

            # 최고 모델 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                torch.save(self.model.state_dict(), self.model_path)

        print(f"\n4단계: 학습 완료!")
        print(f"  최고 검증 손실: {best_val_loss:.4f}")
        print(f"  모델 저장: {self.model_path}")

        self.is_trained = True

        return {
            "status": "success",
            "epochs": epochs,
            "best_val_loss": best_val_loss,
            "model_path": self.model_path
        }

    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        학습 데이터 준비

        통계 기반 승수를 타겟으로 사용
        """
        X = []
        y = []

        # 모든 요일/시간 조합에 대해
        for dow in range(7):  # 0-6 (일-토)
            for hour in range(24):  # 0-23
                # 통계 기반 승수 (타겟)
                target_multiplier = self.statistical_calculator.multipliers.get(
                    (dow, hour),
                    1.0
                )

                # 특징 벡터 생성
                # [요일 원-핫 (7), 시간 원-핫 (24), 최근 평균 (1)] = 총 10차원 아니고...
                # 더 간단하게: [요일, 시간, 코사인(시간), 사인(시간), 주말여부, 업무시간여부 등]
                features = self._create_features(dow, hour)

                X.append(features)
                y.append([(target_multiplier - 0.5) / 1.5])  # 0.5-2.0 -> 0-1 정규화

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        return X, y

    def _create_features(self, dow: int, hour: int) -> list:
        """
        특징 벡터 생성

        Args:
            dow: 요일 (0-6)
            hour: 시간 (0-23)

        Returns:
            특징 벡터 [10차원]
        """
        features = []

        # 1. 요일 정규화 (0-1)
        features.append(dow / 6.0)

        # 2. 시간 정규화 (0-1)
        features.append(hour / 23.0)

        # 3. 시간 주기성 (코사인, 사인)
        hour_rad = (hour / 24.0) * 2 * np.pi
        features.append(np.cos(hour_rad))
        features.append(np.sin(hour_rad))

        # 4. 요일 주기성
        dow_rad = (dow / 7.0) * 2 * np.pi
        features.append(np.cos(dow_rad))
        features.append(np.sin(dow_rad))

        # 5. 주말 여부 (0 or 1)
        features.append(1.0 if dow in [0, 6] else 0.0)

        # 6. 업무 시간 여부
        features.append(1.0 if (dow in [1, 2, 3, 4, 5] and 9 <= hour <= 17) else 0.0)

        # 7. 저녁 시간 여부
        features.append(1.0 if 17 <= hour <= 20 else 0.0)

        # 8. 심야 여부
        features.append(1.0 if (hour >= 22 or hour <= 5) else 0.0)

        return features

    def get_multiplier(
        self,
        timestamp: datetime,
        use_hybrid: bool = True
    ) -> Tuple[float, str]:
        """
        시간대별 승수 예측

        Args:
            timestamp: 시간
            use_hybrid: 하이브리드 사용 여부

        Returns:
            (승수, 방법) - (0.5-2.0, "deep_learning"/"statistical"/"hybrid")
        """
        dow = (timestamp.weekday() + 1) % 7  # Python -> PostgreSQL dow
        hour = timestamp.hour

        # 통계 기반 승수
        stat_multiplier = self.statistical_calculator.multipliers.get(
            (dow, hour),
            1.0
        )

        if not TORCH_AVAILABLE or not self.is_trained:
            return stat_multiplier, "statistical"

        try:
            # 딥러닝 예측
            features = self._create_features(dow, hour)
            x = torch.tensor([features], dtype=torch.float32).to(self.device)

            self.model.eval()
            with torch.no_grad():
                output = self.model(x)
                normalized = output.item()

            # 0-1 -> 0.5-2.0 변환
            dl_multiplier = normalized * 1.5 + 0.5
            dl_multiplier = max(0.5, min(2.0, dl_multiplier))

            if use_hybrid:
                # 하이브리드: 평균
                hybrid_multiplier = (dl_multiplier + stat_multiplier) / 2.0
                return round(hybrid_multiplier, 2), "hybrid"
            else:
                return round(dl_multiplier, 2), "deep_learning"

        except Exception as e:
            print(f"[TimeMultiplierPredictor] 예측 오류: {e}")
            return stat_multiplier, "statistical"

    def apply_to_risk_score(
        self,
        base_risk: float,
        timestamp: datetime
    ) -> Tuple[float, str]:
        """
        기본 위험도에 승수 적용

        Args:
            base_risk: 기본 위험도 (0-100)
            timestamp: 시간

        Returns:
            (조정된 위험도, 방법)
        """
        multiplier, method = self.get_multiplier(timestamp)
        adjusted_risk = base_risk * multiplier

        return min(100, max(0, adjusted_risk)), method


# 싱글톤 인스턴스
time_multiplier_predictor = TimeMultiplierPredictor()
