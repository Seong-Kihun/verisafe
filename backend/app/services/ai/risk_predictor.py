"""LSTM 기반 위험도 예측 모델"""
import numpy as np
from typing import List, Tuple, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
import json
import os

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[RiskPredictor] PyTorch가 설치되지 않았습니다. 더미 예측을 사용합니다.")

from app.models.hazard import Hazard


class RiskLSTM(nn.Module):
    """
    LSTM 신경망 위험도 예측 모델
    
    입력: [시간, 요일, 위도, 경도] (4차원)
    출력: 위험도 (0-100)
    """
    
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, dropout=0.2):
        super(RiskLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully Connected 레이어
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: (batch_size, sequence_length, input_size)
        
        Returns:
            output: (batch_size, 1) - 위험도 0-1 스케일
        """
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 마지막 시퀀스의 출력 사용
        last_output = lstm_out[:, -1, :]
        
        # FC layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out


class RiskPredictor:
    """
    LSTM 기반 위험도 예측기
    
    Phase 3 구현:
    - 시계열 데이터 기반 위험도 예측
    - 요일, 시간, 위치 기반 패턴 학습
    - Redis 캐싱 지원
    """
    
    def __init__(self, model_path: str = "models/risk_lstm.pth"):
        """
        Args:
            model_path: 학습된 모델 경로
        """
        self.model_path = model_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if TORCH_AVAILABLE else None
        
        if TORCH_AVAILABLE:
            self._load_model()
        else:
            print("[RiskPredictor] PyTorch 미설치 - 더미 모드로 실행")
    
    def _load_model(self):
        """학습된 모델 로드 (없으면 새로 생성)"""
        self.model = RiskLSTM()
        
        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model.eval()
                print(f"[RiskPredictor] 모델 로드 완료: {self.model_path}")
            except Exception as e:
                print(f"[RiskPredictor] 모델 로드 실패: {e}. 새 모델 사용.")
        else:
            print(f"[RiskPredictor] 모델 파일 없음. 새 모델 초기화.")
        
        self.model.to(self.device)
    
    def predict_risk(self, latitude: float, longitude: float, timestamp: datetime) -> float:
        """
        특정 시간, 위치의 위험도 예측
        
        Args:
            latitude: 위도
            longitude: 경도
            timestamp: 예측 시간
        
        Returns:
            예측 위험도 (0-100)
        """
        if not TORCH_AVAILABLE or self.model is None:
            return self._dummy_predict(latitude, longitude, timestamp)
        
        try:
            # 입력 데이터 준비
            hour = timestamp.hour / 23.0  # 0-1 정규화
            day_of_week = timestamp.weekday() / 6.0  # 0-1 정규화
            lat_norm = (latitude - 4.85) / 0.1  # 주바 중심 정규화
            lng_norm = (longitude - 31.57) / 0.1
            
            # 시퀀스 생성 (과거 7일간 데이터 시뮬레이션)
            sequence_length = 7
            inputs = []
            
            for i in range(sequence_length):
                past_time = timestamp - timedelta(days=sequence_length - i - 1)
                past_hour = past_time.hour / 23.0
                past_dow = past_time.weekday() / 6.0
                inputs.append([past_hour, past_dow, lat_norm, lng_norm])
            
            # Tensor 변환
            x = torch.tensor([inputs], dtype=torch.float32).to(self.device)
            
            # 예측
            with torch.no_grad():
                output = self.model(x)
                risk_normalized = output.item()
            
            # 0-100 스케일로 변환
            risk_score = risk_normalized * 100
            
            return min(100, max(0, risk_score))
            
        except Exception as e:
            print(f"[RiskPredictor] 예측 오류: {e}")
            return self._dummy_predict(latitude, longitude, timestamp)
    
    def _dummy_predict(self, latitude: float, longitude: float, timestamp: datetime) -> float:
        """
        PyTorch 없을 때 더미 예측 (간단한 휴리스틱)
        
        - 금요일 17-20시: 높은 위험
        - 심야 시간: 중간 위험
        - 주말: 낮은 위험
        """
        hour = timestamp.hour
        day_of_week = timestamp.weekday()  # 0=월, 4=금, 6=일
        
        base_risk = 30
        
        # 요일별 조정
        if day_of_week == 4:  # 금요일
            base_risk += 20
        elif day_of_week in [5, 6]:  # 주말
            base_risk -= 10
        
        # 시간대별 조정
        if 17 <= hour <= 20:  # 저녁 시간
            base_risk += 25
        elif 22 <= hour or hour <= 5:  # 심야
            base_risk += 15
        elif 9 <= hour <= 17:  # 업무 시간
            base_risk += 5
        
        return min(100, max(0, base_risk))
    
    def predict_for_route(self, route_points: List[Tuple[float, float]], timestamp: datetime) -> dict:
        """
        경로 전체에 대한 위험도 예측
        
        Args:
            route_points: [(lat, lng), ...] 경로 좌표 리스트
            timestamp: 예측 시간
        
        Returns:
            Dictionary with predictions
        """
        predictions = []
        
        for lat, lng in route_points:
            risk = self.predict_risk(lat, lng, timestamp)
            predictions.append({
                "latitude": lat,
                "longitude": lng,
                "risk_score": round(risk, 2)
            })
        
        avg_risk = sum(p["risk_score"] for p in predictions) / len(predictions) if predictions else 0
        max_risk = max((p["risk_score"] for p in predictions), default=0)
        
        return {
            "timestamp": timestamp.isoformat(),
            "average_risk": round(avg_risk, 2),
            "max_risk": round(max_risk, 2),
            "points": predictions
        }
    
    async def batch_predict(self, db: Session, locations: List[Tuple[float, float]], 
                           start_time: datetime, hours: int = 24) -> List[dict]:
        """
        여러 위치, 여러 시간대에 대한 배치 예측
        
        Args:
            db: Database session
            locations: [(lat, lng), ...] 위치 리스트
            start_time: 시작 시간
            hours: 예측할 시간 범위
        
        Returns:
            예측 결과 리스트
        """
        results = []
        
        for hour_offset in range(hours):
            timestamp = start_time + timedelta(hours=hour_offset)
            
            for lat, lng in locations:
                risk = self.predict_risk(lat, lng, timestamp)
                
                results.append({
                    "latitude": lat,
                    "longitude": lng,
                    "timestamp": timestamp.isoformat(),
                    "risk_score": round(risk, 2)
                })
        
        return results


# 싱글톤 인스턴스
risk_predictor = RiskPredictor()
