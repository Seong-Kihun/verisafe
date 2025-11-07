"""합성 데이터 생성기 - 딥러닝 학습용 데이터 증강"""
import random
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from sqlalchemy.orm import Session

from app.models.hazard import Hazard


class DataAugmenter:
    """
    실제 위험 데이터를 기반으로 학습용 합성 데이터 생성

    전략:
    1. 시간 변형: 같은 위치, 다른 시간대
    2. 위치 변형: 근처 좌표 (±0.01도 ~ ±0.05도)
    3. 위험도 변형: ±10% 노이즈
    4. 패턴 기반 생성: 요일/시간별 위험도 패턴 학습
    """

    def __init__(self):
        """데이터 증강기 초기화"""
        self.hazard_patterns = {
            'conflict': {
                'base_risk': 75,
                'high_risk_hours': [17, 18, 19, 20],  # 저녁
                'high_risk_days': [4, 5],  # 금요일, 토요일
                'radius_range': (2.0, 10.0)
            },
            'protest': {
                'base_risk': 60,
                'high_risk_hours': [10, 11, 12, 13, 14],  # 낮
                'high_risk_days': [0, 1, 2, 3, 4],  # 평일
                'radius_range': (1.0, 5.0)
            },
            'flood': {
                'base_risk': 65,
                'high_risk_hours': [0, 1, 2, 3, 4, 5],  # 새벽 (비 올 때)
                'high_risk_days': [0, 1, 2, 3, 4, 5, 6],  # 모든 요일
                'radius_range': (5.0, 20.0)
            },
            'natural_disaster': {
                'base_risk': 70,
                'high_risk_hours': list(range(24)),
                'high_risk_days': [0, 1, 2, 3, 4, 5, 6],
                'radius_range': (10.0, 50.0)
            },
            'landslide': {
                'base_risk': 68,
                'high_risk_hours': [0, 1, 2, 3, 4, 5],  # 새벽
                'high_risk_days': [0, 1, 2, 3, 4, 5, 6],
                'radius_range': (3.0, 15.0)
            },
            'other': {
                'base_risk': 50,
                'high_risk_hours': list(range(24)),
                'high_risk_days': [0, 1, 2, 3, 4, 5, 6],
                'radius_range': (1.0, 10.0)
            }
        }

        # 주바(남수단) 좌표 범위
        self.lat_range = (4.6, 5.1)  # 위도
        self.lon_range = (31.4, 31.8)  # 경도

    def generate_synthetic_data(
        self,
        db: Session,
        target_count: int = 1000
    ) -> List[Dict]:
        """
        실제 데이터를 기반으로 합성 데이터 생성

        Args:
            db: Database session
            target_count: 생성할 데이터 개수

        Returns:
            합성 데이터 리스트
        """
        print(f"[DataAugmenter] {target_count}개 합성 데이터 생성 시작...")

        # 실제 데이터 로드
        real_hazards = db.query(Hazard).all()

        if not real_hazards:
            print("[DataAugmenter] 실제 데이터 없음. 순수 패턴 기반 생성")
            return self._generate_pattern_based(target_count)

        print(f"[DataAugmenter] 실제 데이터 {len(real_hazards)}개 발견")

        synthetic_data = []

        # 실제 데이터 기반 증강 (70%)
        augment_count = int(target_count * 0.7)
        for i in range(augment_count):
            # 랜덤하게 실제 데이터 선택
            base_hazard = random.choice(real_hazards)

            # 증강된 데이터 생성
            augmented = self._augment_single_hazard(base_hazard, i)
            synthetic_data.append(augmented)

        # 패턴 기반 생성 (30%)
        pattern_count = target_count - augment_count
        pattern_data = self._generate_pattern_based(pattern_count)
        synthetic_data.extend(pattern_data)

        print(f"[DataAugmenter] 합성 데이터 {len(synthetic_data)}개 생성 완료")

        return synthetic_data

    def _augment_single_hazard(self, hazard: Hazard, index: int) -> Dict:
        """
        단일 위험 데이터를 증강

        전략:
        1. 시간 변형: ±1~30일
        2. 위치 변형: ±0.01~0.05도 (1~5km)
        3. 위험도 변형: ±5~15점
        4. 시간대별 승수 적용
        """
        # 시간 변형
        time_delta_days = random.randint(-30, 30)
        time_delta_hours = random.randint(0, 23)

        new_start_date = hazard.start_date + timedelta(
            days=time_delta_days,
            hours=time_delta_hours
        )

        # 위치 변형 (±0.01~0.05도)
        lat_noise = random.uniform(-0.05, 0.05)
        lon_noise = random.uniform(-0.05, 0.05)

        new_lat = hazard.latitude + lat_noise
        new_lon = hazard.longitude + lon_noise

        # 범위 체크
        new_lat = max(self.lat_range[0], min(self.lat_range[1], new_lat))
        new_lon = max(self.lon_range[0], min(self.lon_range[1], new_lon))

        # 위험도 변형
        risk_noise = random.randint(-15, 15)
        base_risk = hazard.risk_score + risk_noise

        # 시간대별 승수 적용
        hour = new_start_date.hour
        day_of_week = new_start_date.weekday()

        pattern = self.hazard_patterns.get(
            hazard.hazard_type,
            self.hazard_patterns['other']
        )

        time_multiplier = 1.0
        if hour in pattern['high_risk_hours']:
            time_multiplier *= 1.3
        if day_of_week in pattern['high_risk_days']:
            time_multiplier *= 1.2

        new_risk = int(base_risk * time_multiplier)
        new_risk = max(10, min(100, new_risk))

        # 반경 변형
        radius_range = pattern['radius_range']
        new_radius = random.uniform(radius_range[0], radius_range[1])

        return {
            'hazard_type': hazard.hazard_type,
            'latitude': round(new_lat, 6),
            'longitude': round(new_lon, 6),
            'risk_score': new_risk,
            'start_date': new_start_date,
            'radius': round(new_radius, 1),
            'description': f"Synthetic data {index} based on real hazard",
            'verified': True,
            'source': 'synthetic',
            'is_synthetic': True
        }

    def _generate_pattern_based(self, count: int) -> List[Dict]:
        """
        패턴 기반 순수 합성 데이터 생성

        주바 지역의 일반적인 위험 패턴 사용
        """
        synthetic_data = []

        for i in range(count):
            # 랜덤 유형 선택 (실제 분포 고려)
            hazard_type = random.choices(
                ['conflict', 'protest', 'flood', 'natural_disaster', 'landslide', 'other'],
                weights=[30, 20, 15, 20, 10, 5],  # 가중치
                k=1
            )[0]

            pattern = self.hazard_patterns[hazard_type]

            # 랜덤 시간 생성 (최근 60일 내)
            days_ago = random.randint(0, 60)
            hour = random.choice(pattern['high_risk_hours'])

            start_date = datetime.utcnow() - timedelta(days=days_ago, hours=hour)

            # 랜덤 위치 (주바 지역)
            latitude = random.uniform(self.lat_range[0], self.lat_range[1])
            longitude = random.uniform(self.lon_range[0], self.lon_range[1])

            # 위험도 계산
            base_risk = pattern['base_risk']
            risk_noise = random.randint(-10, 10)
            risk_score = max(20, min(95, base_risk + risk_noise))

            # 반경
            radius = random.uniform(pattern['radius_range'][0], pattern['radius_range'][1])

            synthetic_data.append({
                'hazard_type': hazard_type,
                'latitude': round(latitude, 6),
                'longitude': round(longitude, 6),
                'risk_score': risk_score,
                'start_date': start_date,
                'radius': round(radius, 1),
                'description': f"Pattern-based synthetic hazard {i}",
                'verified': True,
                'source': 'synthetic_pattern',
                'is_synthetic': True
            })

        return synthetic_data

    def prepare_lstm_sequences(
        self,
        data: List[Dict],
        sequence_length: int = 7
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        LSTM 학습용 시퀀스 데이터 준비

        Args:
            data: 합성 데이터 리스트
            sequence_length: 시퀀스 길이 (기본 7일)

        Returns:
            (X, y) - 입력 시퀀스와 타겟
        """
        # 시간순 정렬
        sorted_data = sorted(data, key=lambda x: x['start_date'])

        # 특징 정규화
        sequences = []
        targets = []

        for i in range(len(sorted_data) - sequence_length):
            # 시퀀스 생성 (과거 7일)
            seq = []
            for j in range(sequence_length):
                hazard = sorted_data[i + j]

                # 특징 벡터: [시간, 요일, 위도, 경도, 위험도]
                hour = hazard['start_date'].hour / 23.0  # 0-1 정규화
                day_of_week = hazard['start_date'].weekday() / 6.0
                lat = (hazard['latitude'] - 4.85) / 0.3  # 주바 중심 정규화
                lon = (hazard['longitude'] - 31.6) / 0.3
                risk = hazard['risk_score'] / 100.0

                seq.append([hour, day_of_week, lat, lon, risk])

            sequences.append(seq)

            # 타겟: 다음 시점의 위험도
            target_risk = sorted_data[i + sequence_length]['risk_score'] / 100.0
            targets.append([target_risk])

        X = np.array(sequences, dtype=np.float32)
        y = np.array(targets, dtype=np.float32)

        print(f"[DataAugmenter] LSTM 시퀀스 생성 완료: X={X.shape}, y={y.shape}")

        return X, y


# 싱글톤 인스턴스
data_augmenter = DataAugmenter()
