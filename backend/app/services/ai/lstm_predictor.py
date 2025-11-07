"""LSTM 시계열 예측 모델 - 위험 발생 예측"""
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models.hazard import Hazard


class LSTMPredictor:
    """
    LSTM 기반 위험 발생 예측기

    기능:
    - 과거 데이터 패턴 학습
    - 1-7일 후 위험 발생 예측
    - 이상 탐지 (Anomaly Detection)
    - 위험 핫스팟 예측

    Phase 3 구현

    참고: 실제 LSTM 모델 학습은 별도의 학습 스크립트에서 수행됩니다.
    여기서는 통계적 방법과 패턴 인식을 사용합니다.
    프로덕션 환경에서는 사전 학습된 모델을 로드하여 사용합니다.
    """

    def __init__(self):
        """
        LSTM 예측기 초기화
        """
        self.model_loaded = False
        self.model = None

        # 실제 LSTM 모델 로드 시도 (선택적)
        try:
            # TODO: 사전 학습된 모델 로드
            # self.model = load_model("lstm_hazard_predictor.h5")
            # self.model_loaded = True
            print("[LSTMPredictor] 통계 기반 예측 모드로 초기화")
        except Exception as e:
            print(f"[LSTMPredictor] 모델 로드 실패: {e}")

    def predict_future_hazards(
        self,
        db: Session,
        days_ahead: int = 7,
        location_radius_km: float = 50.0
    ) -> List[Dict]:
        """
        향후 N일 동안 예상되는 위험 예측

        Args:
            db: Database session
            days_ahead: 예측할 일수 (1-7일)
            location_radius_km: 예측 범위 (km)

        Returns:
            예측된 위험 리스트
        """
        print(f"[LSTMPredictor] 향후 {days_ahead}일 위험 예측 중...")

        # 과거 데이터 수집 (최근 30일)
        historical_data = self._collect_historical_data(db, days=30)

        if not historical_data:
            print("[LSTMPredictor] 과거 데이터 부족")
            return []

        # 패턴 분석
        patterns = self._analyze_patterns(historical_data)

        # 예측 생성
        predictions = self._generate_predictions(patterns, days_ahead)

        print(f"[LSTMPredictor] {len(predictions)}개 위험 예측 완료")
        return predictions

    def detect_anomalies(
        self,
        db: Session,
        hours: int = 24
    ) -> List[Dict]:
        """
        이상 징후 감지

        최근 데이터에서 비정상적인 패턴 탐지

        Args:
            db: Database session
            hours: 분석할 시간 범위

        Returns:
            감지된 이상 징후 리스트
        """
        print(f"[LSTMPredictor] 최근 {hours}시간 이상 탐지 중...")

        # 최근 데이터
        recent_data = self._collect_recent_data(db, hours=hours)

        # 기준선 데이터 (최근 30일 평균)
        baseline_data = self._collect_historical_data(db, days=30)

        # 이상 감지
        anomalies = self._detect_anomalies(recent_data, baseline_data)

        print(f"[LSTMPredictor] {len(anomalies)}개 이상 징후 감지")
        return anomalies

    def predict_hotspots(
        self,
        db: Session,
        grid_size_km: float = 10.0
    ) -> List[Dict]:
        """
        위험 핫스팟 예측

        지리적으로 위험이 집중될 것으로 예상되는 지역

        Args:
            db: Database session
            grid_size_km: 그리드 크기 (km)

        Returns:
            예측된 핫스팟 리스트
        """
        print("[LSTMPredictor] 위험 핫스팟 예측 중...")

        # 과거 데이터 수집
        historical_data = self._collect_historical_data(db, days=30)

        if not historical_data:
            return []

        # 지리적 그리드 생성 및 분석
        hotspots = self._identify_hotspots(historical_data, grid_size_km)

        print(f"[LSTMPredictor] {len(hotspots)}개 핫스팟 예측")
        return hotspots

    def _collect_historical_data(self, db: Session, days: int) -> List[Dict]:
        """
        과거 데이터 수집

        Returns:
            Hazard 데이터 리스트
        """
        since_date = datetime.utcnow() - timedelta(days=days)

        hazards = db.query(Hazard).filter(
            Hazard.created_at >= since_date
        ).all()

        return [
            {
                "id": h.id,
                "type": h.hazard_type,
                "risk_score": h.risk_score,
                "latitude": h.latitude,
                "longitude": h.longitude,
                "created_at": h.created_at,
                "source": h.source,
                "verified": h.verified
            }
            for h in hazards
        ]

    def _collect_recent_data(self, db: Session, hours: int) -> List[Dict]:
        """
        최근 데이터 수집
        """
        since_time = datetime.utcnow() - timedelta(hours=hours)

        hazards = db.query(Hazard).filter(
            Hazard.created_at >= since_time
        ).all()

        return [
            {
                "id": h.id,
                "type": h.hazard_type,
                "risk_score": h.risk_score,
                "latitude": h.latitude,
                "longitude": h.longitude,
                "created_at": h.created_at,
                "source": h.source
            }
            for h in hazards
        ]

    def _analyze_patterns(self, historical_data: List[Dict]) -> Dict:
        """
        과거 데이터에서 패턴 분석

        Returns:
            패턴 분석 결과
        """
        if not historical_data:
            return {}

        # 1. 위험 유형별 빈도
        type_frequency = defaultdict(int)
        type_risk_avg = defaultdict(list)

        for hazard in historical_data:
            h_type = hazard["type"]
            type_frequency[h_type] += 1
            type_risk_avg[h_type].append(hazard["risk_score"])

        # 2. 요일별 패턴
        weekday_frequency = defaultdict(int)
        for hazard in historical_data:
            weekday = hazard["created_at"].weekday()  # 0=Monday
            weekday_frequency[weekday] += 1

        # 3. 지리적 클러스터
        location_clusters = self._cluster_locations(historical_data)

        # 4. 시간대별 패턴
        hour_frequency = defaultdict(int)
        for hazard in historical_data:
            hour = hazard["created_at"].hour
            hour_frequency[hour] += 1

        # 5. 평균 위험도 계산
        avg_risk_by_type = {
            h_type: np.mean(risks)
            for h_type, risks in type_risk_avg.items()
        }

        return {
            "type_frequency": dict(type_frequency),
            "weekday_frequency": dict(weekday_frequency),
            "hour_frequency": dict(hour_frequency),
            "location_clusters": location_clusters,
            "avg_risk_by_type": avg_risk_by_type,
            "total_hazards": len(historical_data)
        }

    def _generate_predictions(
        self,
        patterns: Dict,
        days_ahead: int
    ) -> List[Dict]:
        """
        패턴 기반 예측 생성

        통계적 방법으로 미래 위험 예측
        """
        predictions = []

        type_freq = patterns.get("type_frequency", {})
        avg_risk = patterns.get("avg_risk_by_type", {})
        location_clusters = patterns.get("location_clusters", [])

        # 각 날짜별 예측
        for day in range(1, days_ahead + 1):
            future_date = datetime.utcnow() + timedelta(days=day)
            weekday = future_date.weekday()

            # 해당 요일의 과거 발생 빈도
            weekday_freq = patterns.get("weekday_frequency", {}).get(weekday, 0)

            # 발생 가능성이 높은 위험 유형
            for h_type, freq in sorted(type_freq.items(), key=lambda x: x[1], reverse=True):
                # 발생 확률 계산
                probability = min(0.9, freq / patterns.get("total_hazards", 1))

                # 확률이 30% 이상인 경우만 예측
                if probability >= 0.3:
                    # 예상 위치 (과거 클러스터 중심)
                    location = self._select_likely_location(location_clusters, h_type)

                    prediction = {
                        "predicted_date": future_date.isoformat(),
                        "days_ahead": day,
                        "hazard_type": h_type,
                        "predicted_risk_score": int(avg_risk.get(h_type, 50)),
                        "probability": round(probability, 2),
                        "latitude": location["lat"],
                        "longitude": location["lon"],
                        "confidence": "high" if probability > 0.7 else "medium",
                        "reasoning": f"Based on historical pattern: {freq} occurrences in past 30 days"
                    }

                    predictions.append(prediction)

        # 확률 높은 순으로 정렬, 최대 10개
        predictions.sort(key=lambda x: x["probability"], reverse=True)
        return predictions[:10]

    def _detect_anomalies(
        self,
        recent_data: List[Dict],
        baseline_data: List[Dict]
    ) -> List[Dict]:
        """
        이상 징후 감지

        최근 데이터가 과거 패턴과 크게 다른 경우 감지
        """
        anomalies = []

        if not baseline_data:
            return anomalies

        # 기준선 계산
        baseline_avg_risk = np.mean([h["risk_score"] for h in baseline_data])
        baseline_count_per_hour = len(baseline_data) / (30 * 24)  # 30일 평균

        # 최근 데이터 분석
        if recent_data:
            recent_avg_risk = np.mean([h["risk_score"] for h in recent_data])
            recent_count_per_hour = len(recent_data) / 24  # 24시간

            # 이상 1: 위험도 급증
            if recent_avg_risk > baseline_avg_risk * 1.5:
                anomalies.append({
                    "type": "risk_spike",
                    "severity": "high",
                    "description": f"Average risk score increased by {((recent_avg_risk / baseline_avg_risk - 1) * 100):.1f}%",
                    "current_value": round(recent_avg_risk, 1),
                    "baseline_value": round(baseline_avg_risk, 1),
                    "detected_at": datetime.utcnow().isoformat()
                })

            # 이상 2: 빈도 급증
            if recent_count_per_hour > baseline_count_per_hour * 2:
                anomalies.append({
                    "type": "frequency_spike",
                    "severity": "medium",
                    "description": f"Hazard frequency increased by {((recent_count_per_hour / baseline_count_per_hour - 1) * 100):.1f}%",
                    "current_value": round(recent_count_per_hour, 2),
                    "baseline_value": round(baseline_count_per_hour, 2),
                    "detected_at": datetime.utcnow().isoformat()
                })

            # 이상 3: 새로운 위험 유형
            baseline_types = set(h["type"] for h in baseline_data)
            recent_types = set(h["type"] for h in recent_data)
            new_types = recent_types - baseline_types

            if new_types:
                anomalies.append({
                    "type": "new_hazard_type",
                    "severity": "low",
                    "description": f"New hazard types detected: {', '.join(new_types)}",
                    "new_types": list(new_types),
                    "detected_at": datetime.utcnow().isoformat()
                })

        return anomalies

    def _identify_hotspots(
        self,
        historical_data: List[Dict],
        grid_size_km: float
    ) -> List[Dict]:
        """
        위험 핫스팟 식별

        지리적 그리드 기반 클러스터링
        """
        if not historical_data:
            return []

        # 지리적 그리드 생성
        # 1도 ≈ 111km
        grid_degree = grid_size_km / 111.0

        grid_counts = defaultdict(lambda: {"count": 0, "total_risk": 0, "hazards": []})

        for hazard in historical_data:
            lat = hazard["latitude"]
            lon = hazard["longitude"]

            # 그리드 좌표 계산
            grid_lat = int(lat / grid_degree) * grid_degree
            grid_lon = int(lon / grid_degree) * grid_degree

            grid_key = (grid_lat, grid_lon)
            grid_counts[grid_key]["count"] += 1
            grid_counts[grid_key]["total_risk"] += hazard["risk_score"]
            grid_counts[grid_key]["hazards"].append(hazard)

        # 핫스팟 추출 (발생 빈도 높은 순)
        hotspots = []
        for (grid_lat, grid_lon), data in grid_counts.items():
            if data["count"] >= 3:  # 최소 3건 이상
                avg_risk = data["total_risk"] / data["count"]

                hotspot = {
                    "latitude": grid_lat + grid_degree / 2,
                    "longitude": grid_lon + grid_degree / 2,
                    "grid_size_km": grid_size_km,
                    "hazard_count": data["count"],
                    "avg_risk_score": round(avg_risk, 1),
                    "confidence": "high" if data["count"] >= 5 else "medium",
                    "most_common_type": self._get_most_common_type(data["hazards"])
                }

                hotspots.append(hotspot)

        # 빈도 높은 순으로 정렬
        hotspots.sort(key=lambda x: x["hazard_count"], reverse=True)
        return hotspots[:10]

    def _cluster_locations(self, hazards: List[Dict]) -> List[Dict]:
        """
        위치 클러스터링 (간단한 그리드 기반)
        """
        if not hazards:
            return []

        # 타입별 평균 위치
        type_locations = defaultdict(lambda: {"lats": [], "lons": []})

        for hazard in hazards:
            h_type = hazard["type"]
            type_locations[h_type]["lats"].append(hazard["latitude"])
            type_locations[h_type]["lons"].append(hazard["longitude"])

        clusters = []
        for h_type, coords in type_locations.items():
            cluster = {
                "type": h_type,
                "lat": np.mean(coords["lats"]),
                "lon": np.mean(coords["lons"]),
                "count": len(coords["lats"])
            }
            clusters.append(cluster)

        return clusters

    def _select_likely_location(
        self,
        location_clusters: List[Dict],
        hazard_type: str
    ) -> Dict:
        """
        예측 위치 선택

        해당 위험 유형이 과거에 자주 발생한 위치
        """
        # 해당 타입의 클러스터 찾기
        for cluster in location_clusters:
            if cluster["type"] == hazard_type:
                return {"lat": cluster["lat"], "lon": cluster["lon"]}

        # 없으면 Juba 중심
        return {"lat": 4.8517, "lon": 31.5825}

    def _get_most_common_type(self, hazards: List[Dict]) -> str:
        """가장 빈번한 위험 유형"""
        type_counts = defaultdict(int)
        for h in hazards:
            type_counts[h["type"]] += 1

        if type_counts:
            return max(type_counts.items(), key=lambda x: x[1])[0]
        return "other"
