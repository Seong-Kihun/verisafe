"""AI 예측 및 분석 API"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime

from app.database import get_db
from app.services.ai.nlp_analyzer import NLPAnalyzer
from app.services.ai.lstm_predictor import LSTMPredictor
from app.services.ai.trust_scorer import TrustScorer
from app.services.ai.improved_risk_predictor import improved_risk_predictor
from app.services.ai.time_multiplier_nn import time_multiplier_predictor
from app.models.hazard import Hazard
from app.models.user import User

router = APIRouter()

# AI 서비스 인스턴스 (싱글톤)
nlp_analyzer = NLPAnalyzer()
lstm_predictor = LSTMPredictor()
trust_scorer = TrustScorer()


@router.post("/analyze-text")
async def analyze_text_endpoint(
    text: str,
    db: Session = Depends(get_db)
):
    """
    텍스트 NLP 분석

    Args:
        text: 분석할 텍스트

    Returns:
        감정, 위험도, 키워드 등 분석 결과
    """
    try:
        analysis = nlp_analyzer.analyze_text(text)
        return {
            "status": "success",
            "analysis": analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 오류: {str(e)}")


@router.get("/predict-future")
async def predict_future_hazards(
    days_ahead: int = Query(default=7, ge=1, le=14),
    db: Session = Depends(get_db)
):
    """
    향후 위험 예측

    Args:
        days_ahead: 예측할 일수 (1-14일)

    Returns:
        예측된 위험 리스트
    """
    try:
        predictions = lstm_predictor.predict_future_hazards(db, days_ahead=days_ahead)
        return {
            "status": "success",
            "predictions": predictions,
            "days_ahead": days_ahead,
            "count": len(predictions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 오류: {str(e)}")


@router.get("/detect-anomalies")
async def detect_anomalies_endpoint(
    hours: int = Query(default=24, ge=1, le=168),
    db: Session = Depends(get_db)
):
    """
    이상 징후 감지

    Args:
        hours: 분석할 시간 범위 (1-168시간)

    Returns:
        감지된 이상 징후 리스트
    """
    try:
        anomalies = lstm_predictor.detect_anomalies(db, hours=hours)
        return {
            "status": "success",
            "anomalies": anomalies,
            "hours_analyzed": hours,
            "count": len(anomalies)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이상 탐지 오류: {str(e)}")


@router.get("/predict-hotspots")
async def predict_hotspots_endpoint(
    grid_size_km: float = Query(default=10.0, ge=1.0, le=50.0),
    db: Session = Depends(get_db)
):
    """
    위험 핫스팟 예측

    Args:
        grid_size_km: 그리드 크기 (km, 1-50)

    Returns:
        예측된 핫스팟 리스트
    """
    try:
        hotspots = lstm_predictor.predict_hotspots(db, grid_size_km=grid_size_km)
        return {
            "status": "success",
            "hotspots": hotspots,
            "grid_size_km": grid_size_km,
            "count": len(hotspots)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"핫스팟 예측 오류: {str(e)}")


@router.get("/hazard/{hazard_id}/trust-score")
async def get_hazard_trust_score(
    hazard_id: str,
    db: Session = Depends(get_db)
):
    """
    특정 위험 정보의 신뢰도 점수 조회

    Args:
        hazard_id: Hazard ID

    Returns:
        신뢰도 점수 및 상세 분석
    """
    try:
        hazard = db.query(Hazard).filter(Hazard.id == hazard_id).first()

        if not hazard:
            raise HTTPException(status_code=404, detail="Hazard not found")

        # Hazard를 딕셔너리로 변환
        hazard_dict = {
            "hazard_type": hazard.hazard_type,
            "risk_score": hazard.risk_score,
            "latitude": hazard.latitude,
            "longitude": hazard.longitude,
            "description": hazard.description,
            "start_date": hazard.start_date,
            "end_date": hazard.end_date,
            "radius": hazard.radius
        }

        # 사용자 조회 (있는 경우)
        user = None
        if hasattr(hazard, 'reported_by') and hazard.reported_by:
            user = db.query(User).filter(User.id == hazard.reported_by).first()

        # 신뢰도 상세 분석
        trust_breakdown = trust_scorer.get_trust_breakdown(hazard_dict, user, db)

        return {
            "status": "success",
            "hazard_id": hazard_id,
            "trust_analysis": trust_breakdown
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"신뢰도 계산 오류: {str(e)}")


@router.post("/hazard/validate")
async def validate_hazard_report(
    hazard_data: dict,
    user_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    위험 제보 검증

    사용자가 새로운 위험을 제보하기 전에 신뢰도 검증

    Args:
        hazard_data: Hazard 데이터 딕셔너리
        user_id: 제보자 사용자 ID (선택적)

    Returns:
        신뢰도 점수 및 스팸 여부
    """
    try:
        # 사용자 조회
        user = None
        if user_id:
            user = db.query(User).filter(User.id == user_id).first()

        # 신뢰도 점수 계산
        trust_score = trust_scorer.calculate_trust_score(hazard_data, user, db)

        # 스팸 여부
        is_spam = trust_scorer.is_likely_spam(hazard_data, user, db)

        # 상세 분석
        breakdown = trust_scorer.get_trust_breakdown(hazard_data, user, db)

        return {
            "status": "success",
            "trust_score": trust_score,
            "is_spam": is_spam,
            "recommendation": "reject" if is_spam else ("review" if trust_score < 50 else "accept"),
            "breakdown": breakdown
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검증 오류: {str(e)}")


@router.get("/hazard/{hazard_id}/nlp-analysis")
async def get_hazard_nlp_analysis(
    hazard_id: str,
    db: Session = Depends(get_db)
):
    """
    특정 위험 정보의 NLP 분석 결과 조회

    Args:
        hazard_id: Hazard ID

    Returns:
        NLP 분석 결과 (감정, 위험도, 키워드 등)
    """
    try:
        hazard = db.query(Hazard).filter(Hazard.id == hazard_id).first()

        if not hazard:
            raise HTTPException(status_code=404, detail="Hazard not found")

        if not hazard.description:
            return {
                "status": "success",
                "hazard_id": hazard_id,
                "analysis": nlp_analyzer._get_default_analysis(),
                "message": "No description available for analysis"
            }

        # NLP 분석 수행
        analysis = nlp_analyzer.analyze_text(hazard.description)

        return {
            "status": "success",
            "hazard_id": hazard_id,
            "description": hazard.description[:200],
            "analysis": analysis
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NLP 분석 오류: {str(e)}")


@router.get("/analytics/overview")
async def get_analytics_overview(db: Session = Depends(get_db)):
    """
    AI 분석 종합 대시보드

    예측, 이상 탐지, 핫스팟을 한 번에 조회

    Returns:
        종합 분석 결과
    """
    try:
        # 향후 7일 예측
        predictions = lstm_predictor.predict_future_hazards(db, days_ahead=7)

        # 최근 24시간 이상 탐지
        anomalies = lstm_predictor.detect_anomalies(db, hours=24)

        # 핫스팟 예측
        hotspots = lstm_predictor.predict_hotspots(db, grid_size_km=10.0)

        return {
            "status": "success",
            "summary": {
                "predictions_count": len(predictions),
                "anomalies_count": len(anomalies),
                "hotspots_count": len(hotspots)
            },
            "predictions": predictions[:5],  # 상위 5개
            "anomalies": anomalies,
            "hotspots": hotspots[:5],  # 상위 5개
            "generated_at": nlp_analyzer._get_default_analysis()["analyzed_at"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 오류: {str(e)}")


@router.post("/predict/deep-learning")
async def deep_learning_predict(
    latitude: float,
    longitude: float,
    timestamp: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    딥러닝 기반 위험도 예측 (LSTM + 시간 승수)

    Args:
        latitude: 위도
        longitude: 경도
        timestamp: 예측 시간 (ISO format, 기본값: 현재 시간)

    Returns:
        딥러닝 예측 결과
    """
    try:
        # 시간 파싱
        if timestamp:
            pred_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        else:
            pred_time = datetime.utcnow()

        # 1. LSTM 위험도 예측
        base_risk, risk_confidence, risk_method = improved_risk_predictor.predict_risk(
            latitude=latitude,
            longitude=longitude,
            timestamp=pred_time,
            confidence_threshold=0.7
        )

        # 2. 시간대별 승수 예측
        time_multiplier, multiplier_method = time_multiplier_predictor.get_multiplier(
            timestamp=pred_time,
            use_hybrid=True
        )

        # 3. 최종 위험도 (승수 적용)
        final_risk = base_risk * time_multiplier
        final_risk = min(100, max(0, final_risk))

        return {
            "status": "success",
            "prediction": {
                "latitude": latitude,
                "longitude": longitude,
                "timestamp": pred_time.isoformat(),
                "base_risk_score": round(base_risk, 2),
                "time_multiplier": time_multiplier,
                "final_risk_score": round(final_risk, 2),
                "confidence": risk_confidence,
                "methods": {
                    "risk_prediction": risk_method,
                    "time_multiplier": multiplier_method
                },
                "model_status": {
                    "lstm_trained": improved_risk_predictor.is_trained,
                    "multiplier_trained": time_multiplier_predictor.is_trained
                }
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"딥러닝 예측 오류: {str(e)}"
        )


@router.get("/predict/compare")
async def compare_prediction_methods(
    latitude: float = Query(..., description="위도"),
    longitude: float = Query(..., description="경도"),
    db: Session = Depends(get_db)
):
    """
    예측 방법 비교 (통계 vs 딥러닝)

    Args:
        latitude: 위도
        longitude: 경도

    Returns:
        통계 기반 vs 딥러닝 예측 비교
    """
    try:
        pred_time = datetime.utcnow()

        # 1. 딥러닝 예측
        dl_risk, dl_confidence, dl_method = improved_risk_predictor.predict_risk(
            latitude=latitude,
            longitude=longitude,
            timestamp=pred_time
        )

        dl_multiplier, dl_mult_method = time_multiplier_predictor.get_multiplier(
            timestamp=pred_time,
            use_hybrid=False
        )

        # 2. 통계 기반 예측 (규칙 기반)
        stat_risk = improved_risk_predictor._rule_based_predict(
            latitude, longitude, pred_time
        )

        stat_multiplier = time_multiplier_predictor.statistical_calculator.get_multiplier(
            pred_time
        )

        return {
            "status": "success",
            "location": {
                "latitude": latitude,
                "longitude": longitude,
                "timestamp": pred_time.isoformat()
            },
            "deep_learning": {
                "base_risk": round(dl_risk, 2),
                "multiplier": dl_multiplier,
                "final_risk": round(dl_risk * dl_multiplier, 2),
                "confidence": dl_confidence,
                "method": dl_method
            },
            "statistical": {
                "base_risk": round(stat_risk, 2),
                "multiplier": stat_multiplier,
                "final_risk": round(stat_risk * stat_multiplier, 2),
                "method": "rule_based"
            },
            "difference": {
                "base_risk_diff": round(dl_risk - stat_risk, 2),
                "multiplier_diff": round(dl_multiplier - stat_multiplier, 2),
                "final_risk_diff": round(
                    (dl_risk * dl_multiplier) - (stat_risk * stat_multiplier),
                    2
                )
            },
            "models_trained": {
                "lstm": improved_risk_predictor.is_trained,
                "time_multiplier": time_multiplier_predictor.is_trained
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"비교 오류: {str(e)}"
        )
