"""AI 모델 학습 API - 딥러닝 모델 학습 및 관리"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Optional
import asyncio

from app.database import get_db
from app.services.ai.improved_risk_predictor import improved_risk_predictor
from app.services.ai.time_multiplier_nn import time_multiplier_predictor
from app.services.ai.data_augmentation import data_augmenter

router = APIRouter()

# 학습 상태 추적
training_status = {
    "risk_predictor": {"status": "idle", "progress": 0},
    "time_multiplier": {"status": "idle", "progress": 0}
}


@router.post("/train/risk-predictor")
async def train_risk_predictor(
    background_tasks: BackgroundTasks,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    db: Session = Depends(get_db)
):
    """
    LSTM 위험도 예측 모델 학습

    Args:
        epochs: 학습 에포크 (기본 100)
        batch_size: 배치 크기 (기본 32)
        learning_rate: 학습률 (기본 0.001)

    Returns:
        학습 결과
    """
    if training_status["risk_predictor"]["status"] == "training":
        raise HTTPException(
            status_code=400,
            detail="모델이 이미 학습 중입니다"
        )

    try:
        training_status["risk_predictor"]["status"] = "training"
        training_status["risk_predictor"]["progress"] = 0

        print(f"\n{'='*60}")
        print(f"LSTM 위험도 예측 모델 학습 시작")
        print(f"  에포크: {epochs}")
        print(f"  배치 크기: {batch_size}")
        print(f"  학습률: {learning_rate}")
        print(f"{'='*60}\n")

        # 모델 학습 (동기적으로 실행)
        result = await improved_risk_predictor.train_model(
            db=db,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )

        training_status["risk_predictor"]["status"] = "completed"
        training_status["risk_predictor"]["progress"] = 100

        return {
            "status": "success",
            "message": "LSTM 위험도 예측 모델 학습 완료",
            "result": result
        }

    except Exception as e:
        training_status["risk_predictor"]["status"] = "error"
        training_status["risk_predictor"]["error"] = str(e)

        raise HTTPException(
            status_code=500,
            detail=f"학습 오류: {str(e)}"
        )


@router.post("/train/time-multiplier")
async def train_time_multiplier(
    background_tasks: BackgroundTasks,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    db: Session = Depends(get_db)
):
    """
    시간대별 승수 예측 모델 학습

    Args:
        epochs: 학습 에포크 (기본 50)
        batch_size: 배치 크기 (기본 32)
        learning_rate: 학습률 (기본 0.01)

    Returns:
        학습 결과
    """
    if training_status["time_multiplier"]["status"] == "training":
        raise HTTPException(
            status_code=400,
            detail="모델이 이미 학습 중입니다"
        )

    try:
        training_status["time_multiplier"]["status"] = "training"
        training_status["time_multiplier"]["progress"] = 0

        print(f"\n{'='*60}")
        print(f"시간대별 승수 예측 모델 학습 시작")
        print(f"  에포크: {epochs}")
        print(f"  배치 크기: {batch_size}")
        print(f"  학습률: {learning_rate}")
        print(f"{'='*60}\n")

        # 모델 학습
        result = await time_multiplier_predictor.train_model(
            db=db,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )

        training_status["time_multiplier"]["status"] = "completed"
        training_status["time_multiplier"]["progress"] = 100

        return {
            "status": "success",
            "message": "시간대별 승수 예측 모델 학습 완료",
            "result": result
        }

    except Exception as e:
        training_status["time_multiplier"]["status"] = "error"
        training_status["time_multiplier"]["error"] = str(e)

        raise HTTPException(
            status_code=500,
            detail=f"학습 오류: {str(e)}"
        )


@router.post("/train/all")
async def train_all_models(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    모든 딥러닝 모델 순차 학습

    Returns:
        학습 결과
    """
    results = {}

    try:
        # 1. 시간대별 승수 모델 학습 (빠름)
        print("\n[1/2] 시간대별 승수 모델 학습...")
        time_result = await time_multiplier_predictor.train_model(
            db=db,
            epochs=50,
            batch_size=32,
            learning_rate=0.01
        )
        results["time_multiplier"] = time_result

        # 2. LSTM 위험도 예측 모델 학습 (느림)
        print("\n[2/2] LSTM 위험도 예측 모델 학습...")
        risk_result = await improved_risk_predictor.train_model(
            db=db,
            epochs=100,
            batch_size=32,
            learning_rate=0.001
        )
        results["risk_predictor"] = risk_result

        return {
            "status": "success",
            "message": "모든 모델 학습 완료",
            "results": results
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"학습 오류: {str(e)}"
        )


@router.get("/train/status")
async def get_training_status():
    """
    학습 상태 조회

    Returns:
        각 모델의 학습 상태
    """
    return {
        "status": "success",
        "training_status": training_status
    }


@router.get("/models/info")
async def get_models_info():
    """
    학습된 모델 정보 조회

    Returns:
        모델 정보 (크기, 학습 여부 등)
    """
    import os

    models_info = {}

    # LSTM 위험도 예측 모델
    lstm_path = improved_risk_predictor.model_path
    models_info["risk_predictor"] = {
        "name": "LSTM Risk Predictor",
        "path": lstm_path,
        "exists": os.path.exists(lstm_path),
        "is_trained": improved_risk_predictor.is_trained,
        "size_mb": round(os.path.getsize(lstm_path) / 1024 / 1024, 2) if os.path.exists(lstm_path) else 0,
        "architecture": "Bidirectional LSTM with Attention",
        "input": "[hour, day_of_week, latitude, longitude, past_risk]",
        "output": "risk_score (0-100)"
    }

    # 시간대별 승수 모델
    multiplier_path = time_multiplier_predictor.model_path
    models_info["time_multiplier"] = {
        "name": "Time Multiplier Network",
        "path": multiplier_path,
        "exists": os.path.exists(multiplier_path),
        "is_trained": time_multiplier_predictor.is_trained,
        "size_mb": round(os.path.getsize(multiplier_path) / 1024 / 1024, 2) if os.path.exists(multiplier_path) else 0,
        "architecture": "Feedforward Neural Network",
        "input": "[day_of_week, hour, time_features]",
        "output": "multiplier (0.5-2.0)"
    }

    return {
        "status": "success",
        "models": models_info
    }


@router.post("/data/generate-synthetic")
async def generate_synthetic_data(
    count: int = 1000,
    db: Session = Depends(get_db)
):
    """
    합성 데이터 생성

    Args:
        count: 생성할 데이터 개수 (기본 1000)

    Returns:
        생성된 데이터 정보
    """
    try:
        print(f"\n합성 데이터 {count}개 생성 중...")

        synthetic_data = data_augmenter.generate_synthetic_data(
            db=db,
            target_count=count
        )

        # 통계 계산
        hazard_types = {}
        for data in synthetic_data:
            hazard_type = data['hazard_type']
            hazard_types[hazard_type] = hazard_types.get(hazard_type, 0) + 1

        avg_risk = sum(d['risk_score'] for d in synthetic_data) / len(synthetic_data)

        return {
            "status": "success",
            "message": f"{len(synthetic_data)}개 합성 데이터 생성 완료",
            "count": len(synthetic_data),
            "distribution": hazard_types,
            "avg_risk_score": round(avg_risk, 1),
            "sample": synthetic_data[:3]  # 샘플 3개
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"합성 데이터 생성 오류: {str(e)}"
        )


@router.delete("/models/{model_name}")
async def delete_model(model_name: str):
    """
    학습된 모델 삭제

    Args:
        model_name: 모델 이름 ("risk_predictor" or "time_multiplier")

    Returns:
        삭제 결과
    """
    import os

    if model_name == "risk_predictor":
        model_path = improved_risk_predictor.model_path
    elif model_name == "time_multiplier":
        model_path = time_multiplier_predictor.model_path
    else:
        raise HTTPException(
            status_code=400,
            detail="잘못된 모델 이름. 'risk_predictor' 또는 'time_multiplier'를 사용하세요"
        )

    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=404,
            detail="모델 파일이 존재하지 않습니다"
        )

    try:
        os.remove(model_path)

        # 모델 상태 초기화
        if model_name == "risk_predictor":
            improved_risk_predictor.is_trained = False
        else:
            time_multiplier_predictor.is_trained = False

        return {
            "status": "success",
            "message": f"{model_name} 모델 삭제 완료",
            "path": model_path
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"모델 삭제 오류: {str(e)}"
        )
