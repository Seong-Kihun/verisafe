"""관리자 대시보드 API"""
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, timedelta
from typing import Optional

from app.database import get_db
from app.models.hazard import Hazard
from app.models.report import Report
from app.models.user import User
from app.services.satellite.image_analyzer import satellite_image_analyzer
from app.utils.helpers import db_transaction

router = APIRouter()


@router.get("/stats/overview")
async def get_overview_stats(db: Session = Depends(get_db)):
    """
    대시보드 개요 통계
    
    Returns:
        전체 시스템 통계
    """
    try:
        # 총 사용자 수
        total_users = db.query(func.count(User.id)).scalar()
        
        # 총 위험 정보
        total_hazards = db.query(func.count(Hazard.id)).scalar()
        
        # 총 제보
        total_reports = db.query(func.count(Report.id)).scalar()
        
        # 검증 대기 중인 제보
        pending_reports = db.query(func.count(Report.id)).filter(
            Report.status == 'pending'
        ).scalar()
        
        # 활성 위험 (end_date가 없거나 미래)
        active_hazards = db.query(func.count(Hazard.id)).filter(
            (Hazard.end_date == None) | (Hazard.end_date > datetime.utcnow())
        ).scalar()
        
        return {
            "status": "success",
            "overview": {
                "total_users": total_users or 0,
                "total_hazards": total_hazards or 0,
                "active_hazards": active_hazards or 0,
                "total_reports": total_reports or 0,
                "pending_reports": pending_reports or 0
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"통계 조회 오류: {str(e)}")


@router.get("/stats/daily")
async def get_daily_stats(days: int = 7, db: Session = Depends(get_db)):
    """
    일별 통계 (최근 N일)
    
    Args:
        days: 조회할 일수
    """
    try:
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # 일별 제보 수
        reports_daily = db.query(
            func.date_trunc('day', Report.created_at).label('date'),
            func.count(Report.id).label('count')
        ).filter(
            Report.created_at >= start_date
        ).group_by('date').order_by('date').all()
        
        # 일별 위험 정보 생성 수
        hazards_daily = db.query(
            func.date_trunc('day', Hazard.created_at).label('date'),
            func.count(Hazard.id).label('count')
        ).filter(
            Hazard.created_at >= start_date
        ).group_by('date').order_by('date').all()
        
        return {
            "status": "success",
            "period": {"start": start_date.isoformat(), "days": days},
            "reports_daily": [
                {"date": str(r.date.date()), "count": r.count}
                for r in reports_daily
            ],
            "hazards_daily": [
                {"date": str(h.date.date()), "count": h.count}
                for h in hazards_daily
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"일별 통계 오류: {str(e)}")


@router.get("/stats/hazards-by-type")
async def get_hazards_by_type(db: Session = Depends(get_db)):
    """위험 유형별 통계"""
    try:
        hazard_types = db.query(
            Hazard.hazard_type,
            func.count(Hazard.id).label('count'),
            func.avg(Hazard.risk_score).label('avg_risk')
        ).group_by(Hazard.hazard_type).all()
        
        return {
            "status": "success",
            "hazard_types": [
                {
                    "type": ht.hazard_type,
                    "count": ht.count,
                    "average_risk_score": round(float(ht.avg_risk or 0), 2)
                }
                for ht in hazard_types
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"유형별 통계 오류: {str(e)}")


@router.get("/stats/sources")
async def get_sources_stats(db: Session = Depends(get_db)):
    """데이터 소스별 통계"""
    try:
        sources = db.query(
            Hazard.source,
            func.count(Hazard.id).label('count')
        ).group_by(Hazard.source).all()
        
        return {
            "status": "success",
            "sources": [
                {"source": s.source, "count": s.count}
                for s in sources
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"소스 통계 오류: {str(e)}")


@router.get("/reports/pending")
async def get_pending_reports(limit: int = 20, db: Session = Depends(get_db)):
    """
    검증 대기 중인 제보 리스트
    
    Args:
        limit: 반환할 최대 개수
    """
    try:
        reports = db.query(Report).filter(
            Report.status == 'pending'
        ).order_by(Report.created_at.desc()).limit(limit).all()
        
        return {
            "status": "success",
            "count": len(reports),
            "reports": [
                {
                    "id": str(r.id),
                    "hazard_type": r.hazard_type,
                    "description": r.description,
                    "latitude": r.latitude,
                    "longitude": r.longitude,
                    "user_id": str(r.user_id) if r.user_id else None,
                    "created_at": r.created_at.isoformat() if r.created_at else None
                }
                for r in reports
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"제보 조회 오류: {str(e)}")


@router.post("/reports/{report_id}/verify")
async def verify_report(report_id: str, status: str, db: Session = Depends(get_db)):
    """
    제보 검증 (승인/거부)

    Args:
        report_id: 제보 ID
        status: 'verified' or 'rejected'
    """
    if status not in ['verified', 'rejected']:
        raise HTTPException(status_code=400, detail="상태는 'verified' 또는 'rejected'여야 합니다")

    try:
        from uuid import UUID
        report = db.query(Report).filter(Report.id == UUID(report_id)).first()

        if not report:
            raise HTTPException(status_code=404, detail="제보를 찾을 수 없습니다")

        with db_transaction(db, "제보 검증"):
            report.status = status
            report.verified_at = datetime.utcnow()
            # TODO: report.verified_by = current_admin_user_id

        return {
            "status": "success",
            "report_id": report_id,
            "new_status": status,
            "verified_at": report.verified_at.isoformat()
        }

    except ValueError:
        raise HTTPException(status_code=400, detail="유효하지 않은 report_id 형식")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검증 오류: {str(e)}")


@router.get("/system/health")
async def get_system_health(db: Session = Depends(get_db)):
    """시스템 헬스 체크"""
    from app.services.redis_manager import redis_manager

    try:
        # DB 연결 테스트
        db.execute("SELECT 1")
        db_status = "healthy"
    except:
        db_status = "unhealthy"

    # Redis 상태
    redis_stats = redis_manager.get_stats()

    return {
        "status": "success",
        "system_health": {
            "database": db_status,
            "redis": redis_stats.get("status", "unknown"),
            "timestamp": datetime.utcnow().isoformat()
        }
    }


@router.get("/system/graph")
async def get_graph_status():
    """GraphManager 상태 확인"""
    from app.services.graph_manager import GraphManager

    graph_manager = GraphManager()
    graph = graph_manager.get_graph()

    if graph:
        # Check risk scores
        edges_with_risk = 0
        total_risk = 0
        for u, v, data in graph.edges(data=True):
            risk = data.get('risk_score', 0)
            if risk > 0:
                edges_with_risk += 1
                total_risk += risk

        return {
            "status": "success",
            "graph": {
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "initialized": graph.number_of_nodes() > 10,
                "edges_with_risk": edges_with_risk,
                "avg_risk": round(total_risk / graph.number_of_edges(), 2) if graph.number_of_edges() > 0 else 0
            }
        }
    else:
        return {
            "status": "error",
            "message": "Graph not initialized"
        }


@router.post("/system/update-risk-scores")
async def update_risk_scores():
    """위험도 업데이트 강제 실행"""
    from app.services.graph_manager import GraphManager
    from app.services.hazard_scorer import HazardScorer
    from app.database import SessionLocal

    try:
        graph_manager = GraphManager()
        hazard_scorer = HazardScorer(graph_manager, session_factory=SessionLocal)

        await hazard_scorer.update_all_risk_scores()

        graph = graph_manager.get_graph()
        edges_with_risk = 0
        for u, v, data in graph.edges(data=True):
            if data.get('risk_score', 0) > 0:
                edges_with_risk += 1

        return {
            "status": "success",
            "message": "Risk scores updated successfully",
            "edges_with_risk": edges_with_risk,
            "total_edges": graph.number_of_edges()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating risk scores: {str(e)}")


@router.post("/satellite/analyze-demo")
async def analyze_satellite_demo(file: UploadFile = File(...)):
    """
    위성 이미지 분석 시각화 (아이디어톤 데모용)

    단계별 분석 과정을 시각화하여 반환:
    1. 원본 이미지
    2. 도로 감지 결과
    3. 건물 감지 결과
    4. 최종 합성 이미지

    Args:
        file: 위성 이미지 파일

    Returns:
        분석 결과 + 각 단계별 이미지 (base64)
    """
    try:
        # 이미지 파일 검증
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다")

        # 파일 크기 제한 (10MB)
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="파일 크기는 10MB 이하여야 합니다")

        # 위성 이미지 분석 + 시각화
        result = satellite_image_analyzer.analyze_image_with_visualization(contents)

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return {
            "status": "success",
            "filename": file.filename,
            "result": result
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"위성 이미지 분석 오류: {str(e)}")


@router.get("/satellite/sample-demo")
async def get_sample_satellite_demo():
    """
    샘플 위성 이미지 분석 데모

    미리 준비된 샘플 이미지로 즉시 시연 가능
    (이미지 업로드 없이 데모 실행)

    Returns:
        샘플 분석 결과 + 시각화
    """
    try:
        import os

        # 샘플 이미지 경로
        sample_path = "app/services/satellite/sample_juba.jpg"

        # 샘플 이미지 확인
        if os.path.exists(sample_path):
            with open(sample_path, 'rb') as f:
                image_data = f.read()

            result = satellite_image_analyzer.analyze_image_with_visualization(image_data)
        else:
            # 샘플 이미지 없으면 더미 데이터 반환
            result = satellite_image_analyzer._dummy_visualization()

        return {
            "status": "success",
            "sample_location": "Juba, South Sudan",
            "note": "This is a demo analysis using sample satellite imagery",
            "result": result
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        # 오류 시 더미 데이터 반환
        result = satellite_image_analyzer._dummy_visualization()
        return {
            "status": "success",
            "sample_location": "Juba, South Sudan (Demo)",
            "note": "Fallback to dummy data due to error",
            "result": result
        }


@router.get("/ai/timeseries-demo")
async def get_timeseries_prediction_demo():
    """
    시계열 위험도 예측 시각화 데모 (아이디어톤용)

    LSTM + 시간대별 승수 예측 과정을 시각화하여 반환

    Returns:
        시계열 데이터 + 예측 결과 + 시각화 데이터
    """
    from datetime import datetime, timedelta
    from app.services.ai.improved_risk_predictor import improved_risk_predictor
    from app.services.ai.time_multiplier_nn import time_multiplier_predictor
    import random
    import numpy as np

    try:
        # 현재 시간 (금요일 저녁 7시로 시뮬레이션)
        current_time = datetime(2025, 11, 7, 19, 0)  # 금요일 19시

        # 남수단 주바 좌표
        latitude = 4.85
        longitude = 31.6

        # === 1. 과거 7일 시계열 데이터 생성 ===
        past_7_days = []
        for i in range(7, 0, -1):
            past_time = current_time - timedelta(days=i)

            # 시뮬레이션: 요일/시간대별 패턴
            hour = past_time.hour
            dow = past_time.weekday()

            # 기본 위험도
            if dow == 4 and 17 <= hour <= 20:  # 금요일 저녁
                base = 70 + random.randint(-10, 10)
            elif dow in [5, 6]:  # 주말
                base = 35 + random.randint(-5, 5)
            elif 22 <= hour or hour <= 5:  # 심야
                base = 55 + random.randint(-10, 10)
            else:
                base = 45 + random.randint(-10, 10)

            past_7_days.append({
                "date": past_time.strftime("%Y-%m-%d"),
                "time": past_time.strftime("%H:%M"),
                "day_name": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][dow],
                "risk_score": base,
                "is_weekend": dow in [5, 6],
                "is_evening": 17 <= hour <= 20
            })

        # === 2. LSTM 예측 ===
        try:
            base_risk, confidence, method = improved_risk_predictor.predict_risk(
                latitude, longitude, current_time
            )
        except:
            # 폴백: 규칙 기반
            base_risk = 65.0
            confidence = 0.75
            method = "rule_based"

        # === 3. 시간대별 승수 예측 ===
        try:
            multiplier, multiplier_method = time_multiplier_predictor.get_multiplier(
                current_time, use_hybrid=True
            )
        except:
            # 폴백
            multiplier = 1.5
            multiplier_method = "statistical"

        # === 4. 최종 위험도 계산 ===
        final_risk = min(100, base_risk * multiplier)

        # === 5. 시간대별 승수 히트맵 데이터 (7일 × 24시간) ===
        heatmap_data = []
        days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

        for dow in range(7):
            day_data = []
            for hour in range(24):
                # 시뮬레이션: 요일/시간별 승수 패턴
                if dow == 5 and 17 <= hour <= 20:  # 금요일 저녁
                    mult = 1.8 + random.uniform(-0.1, 0.1)
                elif dow == 5 and 12 <= hour <= 16:  # 금요일 오후
                    mult = 1.5 + random.uniform(-0.1, 0.1)
                elif dow in [0, 6]:  # 주말
                    mult = 0.7 + random.uniform(-0.1, 0.1)
                elif 22 <= hour or hour <= 5:  # 심야
                    mult = 1.3 + random.uniform(-0.1, 0.1)
                elif 9 <= hour <= 17:  # 업무 시간
                    mult = 1.1 + random.uniform(-0.1, 0.1)
                else:
                    mult = 1.0 + random.uniform(-0.1, 0.1)

                day_data.append(round(mult, 2))

            heatmap_data.append({
                "day": days[dow],
                "multipliers": day_data
            })

        # === 6. 향후 24시간 예측 ===
        future_24h = []
        for i in range(24):
            future_time = current_time + timedelta(hours=i)
            hour = future_time.hour

            # 시뮬레이션
            if 17 <= hour <= 20:
                risk = 85 + random.randint(-5, 5)
            elif 22 <= hour or hour <= 5:
                risk = 65 + random.randint(-5, 5)
            else:
                risk = 50 + random.randint(-10, 10)

            future_24h.append({
                "time": future_time.strftime("%H:%M"),
                "risk_score": risk
            })

        return {
            "status": "success",
            "timestamp": current_time.isoformat(),
            "location": {
                "name": "Juba, South Sudan",
                "latitude": latitude,
                "longitude": longitude
            },
            "past_7_days": past_7_days,
            "prediction": {
                "base_risk_score": round(base_risk, 2),
                "time_multiplier": multiplier,
                "final_risk_score": round(final_risk, 2),
                "confidence": round(confidence, 2),
                "lstm_method": method,
                "multiplier_method": multiplier_method
            },
            "heatmap": {
                "title": "시간대별 위험도 승수 (요일 × 시간)",
                "data": heatmap_data,
                "hours": list(range(24))
            },
            "future_24h": future_24h,
            "model_info": {
                "lstm": {
                    "architecture": "Bidirectional LSTM + Attention",
                    "hidden_size": 128,
                    "num_layers": 3,
                    "sequence_length": 7
                },
                "multiplier": {
                    "architecture": "Feedforward Neural Network",
                    "input_features": 10,
                    "hidden_layers": [32, 16, 8]
                }
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"시계열 예측 오류: {str(e)}")
