"""데이터 수집 대시보드 API"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, timedelta

from app.database import get_db
from app.models.hazard import Hazard
from app.services.graph_manager import GraphManager
from app.services.external_data.data_collector_scheduler import DataCollectorScheduler
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get("/dashboard/stats")
async def get_dashboard_stats(db: Session = Depends(get_db)):
    """
    외부 데이터 수집 대시보드 통계

    Returns:
        전체 통계, 소스별 통계, 위험도 분포 등
    """
    try:
        # 전체 통계
        total_hazards = db.query(func.count(Hazard.id)).scalar() or 0

        # 소스별 통계
        sources = {}
        for source_name in ['acled', 'gdacs', 'reliefweb', 'twitter', 'news', 'sentinel']:
            count = db.query(func.count(Hazard.id)).filter(
                Hazard.source.like(f"{source_name}%")
            ).scalar() or 0

            latest = db.query(func.max(Hazard.created_at)).filter(
                Hazard.source.like(f"{source_name}%")
            ).scalar()

            sources[source_name] = {
                "count": count,
                "last_updated": latest.isoformat() if latest else None,
                "status": "active" if count > 0 else "inactive"
            }

        # 최근 24시간 수집 데이터
        yesterday = datetime.utcnow() - timedelta(hours=24)
        recent_count = db.query(func.count(Hazard.id)).filter(
            Hazard.created_at >= yesterday
        ).scalar() or 0

        # 위험도별 분포
        risk_distribution = {
            "low": db.query(func.count(Hazard.id)).filter(
                Hazard.risk_score < 40
            ).scalar() or 0,
            "medium": db.query(func.count(Hazard.id)).filter(
                Hazard.risk_score >= 40,
                Hazard.risk_score < 70
            ).scalar() or 0,
            "high": db.query(func.count(Hazard.id)).filter(
                Hazard.risk_score >= 70
            ).scalar() or 0,
        }

        # 위험 유형별 분포
        type_distribution = {}
        for hazard_type in ['conflict', 'protest', 'checkpoint', 'road_damage', 'natural_disaster', 'safe_haven', 'other']:
            count = db.query(func.count(Hazard.id)).filter(
                Hazard.hazard_type == hazard_type
            ).scalar() or 0
            if count > 0:
                type_distribution[hazard_type] = count

        return {
            "total_hazards": total_hazards,
            "sources": sources,
            "recent_24h": recent_count,
            "risk_distribution": risk_distribution,
            "type_distribution": type_distribution,
            "last_check": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"통계 조회 오류: {str(e)}")


@router.post("/dashboard/trigger-collection")
async def trigger_manual_collection(db: Session = Depends(get_db)):
    """
    수동으로 외부 데이터 수집 트리거

    관리자가 즉시 데이터를 수집하고 싶을 때 사용
    """
    try:
        graph_manager = GraphManager()
        scheduler = DataCollectorScheduler(lambda: db, graph_manager)

        logger.info("수동 데이터 수집 트리거됨")
        stats = await scheduler.run_once(db)

        return {
            "status": "success",
            "message": f"데이터 수집 완료: 총 {stats['total']}개 항목",
            "statistics": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"데이터 수집 오류: {str(e)}")


@router.get("/dashboard/recent-hazards")
async def get_recent_hazards(
    limit: int = 20,
    source: str = None,
    db: Session = Depends(get_db)
):
    """
    최근 수집된 위험 정보 목록

    Args:
        limit: 반환할 최대 개수 (기본 20)
        source: 소스 필터 (acled, gdacs, reliefweb 등)
    """
    try:
        query = db.query(Hazard)

        if source:
            query = query.filter(Hazard.source.like(f"{source}%"))

        hazards = query.order_by(Hazard.created_at.desc()).limit(limit).all()

        return {
            "status": "success",
            "count": len(hazards),
            "hazards": [
                {
                    "id": str(h.id),
                    "type": h.hazard_type,
                    "risk_score": h.risk_score,
                    "latitude": h.latitude,
                    "longitude": h.longitude,
                    "radius": h.radius,
                    "source": h.source,
                    "description": h.description[:200] if h.description else "",
                    "verified": h.verified,
                    "created_at": h.created_at.isoformat() if h.created_at else None
                }
                for h in hazards
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"조회 오류: {str(e)}")


@router.get("/dashboard/collection-history")
async def get_collection_history(days: int = 7, db: Session = Depends(get_db)):
    """
    최근 N일간 수집 히스토리

    Args:
        days: 조회할 일수 (기본 7일)
    """
    try:
        since_date = datetime.utcnow() - timedelta(days=days)

        # 일별 수집 통계
        daily_stats = []
        for i in range(days):
            day_start = since_date + timedelta(days=i)
            day_end = day_start + timedelta(days=1)

            count = db.query(func.count(Hazard.id)).filter(
                Hazard.created_at >= day_start,
                Hazard.created_at < day_end
            ).scalar() or 0

            daily_stats.append({
                "date": day_start.strftime("%Y-%m-%d"),
                "count": count
            })

        return {
            "status": "success",
            "period": f"{days} days",
            "daily_stats": daily_stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"히스토리 조회 오류: {str(e)}")
