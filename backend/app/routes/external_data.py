"""외부 데이터 수집 API 엔드포인트"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.graph_manager import GraphManager
from app.services.external_data.data_collector_scheduler import DataCollectorScheduler

router = APIRouter()


@router.post("/collect")
async def trigger_data_collection(db: Session = Depends(get_db)):
    """
    외부 데이터 수집 수동 트리거
    
    관리자용 엔드포인트: 즉시 모든 외부 데이터 소스에서 데이터 수집
    """
    try:
        graph_manager = GraphManager()
        scheduler = DataCollectorScheduler(lambda: db, graph_manager)
        
        stats = await scheduler.run_once(db)
        
        return {
            "status": "success",
            "message": f"데이터 수집 완료: 총 {stats['total']}개 항목",
            "statistics": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"데이터 수집 오류: {str(e)}")


@router.get("/status")
async def get_collection_status(db: Session = Depends(get_db)):
    """
    외부 데이터 수집 상태 조회
    
    Returns:
        각 데이터 소스별 최근 수집 통계
    """
    from app.models.hazard import Hazard
    from sqlalchemy import func
    
    try:
        # 각 소스별 위험 정보 개수
        acled_count = db.query(func.count(Hazard.id)).filter(
            Hazard.source.like("acled%")
        ).scalar()
        
        gdacs_count = db.query(func.count(Hazard.id)).filter(
            Hazard.source.like("gdacs%")
        ).scalar()
        
        reliefweb_count = db.query(func.count(Hazard.id)).filter(
            Hazard.source.like("reliefweb%")
        ).scalar()
        
        # 각 소스별 최근 업데이트 시간
        acled_latest = db.query(func.max(Hazard.created_at)).filter(
            Hazard.source.like("acled%")
        ).scalar()
        
        gdacs_latest = db.query(func.max(Hazard.created_at)).filter(
            Hazard.source.like("gdacs%")
        ).scalar()
        
        reliefweb_latest = db.query(func.max(Hazard.created_at)).filter(
            Hazard.source.like("reliefweb%")
        ).scalar()
        
        return {
            "status": "success",
            "data_sources": {
                "acled": {
                    "name": "ACLED (Armed Conflict Location & Event Data)",
                    "count": acled_count,
                    "last_updated": acled_latest.isoformat() if acled_latest else None,
                    "description": "분쟁 및 폭력 사건 데이터"
                },
                "gdacs": {
                    "name": "GDACS (Global Disaster Alert and Coordination System)",
                    "count": gdacs_count,
                    "last_updated": gdacs_latest.isoformat() if gdacs_latest else None,
                    "description": "재난 및 자연재해 데이터"
                },
                "reliefweb": {
                    "name": "ReliefWeb (Humanitarian Information Service)",
                    "count": reliefweb_count,
                    "last_updated": reliefweb_latest.isoformat() if reliefweb_latest else None,
                    "description": "인도적 지원 보고서"
                }
            },
            "total_hazards": acled_count + gdacs_count + reliefweb_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"상태 조회 오류: {str(e)}")


@router.get("/hazards/recent")
async def get_recent_hazards(
    source: str = None,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """
    최근 수집된 위험 정보 조회
    
    Args:
        source: 데이터 소스 필터 (acled, gdacs, reliefweb)
        limit: 반환할 최대 개수
    """
    from app.models.hazard import Hazard
    
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
                    "description": h.description,
                    "start_date": h.start_date.isoformat() if h.start_date else None,
                    "end_date": h.end_date.isoformat() if h.end_date else None,
                    "verified": h.verified,
                    "created_at": h.created_at.isoformat() if h.created_at else None
                }
                for h in hazards
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"조회 오류: {str(e)}")
