"""관리자 대시보드 API"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, timedelta

from app.database import get_db
from app.models.hazard import Hazard
from app.models.report import Report
from app.models.user import User

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
        
        report.status = status
        report.verified_at = datetime.utcnow()
        # TODO: report.verified_by = current_admin_user_id
        
        db.commit()
        
        return {
            "status": "success",
            "report_id": report_id,
            "new_status": status,
            "verified_at": report.verified_at.isoformat()
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="유효하지 않은 report_id 형식")
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
