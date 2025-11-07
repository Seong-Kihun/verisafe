"""제보 API (JWT 인증 적용)"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import Optional
import uuid

from app.database import get_db
from app.models import Report, User, Hazard, HazardScoringRule
from app.schemas.report import ReportCreate, ReportResponse, ReportListResponse
from app.services.auth_service import get_current_active_user
from datetime import datetime, timedelta

router = APIRouter()


@router.post("/create", response_model=ReportResponse)
async def create_report(
    report: ReportCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """제보 등록 (새 필드 지원)"""
    db_report = Report(
        id=uuid.uuid4(),
        user_id=current_user.id,
        hazard_type=report.hazard_type,
        description=report.description,
        latitude=report.latitude,
        longitude=report.longitude,
        image_url=report.image_url,
        status='pending',
        # 새 필드들
        severity=getattr(report, 'severity', 'medium'),
        reported_at=getattr(report, 'reported_at', datetime.utcnow()),
        photos=getattr(report, 'photos', []),
        is_draft=getattr(report, 'is_draft', False),
        accuracy=getattr(report, 'accuracy', None),
        conditional_data=getattr(report, 'conditional_data', {}),
        impact_count=0
    )

    db.add(db_report)
    db.commit()
    db.refresh(db_report)

    return db_report


@router.get("/list", response_model=ReportListResponse)
async def list_reports(
    lat: float = Query(4.8594, description="중심 위도"),
    lng: float = Query(31.5713, description="중심 경도"),
    radius: float = Query(15.0, description="반경 (km)"),
    status_filter: Optional[str] = Query(None, description="필터링 (pending/verified/rejected)"),
    db: Session = Depends(get_db)
):
    """제보 목록 조회"""
    query = db.query(Report)
    
    if status_filter:
        query = query.filter(Report.status == status_filter)
    
    reports = query.all()
    
    return ReportListResponse(reports=reports)


@router.get("/drafts", response_model=ReportListResponse)
async def get_drafts(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """임시 저장된 제보 목록 조회"""
    reports = db.query(Report).filter(
        Report.user_id == current_user.id,
        Report.is_draft == True
    ).order_by(Report.created_at.desc()).all()

    return ReportListResponse(reports=reports)


@router.get("/nearby")
async def get_nearby_reports(
    latitude: float = Query(..., description="중심 위도"),
    longitude: float = Query(..., description="중심 경도"),
    radius: float = Query(0.5, description="반경 (km)"),
    hours: int = Query(24, description="최근 몇 시간 이내"),
    db: Session = Depends(get_db)
):
    """주변 제보 조회 (중복 체크용)"""
    from datetime import datetime, timedelta

    # 시간 필터
    time_threshold = datetime.utcnow() - timedelta(hours=hours)

    # 간단한 거리 계산 (위도/경도 차이)
    # 실제로는 PostGIS ST_Distance 사용 권장
    lat_diff = radius / 111.0  # 1도 = 약 111km
    lng_diff = radius / (111.0 * abs(latitude))

    reports = db.query(Report).filter(
        Report.created_at >= time_threshold,
        Report.latitude >= latitude - lat_diff,
        Report.latitude <= latitude + lat_diff,
        Report.longitude >= longitude - lng_diff,
        Report.longitude <= longitude + lng_diff,
        Report.is_draft == False
    ).all()

    return {"reports": [
        {
            "id": str(r.id),
            "hazard_type": r.hazard_type,
            "latitude": r.latitude,
            "longitude": r.longitude,
            "created_at": r.created_at.isoformat(),
            "severity": r.severity
        }
        for r in reports
    ]}


@router.delete("/{report_id}")
async def delete_report(
    report_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """제보 삭제 (본인 또는 관리자만)"""
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="제보를 찾을 수 없습니다"
        )

    # 권한 체크
    if report.user_id != current_user.id and current_user.role != 'admin':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="삭제 권한이 없습니다"
        )

    db.delete(report)
    db.commit()

    return {"message": "제보가 삭제되었습니다"}


@router.post("/{report_id}/verify", response_model=ReportResponse)
async def verify_report(
    report_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """관리자 제보 검증 (승인)"""
    if current_user.role != 'admin':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="관리자만 검증할 수 있습니다"
        )

    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="제보를 찾을 수 없습니다"
        )

    # 제보 승인
    report.status = 'verified'
    report.verified_by = current_user.id
    report.verified_at = datetime.now()

    db.commit()
    db.refresh(report)

    # hazards 테이블에 추가
    rule = db.query(HazardScoringRule).filter(
        HazardScoringRule.hazard_type == report.hazard_type
    ).first()

    if rule:
        hazard = Hazard(
            id=uuid.uuid4(),
            hazard_type=report.hazard_type,
            risk_score=rule.base_risk_score,
            latitude=report.latitude,
            longitude=report.longitude,
            radius=rule.default_radius_km,
            source='user_report',
            description=report.description,
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(hours=rule.default_duration_hours),
            verified=True
        )

        db.add(hazard)
        db.commit()

    return report
