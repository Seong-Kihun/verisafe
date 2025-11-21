"""제보 API (JWT 인증 적용)"""
from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from sqlalchemy.orm import Session
from typing import Optional
import uuid

from app.database import get_db
from app.models import Report, User, Hazard, HazardScoringRule
from app.schemas.report import ReportCreate, ReportResponse, ReportListResponse
from app.services.auth_service import get_current_active_user, get_current_user_optional
from app.utils.geo import haversine_distance
from app.utils.helpers import db_transaction
from app.middleware.rate_limiter import limiter
from datetime import datetime, timedelta

router = APIRouter()


@router.post("/create", response_model=ReportResponse)
@limiter.limit("10/minute")  # 1분에 10개 제보로 제한
async def create_report(
    request: Request,
    report: ReportCreate,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """제보 등록 (익명 제보 허용, 새 필드 지원, 속도 제한 10/분)"""
    with db_transaction(db, "제보 등록"):
        db_report = Report(
            id=uuid.uuid4(),
            user_id=current_user.id if current_user else None,  # 익명 제보 허용
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
@limiter.limit("30/minute")  # 1분에 30회 조회로 제한
async def get_nearby_reports(
    request: Request,
    latitude: float = Query(..., description="중심 위도"),
    longitude: float = Query(..., description="중심 경도"),
    radius: float = Query(0.5, description="반경 (km)"),
    hours: int = Query(24, description="최근 몇 시간 이내"),
    db: Session = Depends(get_db)
):
    """주변 제보 조회 (중복 체크용) - Haversine 거리 계산 사용, 속도 제한 30/분"""
    # 시간 필터
    time_threshold = datetime.utcnow() - timedelta(hours=hours)

    # 1단계: 대략적인 범위로 후보군 조회 (DB 쿼리 최적화)
    # 안전 마진을 위해 radius * 1.5 범위로 조회
    lat_diff = (radius * 1.5) / 111.0  # 1도 = 약 111km
    lng_diff = (radius * 1.5) / (111.0 * abs(latitude) if latitude != 0 else 111.0)

    candidate_reports = db.query(Report).filter(
        Report.created_at >= time_threshold,
        Report.latitude >= latitude - lat_diff,
        Report.latitude <= latitude + lat_diff,
        Report.longitude >= longitude - lng_diff,
        Report.longitude <= longitude + lng_diff,
        Report.is_draft == False
    ).all()

    # 2단계: Haversine 공식으로 정확한 거리 계산 및 필터링
    nearby_reports = []
    for r in candidate_reports:
        distance = haversine_distance(latitude, longitude, r.latitude, r.longitude)
        if distance <= radius:
            nearby_reports.append({
                "id": str(r.id),
                "hazard_type": r.hazard_type,
                "latitude": r.latitude,
                "longitude": r.longitude,
                "created_at": r.created_at.isoformat(),
                "severity": r.severity,
                "distance_km": round(distance, 3)  # 거리 정보 추가
            })

    # 거리순 정렬
    nearby_reports.sort(key=lambda x: x["distance_km"])

    return {"reports": nearby_reports}


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

    with db_transaction(db, "제보 삭제"):
        db.delete(report)

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

    # 제보 승인 및 hazards 테이블 추가를 하나의 트랜잭션으로 처리
    with db_transaction(db, "제보 검증"):
        # 제보 승인
        report.status = 'verified'
        report.verified_by = current_user.id
        report.verified_at = datetime.now()

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

    db.refresh(report)
    return report
