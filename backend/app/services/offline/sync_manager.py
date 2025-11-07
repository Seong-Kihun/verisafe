"""오프라인 데이터 동기화 관리자"""
from typing import Dict, List
from datetime import datetime
from sqlalchemy.orm import Session

from app.models.hazard import Hazard
from app.models.report import Report


class SyncManager:
    """
    오프라인 데이터 동기화 관리
    
    Phase 5 구현:
    - 오프라인 변경사항 큐잉
    - 온라인 복귀 시 동기화
    - 충돌 해결
    """
    
    def __init__(self):
        self.sync_queue = []
    
    def queue_offline_change(self, change_type: str, data: Dict) -> str:
        """
        오프라인 변경사항 큐에 추가
        
        Args:
            change_type: 'create_report', 'update_location', etc.
            data: 변경 데이터
        
        Returns:
            큐 ID
        """
        queue_id = f"sync_{datetime.utcnow().timestamp()}"
        
        change = {
            "id": queue_id,
            "type": change_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
            "synced": False
        }
        
        self.sync_queue.append(change)
        
        print(f"[SyncManager] 오프라인 변경 큐잉: {change_type}")
        return queue_id
    
    async def sync_all(self, db: Session) -> Dict:
        """
        모든 오프라인 변경사항 동기화
        
        Args:
            db: Database session
        
        Returns:
            동기화 결과
        """
        print(f"[SyncManager] 동기화 시작: {len(self.sync_queue)}개 항목")
        
        synced = 0
        failed = 0
        errors = []
        
        for change in self.sync_queue:
            if change["synced"]:
                continue
            
            try:
                if change["type"] == "create_report":
                    # 제보 생성
                    report_data = change["data"]
                    report = Report(**report_data)
                    db.add(report)
                    db.commit()
                    
                    change["synced"] = True
                    synced += 1
                    
                # 다른 타입 처리...
                
            except Exception as e:
                print(f"[SyncManager] 동기화 오류: {e}")
                errors.append(str(e))
                failed += 1
        
        # 동기화 완료된 항목 제거
        self.sync_queue = [c for c in self.sync_queue if not c["synced"]]
        
        print(f"[SyncManager] 동기화 완료: {synced}개 성공, {failed}개 실패")
        
        return {
            "status": "success" if failed == 0 else "partial",
            "synced": synced,
            "failed": failed,
            "remaining": len(self.sync_queue),
            "errors": errors
        }
    
    def get_sync_status(self) -> Dict:
        """동기화 상태 조회"""
        pending = len([c for c in self.sync_queue if not c["synced"]])
        
        return {
            "pending_changes": pending,
            "total_queued": len(self.sync_queue),
            "last_sync": datetime.utcnow().isoformat() if pending == 0 else None
        }


# 싱글톤
sync_manager = SyncManager()
