"""
Hazard Type Migration Script
- conflict -> armed_conflict
- protest -> protest_riot
"""
from app.database import SessionLocal
from app.models.hazard import Hazard
from sqlalchemy import update

def migrate_hazard_types():
    db = SessionLocal()
    try:
        # conflict -> armed_conflict
        conflict_count = db.query(Hazard).filter(Hazard.hazard_type == 'conflict').count()
        if conflict_count > 0:
            db.execute(
                update(Hazard)
                .where(Hazard.hazard_type == 'conflict')
                .values(hazard_type='armed_conflict')
            )
            print(f"[OK] Migrated {conflict_count} 'conflict' records to 'armed_conflict'")
        else:
            print("[OK] No 'conflict' records to migrate")

        # protest -> protest_riot
        protest_count = db.query(Hazard).filter(Hazard.hazard_type == 'protest').count()
        if protest_count > 0:
            db.execute(
                update(Hazard)
                .where(Hazard.hazard_type == 'protest')
                .values(hazard_type='protest_riot')
            )
            print(f"[OK] Migrated {protest_count} 'protest' records to 'protest_riot'")
        else:
            print("[OK] No 'protest' records to migrate")

        db.commit()
        print("\n[OK] Migration completed successfully!")

        # 결과 확인
        print("\nCurrent hazard types in database:")
        from sqlalchemy import distinct
        types = db.query(distinct(Hazard.hazard_type)).all()
        for t in types:
            if t[0]:
                count = db.query(Hazard).filter(Hazard.hazard_type == t[0]).count()
                print(f"  - {t[0]}: {count} records")

    except Exception as e:
        db.rollback()
        print(f"[ERROR] Error during migration: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    migrate_hazard_types()
