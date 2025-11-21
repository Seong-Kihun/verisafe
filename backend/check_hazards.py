"""Check hazard data in database"""
from app.database import SessionLocal
from app.models.hazard import Hazard

db = SessionLocal()

# Count total hazards
hazards = db.query(Hazard).all()
print(f'Total hazards: {len(hazards)}')

# Count verified hazards
verified_hazards = db.query(Hazard).filter(Hazard.verified == True).all()
print(f'Verified hazards: {len(verified_hazards)}')

# Count unverified hazards
unverified_hazards = db.query(Hazard).filter(Hazard.verified == False).all()
print(f'Unverified hazards: {len(unverified_hazards)}')

# Show sample data
if hazards:
    print('\nSample hazards (first 5):')
    for i, h in enumerate(hazards[:5]):
        print(f'  {i+1}. type={h.hazard_type}, risk_score={h.risk_score}, verified={h.verified}, radius={h.radius}km, lat={h.latitude}, lng={h.longitude}')

db.close()
