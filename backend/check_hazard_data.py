"""Check hazard data status"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app.database import SessionLocal
from app.models.hazard import Hazard, HazardScoringRule
from collections import Counter

db = SessionLocal()

try:
    # Check hazards
    hazards = db.query(Hazard).all()
    print(f'Total hazards: {len(hazards)}')
    print(f'Hazards with country data: {sum(1 for h in hazards if h.country)}')
    print(f'Hazards without country data: {sum(1 for h in hazards if not h.country)}')

    print('\nCountry distribution:')
    countries = Counter(h.country for h in hazards if h.country)
    for country, count in sorted(countries.items()):
        print(f'  {country}: {count}')

    # Check scoring rules
    print('\n' + '='*50)
    rules = db.query(HazardScoringRule).all()
    print(f'HazardScoringRule records: {len(rules)}')

    if rules:
        print('\nRegistered rules:')
        for rule in rules:
            print(f'  - {rule.hazard_type}:')
            print(f'      base_risk_score: {rule.base_risk_score}')
            print(f'      default_radius_km: {rule.default_radius_km}')
            print(f'      default_duration_hours: {rule.default_duration_hours}')
    else:
        print('\nWARNING: No HazardScoringRule data found!')
        print('Sample hazards were created with hardcoded values, not from scoring rules.')

    # Check if sample data used scoring rules
    print('\n' + '='*50)
    print('Sample data analysis:')
    hazard_types = Counter(h.hazard_type for h in hazards)
    print('\nHazard type distribution:')
    for htype, count in sorted(hazard_types.items()):
        print(f'  {htype}: {count}')

finally:
    db.close()
