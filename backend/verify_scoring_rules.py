"""Verify that hazards follow scoring rules"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app.database import SessionLocal
from app.models.hazard import Hazard, HazardScoringRule
from collections import defaultdict

db = SessionLocal()

try:
    # Load scoring rules
    rules = {r.hazard_type: r for r in db.query(HazardScoringRule).all()}

    # Get all hazards
    hazards = db.query(Hazard).all()

    print("=== Scoring Rule Validation ===\n")

    violations = defaultdict(list)
    valid_count = 0

    for hazard in hazards:
        rule = rules.get(hazard.hazard_type)
        if not rule:
            violations['missing_rule'].append(hazard.hazard_type)
            continue

        # Check risk score range
        if not (rule.min_risk_score <= hazard.risk_score <= rule.max_risk_score):
            violations['risk_score'].append(
                f"{hazard.hazard_type}: score={hazard.risk_score} (expected {rule.min_risk_score}-{rule.max_risk_score})"
            )
            continue

        # Check radius range (should be within 50%-150% of default)
        expected_min = rule.default_radius_km * 0.5
        expected_max = rule.default_radius_km * 1.5
        if not (expected_min <= hazard.radius <= expected_max):
            violations['radius'].append(
                f"{hazard.hazard_type}: radius={hazard.radius}km (expected {expected_min:.1f}-{expected_max:.1f}km)"
            )
            continue

        valid_count += 1

    print(f"Total hazards: {len(hazards)}")
    print(f"Valid hazards (following rules): {valid_count}")
    print(f"Violations: {len(hazards) - valid_count}")

    if violations:
        print("\n=== Violation Details ===")
        for vtype, items in violations.items():
            print(f"\n{vtype.upper()} violations ({len(items)}):")
            for item in items[:5]:  # Show first 5
                print(f"  - {item}")
            if len(items) > 5:
                print(f"  ... and {len(items) - 5} more")
    else:
        print("\n✅ ALL hazards follow scoring rules correctly!")

    # Sample verification
    print("\n=== Sample Verification (first 3 hazards) ===")
    for hazard in hazards[:3]:
        rule = rules.get(hazard.hazard_type)
        print(f"\nHazard: {hazard.hazard_type} ({hazard.country})")
        print(f"  Risk Score: {hazard.risk_score} (rule: {rule.min_risk_score}-{rule.max_risk_score}) ✓")
        print(f"  Radius: {hazard.radius}km (rule: {rule.default_radius_km * 0.5:.1f}-{rule.default_radius_km * 1.5:.1f}km) ✓")
        print(f"  Description: {hazard.description}")

finally:
    db.close()
