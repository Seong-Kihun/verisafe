"""Comprehensive diagnosis of route calculation issues"""
import asyncio
import sys
sys.path.append('.')

from app.services.graph_manager import GraphManager
from app.services.route_calculator import RouteCalculator
from app.services.hazard_scorer import HazardScorer
from app.database import SessionLocal
from app.models.hazard import Hazard
from datetime import datetime

async def diagnose():
    print("=" * 80)
    print("ROUTE CALCULATION DIAGNOSIS")
    print("=" * 80)

    # 1. Check database hazards
    print("\n1. DATABASE HAZARDS CHECK")
    print("-" * 80)
    db = SessionLocal()
    try:
        hazards = db.query(Hazard).filter(
            (Hazard.end_date == None) | (Hazard.end_date > datetime.utcnow())
        ).all()

        print(f"Total active hazards in DB: {len(hazards)}")
        print(f"\nFirst 5 hazards:")
        for h in hazards[:5]:
            print(f"  - ID: {h.id}")
            print(f"    Type: {h.hazard_type}, Risk: {h.risk_score}/100")
            print(f"    Location: ({h.latitude:.4f}, {h.longitude:.4f})")
            print(f"    Radius: {h.radius}km, Verified: {h.verified}")
            print()
    finally:
        db.close()

    # 2. Check graph initialization
    print("\n2. GRAPH INITIALIZATION CHECK")
    print("-" * 80)
    graph_manager = GraphManager()
    db = SessionLocal()
    try:
        await graph_manager.initialize(db)
    finally:
        db.close()

    graph = graph_manager.get_graph()
    print(f"Graph nodes: {graph.number_of_nodes()}")
    print(f"Graph edges: {graph.number_of_edges()}")

    # Check initial risk scores
    edges_with_risk = sum(1 for _, _, d in graph.edges(data=True) if d.get('risk_score', 0) > 0)
    print(f"Edges with risk_score > 0 (before update): {edges_with_risk}")

    # Sample edges
    print("\nSample edges (before risk update):")
    for i, (u, v, data) in enumerate(list(graph.edges(data=True))[:3]):
        u_node = graph.nodes[u]
        v_node = graph.nodes[v]
        print(f"  Edge {i+1}: ({u_node['y']:.4f}, {u_node['x']:.4f}) -> ({v_node['y']:.4f}, {v_node['x']:.4f})")
        print(f"    Length: {data.get('length', 0):.4f}km, Risk: {data.get('risk_score', 0)}")

    # 3. Test hazard scorer
    print("\n3. HAZARD SCORER UPDATE")
    print("-" * 80)
    hazard_scorer = HazardScorer(graph_manager, session_factory=SessionLocal)

    print("Calling update_all_risk_scores()...")
    await hazard_scorer.update_all_risk_scores()

    # Check after update
    edges_with_risk = 0
    total_risk = 0
    max_risk = 0
    risky_edges = []

    for u, v, data in graph.edges(data=True):
        risk = data.get('risk_score', 0)
        if risk > 0:
            edges_with_risk += 1
            total_risk += risk
            max_risk = max(max_risk, risk)
            if len(risky_edges) < 10:
                u_node = graph.nodes[u]
                v_node = graph.nodes[v]
                risky_edges.append({
                    'u': (u_node['y'], u_node['x']),
                    'v': (v_node['y'], v_node['x']),
                    'risk': risk,
                    'length': data.get('length', 0)
                })

    print(f"\nAfter update:")
    print(f"  Edges with risk_score > 0: {edges_with_risk} / {graph.number_of_edges()}")
    print(f"  Average risk (all edges): {total_risk / graph.number_of_edges():.2f}")
    print(f"  Max risk: {max_risk}")

    if risky_edges:
        print(f"\nTop 10 risky edges:")
        for i, edge in enumerate(risky_edges):
            print(f"  {i+1}. ({edge['u'][0]:.4f}, {edge['u'][1]:.4f}) -> ({edge['v'][0]:.4f}, {edge['v'][1]:.4f})")
            print(f"      Risk: {edge['risk']}, Length: {edge['length']:.4f}km")
    else:
        print("\nWARNING: NO EDGES HAVE RISK SCORES AFTER UPDATE!")
        print("     This is the root cause of Issue 2")

    # 4. Test route calculation
    print("\n4. ROUTE CALCULATION TEST")
    print("-" * 80)

    # Test coordinates in Juba area
    start = (4.8594, 31.5713)  # Juba center
    end = (4.8400, 31.5900)     # Nearby location

    print(f"Start: {start}")
    print(f"End: {end}")

    route_calculator = RouteCalculator(graph_manager)

    # Test with safe preference
    print("\nCalculating routes (safe preference, max_routes=10)...")
    result = route_calculator.calculate_route(
        start=start,
        end=end,
        preference='safe',
        transportation_mode='car',
        max_routes=10
    )

    if 'error' in result:
        print(f"[ERROR] {result['error']}")
        return

    routes = result.get('routes', [])
    print(f"\n[OK] Routes calculated: {len(routes)}")

    # Check for duplicate routes
    route_signatures = []
    for i, route in enumerate(routes, 1):
        signature = (route['distance'], route['risk_score'], route['duration'])
        route_signatures.append(signature)

        print(f"\nRoute {i} ({route['type']}):")
        print(f"  Label: {route.get('label', 'N/A')}")
        print(f"  Distance: {route['distance']:.2f}km")
        print(f"  Duration: {route['duration']} min")
        print(f"  Risk: {route['risk_score']}/10")
        print(f"  Waypoints: {len(route['waypoints'])}")
        print(f"  Polyline points: {len(route['polyline'])}")

    # Check for duplicates
    print("\n5. DUPLICATE ROUTE CHECK")
    print("-" * 80)
    unique_routes = len(set(route_signatures))
    print(f"Unique route signatures: {unique_routes} / {len(routes)}")

    if unique_routes < len(routes):
        print("WARNING: DUPLICATE ROUTES DETECTED!")
        print("     This is the root cause of Issue 1")

        # Show which routes are duplicates
        from collections import Counter
        signature_counts = Counter(route_signatures)
        for sig, count in signature_counts.items():
            if count > 1:
                print(f"     Duplicate: distance={sig[0]:.2f}km, risk={sig[1]}, duration={sig[2]}min ({count} times)")
    else:
        print("[OK] All routes are unique")

    # 6. Summary
    print("\n" + "=" * 80)
    print("DIAGNOSIS SUMMARY")
    print("=" * 80)
    print(f"Issue 1 (Duplicate routes): {'CONFIRMED' if unique_routes < len(routes) else 'NOT DETECTED'}")
    print(f"Issue 2 (Risk scores = 0): {'CONFIRMED' if edges_with_risk == 0 else 'NOT DETECTED'}")

    if edges_with_risk == 0:
        print("\nWARNING: Risk scores are not being applied to edges!")
        print("    Checking hazard_scorer logic...")

        # Additional debug: check if hazards are in the right location
        db = SessionLocal()
        try:
            hazards = db.query(Hazard).filter(Hazard.verified == True).all()
            print(f"\n    Verified hazards: {len(hazards)}")

            if hazards:
                # Check if any hazards are near the graph edges
                print(f"    Sample hazard locations:")
                for h in hazards[:3]:
                    print(f"      ({h.latitude:.4f}, {h.longitude:.4f}) radius={h.radius}km")

                # Check graph node bounds
                lats = [data['y'] for _, data in graph.nodes(data=True)]
                lngs = [data['x'] for _, data in graph.nodes(data=True)]
                print(f"\n    Graph bounds:")
                print(f"      Lat: {min(lats):.4f} to {max(lats):.4f}")
                print(f"      Lng: {min(lngs):.4f} to {max(lngs):.4f}")
        finally:
            db.close()

if __name__ == '__main__':
    asyncio.run(diagnose())
