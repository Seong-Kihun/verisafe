"""Test route calculation with safe vs fast routes"""
import requests
import json

# Backend URL
BASE_URL = "http://localhost:8000"

# Test coordinates (within Juba area with hazards)
start = {"lat": 4.850, "lng": 31.570}
end = {"lat": 4.860, "lng": 31.585}

# Route calculation request
payload = {
    "start": start,
    "end": end,
    "preference": "safe",  # Request safe route
    "transportation_mode": "car",
    "excluded_hazard_types": []
}

print("Testing route calculation...")
print(f"Start: {start}")
print(f"End: {end}")
print()

# Send request
response = requests.post(f"{BASE_URL}/api/route/calculate", json=payload)

if response.status_code == 200:
    result = response.json()
    routes = result.get("routes", [])

    print(f"[OK] Success! Received {len(routes)} routes:")
    print()

    for i, route in enumerate(routes):
        route_type = route.get("type", "unknown")
        distance = route.get("distance", 0)
        duration = route.get("duration", 0)
        risk_score = route.get("risk_score", 0)

        print(f"Route {i+1} ({route_type}):")
        print(f"  - Distance: {distance}km")
        print(f"  - Duration: {duration} minutes")
        print(f"  - Risk Score: {risk_score}/10")
        print()

    # Check if safe and fast routes are different
    if len(routes) >= 2:
        safe_route = next((r for r in routes if r.get("type") == "safe"), None)
        fast_route = next((r for r in routes if r.get("type") == "fast"), None)

        if safe_route and fast_route:
            if safe_route.get("risk_score") != fast_route.get("risk_score"):
                print("[SUCCESS] Safe and fast routes have different risk scores!")
                print(f"  Safe route risk: {safe_route.get('risk_score')}/10")
                print(f"  Fast route risk: {fast_route.get('risk_score')}/10")
            else:
                print("[ISSUE] Safe and fast routes have the same risk score")
                print(f"  Both have risk score: {safe_route.get('risk_score')}/10")
        else:
            print("[ISSUE] Could not find both safe and fast routes")
    else:
        print("[ISSUE] Not enough routes returned")
else:
    print(f"[ERROR] Status code: {response.status_code}")
    print(response.text)
