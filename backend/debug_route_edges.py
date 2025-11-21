"""Debug route edges risk scores"""
from app.services.graph_manager import GraphManager
from app.services.route_calculator import RouteCalculator

# Initialize
graph_manager = GraphManager()
route_calculator = RouteCalculator(graph_manager)

# Test coordinates
start = (4.850, 31.570)
end = (4.860, 31.585)

print(f"Testing route from {start} to {end}")
print()

# Get graph
graph = graph_manager.get_graph()
print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

# Check sample edges
edges_checked = 0
edges_with_risk = 0
total_risk = 0

print("\nChecking first 10 edges:")
for u, v, data in list(graph.edges(data=True))[:10]:
    risk = data.get('risk_score', 0)
    edges_checked += 1
    if risk > 0:
        edges_with_risk += 1
        total_risk += risk
    print(f"  Edge ({u} -> {v}): risk_score = {risk}")

print(f"\nSummary: {edges_with_risk}/{edges_checked} edges have risk > 0")
print(f"Average risk of checked edges: {total_risk / edges_checked if edges_checked > 0 else 0:.2f}")

# Calculate route
print("\n" + "="*50)
print("Calculating route...")
result = route_calculator.calculate_route(start, end, preference='safe', transportation_mode='car', max_routes=2)

if 'error' in result:
    print(f"Error: {result['error']}")
else:
    routes = result.get('routes', [])
    print(f"Found {len(routes)} routes:")
    for i, route in enumerate(routes):
        print(f"\nRoute {i+1} ({route.get('type')}):")
        print(f"  Distance: {route.get('distance')}km")
        print(f"  Risk score: {route.get('risk_score')}/10")
        print(f"  Number of waypoints: {len(route.get('waypoints', []))}")
