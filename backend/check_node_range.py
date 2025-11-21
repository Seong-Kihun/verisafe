"""Check node coordinate range in graph"""
from app.services.graph_manager import GraphManager

graph_manager = GraphManager()
graph = graph_manager.get_graph()

if graph.number_of_nodes() == 0:
    print("Graph is empty!")
else:
    print(f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")

    # Get lat/lng ranges
    lats = []
    lngs = []

    for node, data in graph.nodes(data=True):
        lats.append(data['y'])
        lngs.append(data['x'])

    print(f"\nCoordinate ranges:")
    print(f"  Latitude:  {min(lats):.6f} to {max(lats):.6f}")
    print(f"  Longitude: {min(lngs):.6f} to {max(lngs):.6f}")

    # Sample nodes
    print(f"\nSample nodes (first 5):")
    for i, (node, data) in enumerate(list(graph.nodes(data=True))[:5]):
        print(f"  Node {node}: lat={data['y']:.6f}, lng={data['x']:.6f}")
