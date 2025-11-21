"""Check if graph edges have risk scores"""
import asyncio
from app.services.graph_manager import GraphManager
from app.services.hazard_scorer import HazardScorer
from app.database import SessionLocal

async def main():
    # Initialize graph and hazard scorer
    graph_manager = GraphManager()
    hazard_scorer = HazardScorer(graph_manager, session_factory=SessionLocal)

    # Get graph
    graph = graph_manager.get_graph()

    print(f'Total nodes: {graph.number_of_nodes()}')
    print(f'Total edges: {graph.number_of_edges()}')

    # Check risk scores before update
    edges_with_risk = 0
    total_risk = 0
    for u, v, data in graph.edges(data=True):
        risk = data.get('risk_score', 0)
        if risk > 0:
            edges_with_risk += 1
            total_risk += risk

    print(f'\nBefore update:')
    print(f'  Edges with risk > 0: {edges_with_risk}')
    print(f'  Average risk (all edges): {total_risk / graph.number_of_edges():.2f}')

    # Update risk scores
    print('\nUpdating risk scores...')
    await hazard_scorer.update_all_risk_scores()

    # Check risk scores after update
    edges_with_risk = 0
    total_risk = 0
    max_risk = 0
    sample_edges = []

    for u, v, data in graph.edges(data=True):
        risk = data.get('risk_score', 0)
        if risk > 0:
            edges_with_risk += 1
            total_risk += risk
            max_risk = max(max_risk, risk)
            if len(sample_edges) < 5:
                u_data = graph.nodes[u]
                v_data = graph.nodes[v]
                sample_edges.append({
                    'u': (u_data['y'], u_data['x']),
                    'v': (v_data['y'], v_data['x']),
                    'risk': risk
                })

    print(f'\nAfter update:')
    print(f'  Edges with risk > 0: {edges_with_risk}')
    print(f'  Average risk (all edges): {total_risk / graph.number_of_edges():.2f}')
    print(f'  Max risk: {max_risk}')

    if sample_edges:
        print(f'\nSample edges with risk:')
        for i, edge in enumerate(sample_edges):
            print(f"  {i+1}. ({edge['u'][0]:.4f}, {edge['u'][1]:.4f}) -> ({edge['v'][0]:.4f}, {edge['v'][1]:.4f}): risk={edge['risk']}")

if __name__ == '__main__':
    asyncio.run(main())
