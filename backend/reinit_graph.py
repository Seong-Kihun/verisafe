"""Re-initialize GraphManager with new dummy graph"""
import asyncio
from app.services.graph_manager import GraphManager
from app.services.hazard_scorer import HazardScorer
from app.database import SessionLocal

async def main():
    print("Re-initializing GraphManager...")

    # Force reset GraphManager
    GraphManager._instance = None
    GraphManager._graph = None

    # Create new instance and initialize
    graph_manager = GraphManager()
    db = SessionLocal()

    try:
        await graph_manager.initialize(db)

        # Get graph
        graph = graph_manager.get_graph()
        print(f'[OK] Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges')

        # Initialize hazard scorer and update risk scores
        hazard_scorer = HazardScorer(graph_manager, session_factory=SessionLocal)
        print('\nUpdating risk scores...')
        await hazard_scorer.update_all_risk_scores()

        # Check results
        edges_with_risk = 0
        total_risk = 0
        max_risk = 0

        for u, v, data in graph.edges(data=True):
            risk = data.get('risk_score', 0)
            if risk > 0:
                edges_with_risk += 1
                total_risk += risk
                max_risk = max(max_risk, risk)

        print(f'\n[OK] Risk scores updated:')
        print(f'  - Edges with risk > 0: {edges_with_risk}')
        print(f'  - Average risk: {total_risk / graph.number_of_edges():.2f}')
        print(f'  - Max risk: {max_risk}')

        print('\n[OK] GraphManager re-initialization complete!')
        print('  The backend server will now use the new graph.')

    finally:
        db.close()

if __name__ == '__main__':
    asyncio.run(main())
