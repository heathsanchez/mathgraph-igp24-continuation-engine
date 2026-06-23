from mathgraph_igp24.leaderboard import load_leaderboard_context
from mathgraph_igp24.obstruction_atlas import SurvivorObstruction
from mathgraph_igp24.survivor_router import recommend_routes


def test_survivor_router_penalizes_crowded_and_rewards_boundary(tmp_path):
    ctx = load_leaderboard_context(tmp_path)
    basins = [
        {"basin_id": "crowded", "survival_rate": 1.0, "novelty_score": 0.1, "low_k_proxy_score": 0.1,
         "high_r_score": 0.0, "crowded_pair_share": 1.0, "banned_pair_share": 1.0, "unique_t": 1, "unique_r": 1, "count_total": 10},
        {"basin_id": "boundary", "survival_rate": 0.5, "novelty_score": 1.0, "low_k_proxy_score": 1.0,
         "high_r_score": 0.5, "crowded_pair_share": 0.0, "banned_pair_share": 0.0, "unique_t": 2, "unique_r": 2, "count_total": 4},
    ]
    obs = [SurvivorObstruction("o1", "crowded_attractor", 9, ("crowded",), ("x",), ("(25000,2)",), (), 0.9, "quotient_escape", ())]
    routes = recommend_routes(basins, ctx, obs)
    assert routes[0].source_basin_id == "boundary"
    assert routes[0].evidence["boundary_bonus"] > 1
