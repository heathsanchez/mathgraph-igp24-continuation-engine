from mathgraph_igp24.leaderboard import load_leaderboard_context
from mathgraph_igp24.obstruction_atlas import learn_obstruction_atlas


def test_obstruction_atlas_detects_reciprocal_collapse_and_crowded(parent):
    records = [
        {"basin_id": "B1", "constructor_family": "reciprocal_random", "reciprocal_score": 0.99,
         "pair": (25000, 2), "t": 25000, "r": 2, "status": "accepted", "support": (0, 24), "line": ",".join(map(str, parent))},
        {"basin_id": "B1", "constructor_family": "high_r_lift", "reciprocal_score": 0.2,
         "pair": (24979, 4), "t": 24979, "r": 4, "status": "accepted", "support": (0, 12, 24), "line": ",".join(map(str, parent))},
        {"basin_id": "B2", "constructor_family": "support_transport", "reciprocal_score": 0.1,
         "pair": None, "status": "failed", "scoringReason": "reducible", "support": (0, 1, 24), "line": ",".join(map(str, parent))},
    ]
    obs = learn_obstruction_atlas(records, [], load_leaderboard_context("/missing"))
    labels = {item.label for item in obs}
    assert {"reciprocal_collapse", "crowded_attractor", "irreducibility_failure"} <= labels
