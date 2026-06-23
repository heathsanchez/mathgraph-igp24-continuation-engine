from mathgraph_igp24.leaderboard import load_leaderboard_context
from mathgraph_igp24.portfolio_v2 import generate_survivor_candidates, select_survivor_portfolio, write_portfolio
from mathgraph_igp24.survivor_router import Route


def test_portfolio_v2_constraints_and_format(parent, tmp_path):
    route = Route("r1", "seed", "virgin_support_probe", "virgin", (), 1.0, {}, "virgin_support_probe")
    candidates = generate_survivor_candidates([parent], [route], load_leaderboard_context(tmp_path), 500, seed=1)
    selected = select_survivor_portfolio(candidates)
    assert 0 < len(selected) <= 100
    assert len({item.line for item in selected}) == len(selected)
    assert all("," in item.line and len(item.line.split(",")) == 25 for item in selected)
    report = write_portfolio(tmp_path, selected, candidates)
    assert (tmp_path / "submission.txt").exists()
    assert (tmp_path / "api_submit_cleaned" / "submitted_valid_polys.txt").exists()
    assert report["selected"] == len(selected)
