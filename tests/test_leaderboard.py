import json

from mathgraph_igp24.leaderboard import HARD_BLACKLIST, classify_pair, load_leaderboard_context


def test_leaderboard_loads_csv_json_and_classifies(tmp_path):
    (tmp_path / "igp24_leaderboard.csv").write_text("t,r,k,score\n24648,4,1,1.0\n9993,0,20,0.0\n", encoding="utf-8")
    (tmp_path / "leaderboard_extra.json").write_text(json.dumps({"pairs": [{"pair": [16055, 4], "k": 2, "score": 0.5}]}), encoding="utf-8")
    ctx = load_leaderboard_context(tmp_path)
    assert (24648, 4) in ctx.known_pairs
    assert classify_pair((24648, 4), ctx).score_class == "LOW_K"
    assert classify_pair((16055, 4), ctx).target_value == 500
    assert classify_pair((25000, 2), ctx).target_value < 0
    assert (25000, 2) in HARD_BLACKLIST
