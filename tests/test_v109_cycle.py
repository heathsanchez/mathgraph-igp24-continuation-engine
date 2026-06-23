import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_v109_cycle_dry_run_creates_expected_artifacts(tmp_path):
    env = dict(os.environ)
    env.pop("SAIR_API_KEY", None)
    completed = subprocess.run(
        [sys.executable, "scripts/run_v109_survivor_cycle.py", "--root", str(tmp_path), "--candidate-count", "500", "--dry-run"],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    out = tmp_path / "v109_survivor_geometry_cycle"
    for name in ("submission.txt", "selected_meta.json", "candidate_pool.csv", "portfolio_report.json",
                 "basin_atlas.csv", "obstruction_atlas.csv", "route_scores.csv", "phase_boundaries.csv",
                 "api_submit_cleaned/submitted_valid_polys.txt"):
        assert (out / name).exists(), name
    report = json.loads((out / "run_report.json").read_text())
    assert report["submitted"] is False
    assert report["selected"] == 100
    assert "constructor_rationale" in report
    assert "submitted" in completed.stdout


def test_v109_mock_api_join_updates_selected_meta(tmp_path):
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from run_v109_survivor_cycle import join_selected_meta

    out = tmp_path / "v109_survivor_geometry_cycle"
    out.mkdir()
    (out / "selected_meta.json").write_text(json.dumps([
        {"index": 0, "line": "1," + ",".join(["0"] * 23) + ",1", "trust_label": "EMPIRICAL_ROUTE"},
        {"index": 1, "line": "1,1," + ",".join(["0"] * 22) + ",1", "trust_label": "EMPIRICAL_ROUTE"},
    ]), encoding="utf-8")
    join_selected_meta(out, {
        "verifiedPolynomials": [{"polynomialIndex": 0, "computed_label": "24T24648", "computed_r": 4}],
        "failedPolynomials": [{"polynomialIndex": 1, "reason": "reducible"}],
    })
    rows = json.loads((out / "selected_meta.json").read_text())
    assert rows[0]["trust_label"] == "VERIFIED_BY_API"
    assert rows[1]["trust_label"] == "OBSTRUCTED"
