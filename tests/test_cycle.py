import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_run_v102_cycle_dry_without_api_key(tmp_path):
    env = dict(os.environ); env.pop("SAIR_API_KEY", None); env.pop("SAIR_SUBMIT", None); env.pop("PYTHONPATH", None)
    completed = subprocess.run([sys.executable, "scripts/run_v102_cycle.py", "--root", str(tmp_path), "--candidate-count", "2000"],
                               cwd=ROOT, env=env, check=True, capture_output=True, text=True)
    out = tmp_path / "v102_provenance_cycle"
    report_paths = list(out.glob("cycles/*/run_report.json"))
    assert len(report_paths) == 1
    report = json.loads((out / "run_report.json").read_text())
    assert report["selected"] == 100 and report["submitted"] is False
    for name in ("submission.txt", "submission.json", "selected_top100.csv", "candidate_pool.csv",
                 "trials.jsonl", "lawbook.json", "obstructions.json", "run_report.json"):
        assert (out / name).exists(), name
    rows = [json.loads(line) for line in (out / "trials.jsonl").read_text().splitlines()]
    assert len(rows) == 100
    assert all(row["trial_id"] and row["child_hash"] for row in rows)
    assert "recommended next action" in completed.stdout
