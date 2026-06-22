import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_run_v102_cycle_dry_without_api_key(tmp_path):
    env = dict(os.environ); env.pop("SAIR_API_KEY", None); env.pop("SAIR_SUBMIT", None); env["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run([sys.executable, "scripts/run_v102_cycle.py", "--root", str(tmp_path)],
                               cwd=ROOT, env=env, check=True, capture_output=True, text=True)
    report_paths = list(tmp_path.glob("v102_provenance_lawbook_scoring_engine/cycles/*/run_report.json"))
    assert len(report_paths) == 1
    report = json.loads(report_paths[0].read_text())
    assert report["selected"] == 100 and report["submitted"] is False
    pending = report_paths[0].parent / "trials_pending.jsonl"
    rows = [json.loads(line) for line in pending.read_text().splitlines()]
    assert len(rows) == 100
    assert all(row["trial_id"] and row["child_hash"] for row in rows)
    assert "recommended next action" in completed.stdout
