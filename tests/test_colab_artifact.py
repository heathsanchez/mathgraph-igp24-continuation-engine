import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_colab_artifact_is_deterministic_and_runs_cycle(tmp_path):
    subprocess.run([sys.executable, "scripts/build_colab_artifact.py"], cwd=ROOT, check=True)
    artifact = ROOT / "dist/mathgraph_igp24_v102_colab.py"; first = artifact.read_bytes()
    subprocess.run([sys.executable, "scripts/build_colab_artifact.py"], cwd=ROOT, check=True)
    assert first == artifact.read_bytes()
    manifest = json.loads((ROOT / "dist/artifact_manifest.json").read_text())
    assert hashlib.sha256(first).hexdigest() == manifest["artifact_sha256"]
    env = dict(os.environ); env.pop("SAIR_API_KEY", None); env.pop("SAIR_SUBMIT", None)
    subprocess.run([sys.executable, "-I", str(artifact), "--root", str(tmp_path), "--candidate-count", "2000"], cwd=ROOT, env=env,
                   check=True, capture_output=True, text=True)
    assert list(tmp_path.glob("v102_provenance_cycle/cycles/*/submission.txt"))
