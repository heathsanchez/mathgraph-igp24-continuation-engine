from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from mathgraph_igp24 import (
    BasinLaw,
    Evidence,
    TrialLedger,
    basin_id,
    create_trial,
    hash_poly,
    join_result,
    recommend,
    replay_law,
)
from mathgraph_igp24.models import TRUST_OBSERVED_TRIAL, TRUST_SUBMITTED

ROOT = Path(__file__).resolve().parents[1]
PARENT = (1,) + (0,) * 23 + (1,)


def center_mass_law() -> BasinLaw:
    deltas = [0] * 25
    deltas[12] = 18
    return BasinLaw(
        law_id="LAW-center-mass-18",
        source_basin=basin_id(PARENT),
        target_basin="B-target",
        target_pair=(14010, 8),
        coefficient_deltas=tuple(deltas),
        preconditions={"source_basin": basin_id(PARENT)},
        evidence=Evidence(7, 3, 0.40, 0.86, ("TR1", "TR2"), 3, 1),
    )


def test_recommendation_replays_to_frozen_child_hash():
    rec = recommend(PARENT, (14010, 8), laws=[center_mass_law()])
    child = replay_law(PARENT, rec.law)
    assert hash_poly(child) == rec.expected_child_hash
    assert child[12] == 18
    assert rec.evidence.successes == 7
    assert rec.evidence.failures == 3


def test_recommendation_is_deterministic():
    first = recommend(PARENT, (14010, 8), laws=[center_mass_law()])
    second = recommend(PARENT, (14010, 8), laws=[center_mass_law()])
    assert first == second


def test_join_result_uses_submission_and_index(tmp_path):
    law = center_mass_law()
    child = replay_law(PARENT, law)
    trial = create_trial(
        PARENT, child, "law_apply", {"law_id": law.law_id}, (14010, 8), random_seed=17
    )
    submitted = replace(
        trial, submission_id="submission-1", polynomial_index=4,
        status="SUBMITTED", trust_label=TRUST_SUBMITTED,
    )
    ledger = TrialLedger(tmp_path / "trials.jsonl")
    ledger.append(submitted)
    observed = join_result(
        "submission-1", 4,
        {"status": "ok", "computed_label": "24T14010", "computed_r": 8},
        ledger,
    )
    assert observed.trial_id == trial.trial_id
    assert observed.pair_after == (14010, 8)
    assert observed.outcome["computed_label"] == "24T14010"


def test_generated_colab_artifact_is_deterministic_and_self_contained(tmp_path):
    subprocess.run([sys.executable, "scripts/build_colab.py"], cwd=ROOT, check=True, capture_output=True, text=True)
    artifact = ROOT / "dist" / "mathgraph_igp24_v102_colab.py"
    first = artifact.read_bytes()
    subprocess.run([sys.executable, "scripts/build_colab.py"], cwd=ROOT, check=True, capture_output=True, text=True)
    second = artifact.read_bytes()
    assert first == second
    manifest = json.loads((ROOT / "dist" / "artifact_manifest.json").read_text())
    assert hashlib.sha256(first).hexdigest() == manifest["artifact_sha256"]
    code = """
import runpy
m = runpy.run_path('dist/mathgraph_igp24_v102_colab.py', run_name='embedded_test')
p = (1,) + (0,) * 23 + (1,)
rec = m['recommend'](p, (14010, 8))
child = m['replay_law'](p, rec.law)
assert m['hash_poly'](child) == rec.expected_child_hash
print(rec.expected_child_hash)
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code], cwd=ROOT,
        check=True, capture_output=True, text=True,
    )
    assert len(completed.stdout.strip()) == 64
