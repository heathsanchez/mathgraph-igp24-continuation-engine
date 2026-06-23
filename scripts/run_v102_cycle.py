from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mathgraph_igp24 import (
    LawBook, SairClient, TrialLedger, build_memory, build_portfolio, build_submission,
    join_result, learn_laws, learn_obstructions, recommend,
)
from mathgraph_igp24.api import get_api_key
from mathgraph_igp24.models import TRUST_OBSERVED_TRIAL
from mathgraph_igp24.obstructions import DEFAULT_HARMFUL_PAIRS

DEFAULT_ROOT = Path(os.environ.get("MATHGRAPH_ROOT", "/content/drive/MyDrive/MathGraph_IGP24"))
TARGET_PAIR = (14010, 8)
SEED_PARENT = tuple([1, 1] + [0] * 22 + [1])


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows: handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fields = list(fields or (rows[0].keys() if rows else []))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


def parents_from_trials(trials) -> list[tuple[int, ...]]:
    parents = []; seen = set()
    for trial in reversed(trials):
        if trial.trust_label == TRUST_OBSERVED_TRIAL and trial.child_hash not in seen:
            parents.append(trial.child_polynomial); seen.add(trial.child_hash)
        if len(parents) >= 50: break
    return parents or [SEED_PARENT]


def api_rows(final: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    return list(final.get("verifiedPolynomials", []) or []), list(final.get("failedPolynomials", []) or [])


def cycle(root: Path = DEFAULT_ROOT, target_pair=TARGET_PAIR, submit: bool = False, poll: bool = False, seed: int = 102) -> dict[str, Any]:
    cycle_id = f"v102_{utc_stamp()}"; out = root / "v102_provenance_lawbook_scoring_engine"; cycle_dir = out / "cycles" / cycle_id
    cycle_dir.mkdir(parents=True, exist_ok=True); ledger = TrialLedger(out / "trials.jsonl")
    trials_before = ledger.load(); memory = build_memory(trials_before)
    laws_before = learn_laws(trials_before); obstructions_before = learn_obstructions(trials_before)
    parents = parents_from_trials(trials_before)
    selected = build_portfolio(parents, target_pair, laws_before, obstructions_before,
                               memory.known_child_hashes, ledger, cycle_id, seed)
    polynomials = [candidate.polynomial for candidate in selected]
    submission_text = build_submission(polynomials)
    (cycle_dir / "submission.txt").write_text(submission_text, encoding="utf-8")
    (out / "submission.txt").write_text(submission_text, encoding="utf-8")
    pending = [candidate.trial.to_dict() for candidate in selected]
    write_jsonl(cycle_dir / "trials_pending.jsonl", pending)
    selected_rows = [{
        "index": index, "trial_id": candidate.trial.trial_id, "child_hash": candidate.trial.child_hash,
        "polynomial": ",".join(map(str, candidate.polynomial)), "mutation_type": candidate.trial.mutation_type,
        "source_basin": candidate.trial.source_basin, "expected_value": candidate.expected_value,
        "obstruction_penalty": candidate.obstruction_penalty,
    } for index, candidate in enumerate(selected)]
    write_csv(cycle_dir / "selected_top100.csv", selected_rows)
    LawBook(tuple(laws_before), tuple(obstructions_before)).save(cycle_dir / "basin_lawbook.json")
    write_json(cycle_dir / "obstructions.json", [asdict(item) for item in obstructions_before])
    report = {
        "cycle_id": cycle_id, "generated_candidates": 1200 if laws_before else 2000,
        "selected": len(selected), "expected_value": sum(candidate.expected_value for candidate in selected),
        "target_pair_distribution": dict(Counter(str(candidate.trial.predicted_target_pair) for candidate in selected)),
        "mutation_type_distribution": dict(Counter(candidate.trial.mutation_type for candidate in selected)),
        "law_trust_distribution": dict(Counter(law.trust_label for law in laws_before)),
        "obstruction_penalties_applied": sum(candidate.obstruction_penalty < 1 for candidate in selected),
        "novelty_rate": sum(candidate.trial.child_hash not in memory.known_child_hashes for candidate in selected) / max(1, len(selected)),
        "submission_bytes": len(submission_text.encode()), "submitted": False,
    }
    write_json(cycle_dir / "run_report.json", report)
    print(json.dumps(report, indent=2))
    if not submit:
        print("recommended next action: inspect dry-run portfolio; set SAIR_SUBMIT=1 to submit")
        return report

    key = get_api_key(prompt=True); client = SairClient(key)
    response = client.submit(polynomials, f"MathGraph v102 cycle {cycle_id}")
    submission_id = str(response.get("submissionId") or response.get("id") or "")
    if not submission_id: raise RuntimeError("submission response lacks submissionId")
    registered = ledger.register_submission([candidate.trial for candidate in selected], submission_id)
    report.update({"submitted": True, "submission_id": submission_id})
    if not poll:
        write_json(cycle_dir / "run_report.json", report); return report

    final = client.poll(submission_id); write_json(cycle_dir / "api_results.json", final)
    verified, failed = api_rows(final)
    for row in verified + failed:
        index = row.get("polynomialIndex", row.get("polynomial_index", row.get("index")))
        if index is not None: join_result(submission_id, int(index), row, ledger)
    write_csv(cycle_dir / "verified_from_api.csv", verified, sorted(set().union(*(row.keys() for row in verified))) if verified else ["polynomialIndex", "status", "computed_label", "computed_r"])
    write_csv(cycle_dir / "failed_from_api.csv", failed, sorted(set().union(*(row.keys() for row in failed))) if failed else ["polynomialIndex", "status", "reason"])
    actual_pairs = Counter()
    for row in verified:
        label = str(row.get("computed_label", "")); digits = "".join(ch if ch.isdigit() else " " for ch in label).split()
        t = row.get("t", int(digits[-1]) if digits else None); r = row.get("r", row.get("computed_r"))
        if t is not None and r is not None: actual_pairs[(int(t), int(r))] += 1
    write_csv(cycle_dir / "verified_pairs_summary.csv", [{"t": pair[0], "r": pair[1], "count": count, "bad_attractor": pair in DEFAULT_HARMFUL_PAIRS} for pair, count in actual_pairs.items()])
    trials_after = ledger.load(); laws_after = learn_laws(trials_after); obstructions_after = learn_obstructions(trials_after)
    LawBook(tuple(laws_after), tuple(obstructions_after)).save(cycle_dir / "updated_lawbook.json")
    write_json(cycle_dir / "updated_obstructions.json", [asdict(item) for item in obstructions_after])
    observed = [trial for trial in trials_after if trial.submission_id == submission_id and trial.trust_label == TRUST_OBSERVED_TRIAL]
    write_jsonl(cycle_dir / "trial_certificates.jsonl", [trial.to_dict() for trial in observed])
    next_rows = []
    next_parents = parents_from_trials(trials_after)
    for parent in next_parents[:20]:
        rec = recommend(parent, target_pair, laws_after, obstructions_after)
        next_rows.append({"recommendation_id": rec.recommendation_id, "parent_hash": rec.parent_hash,
                          "law_id": rec.law.law_id, "expected_child_hash": rec.expected_child_hash,
                          "posterior": rec.target_pair_posterior, "confidence_lower": rec.confidence_interval[0],
                          "support": rec.support_count, "failures": rec.failure_count, "rationale": " | ".join(rec.rationale)})
    write_csv(cycle_dir / "next_recommendations.csv", next_rows)
    before = {law.law_id: law for law in laws_before}
    strengthened = [law.law_id for law in laws_after if law.law_id not in before or law.success_count > before[law.law_id].success_count]
    weakened = [law.law_id for law in laws_after if law.law_id in before and law.failure_count > before[law.law_id].failure_count]
    report.update({
        "actual_pair_distribution": {str(pair): count for pair, count in actual_pairs.items()},
        "score_candidate_pairs": {str(pair): count for pair, count in actual_pairs.items() if pair not in DEFAULT_HARMFUL_PAIRS},
        "bad_attractor_pairs": {str(pair): count for pair, count in actual_pairs.items() if pair in DEFAULT_HARMFUL_PAIRS},
        "laws_strengthened": strengthened, "laws_weakened": weakened,
        "new_obstructions": len(obstructions_after) - len(obstructions_before),
        "recommended_next_action": "apply highest-evidence unobstructed laws in the next cycle",
    })
    write_json(cycle_dir / "run_report.json", report); print(json.dumps(report, indent=2)); return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--root", default=str(DEFAULT_ROOT)); parser.add_argument("--target-t", type=int, default=14010)
    parser.add_argument("--target-r", type=int, default=8); parser.add_argument("--seed", type=int, default=102)
    args, _unknown = parser.parse_known_args(argv)
    cycle(Path(args.root), (args.target_t, args.target_r), os.environ.get("SAIR_SUBMIT") == "1", os.environ.get("SAIR_POLL") == "1", args.seed)


if __name__ == "__main__": main()
