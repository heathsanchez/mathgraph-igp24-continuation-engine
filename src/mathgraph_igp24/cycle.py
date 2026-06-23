from __future__ import annotations

import csv
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .api import SairClient, get_api_key
from .lawbook import LawBook
from .memory import build_memory
from .models import Pair, Polynomial, TRUST_OBSERVED_TRIAL
from .obstructions import DEFAULT_HARMFUL_PAIRS, learn_obstructions
from .portfolio import (
    PortfolioCandidate,
    fallback_atlas_candidates,
    generate_candidates,
    select_portfolio,
)
from .router import recommend
from .submission import build_submission
from .trials import TrialLedger, join_result
from .laws import learn_laws

DEFAULT_ROOT = Path("/content/drive/MyDrive/MathGraph_IGP24")
DEFAULT_TARGET_PAIR: Pair = (14010, 8)
DEFAULT_VERSION = "v102_provenance_cycle"
DEFAULT_CANDIDATES = 250_000
SEED_PARENT: Polynomial = tuple([1, 1] + [0] * 22 + [1])


@dataclass(frozen=True)
class CycleConfig:
    root: Path = DEFAULT_ROOT
    target_pair: Pair = DEFAULT_TARGET_PAIR
    seed: int = 102
    candidate_count: int = DEFAULT_CANDIDATES
    version: str = DEFAULT_VERSION
    submit: bool = False
    poll: bool = False
    api_base: str = "https://api.sair.foundation/api/public/v1"

    @classmethod
    def from_env(
        cls,
        root: str | Path | None = None,
        target_pair: Pair = DEFAULT_TARGET_PAIR,
        seed: int | None = None,
        candidate_count: int | None = None,
        version: str | None = None,
    ) -> "CycleConfig":
        return cls(
            root=Path(root or os.environ.get("MATHGRAPH_ROOT", str(DEFAULT_ROOT))),
            target_pair=target_pair,
            seed=int(seed if seed is not None else os.environ.get("MATHGRAPH_SEED", "102")),
            candidate_count=int(candidate_count if candidate_count is not None else os.environ.get("MATHGRAPH_CANDIDATES", str(DEFAULT_CANDIDATES))),
            version=version or os.environ.get("MATHGRAPH_VERSION", DEFAULT_VERSION),
            submit=os.environ.get("SAIR_SUBMIT", "0") == "1",
            poll=os.environ.get("SAIR_POLL", "0") == "1",
            api_base=os.environ.get("SAIR_API_BASE", "https://api.sair.foundation/api/public/v1"),
        )


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = list(fields or (rows[0].keys() if rows else []))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parents_from_trials(trials: Sequence[Any]) -> list[Polynomial]:
    parents: list[Polynomial] = []
    seen: set[str] = set()
    for trial in reversed(trials):
        if trial.trust_label == TRUST_OBSERVED_TRIAL and trial.child_hash not in seen:
            parents.append(trial.child_polynomial)
            seen.add(trial.child_hash)
        if len(parents) >= 50:
            break
    return parents or [SEED_PARENT]


def api_rows(final: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return list(final.get("verifiedPolynomials", []) or []), list(final.get("failedPolynomials", []) or [])


def candidate_rows(candidates: Sequence[PortfolioCandidate], limit: int = 50_000) -> list[dict[str, Any]]:
    ranked = sorted(candidates, key=lambda item: item.expected_value, reverse=True)[:limit]
    return [
        {
            "rank": index,
            "trial_id": candidate.trial.trial_id,
            "child_hash": candidate.trial.child_hash,
            "mutation_type": candidate.trial.mutation_type,
            "source_basin": candidate.trial.source_basin,
            "expected_value": candidate.expected_value,
            "obstruction_penalty": candidate.obstruction_penalty,
        }
        for index, candidate in enumerate(ranked)
    ]


def selected_rows(selected: Sequence[PortfolioCandidate]) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "trial_id": candidate.trial.trial_id,
            "child_hash": candidate.trial.child_hash,
            "polynomial": ",".join(map(str, candidate.polynomial)),
            "mutation_type": candidate.trial.mutation_type,
            "source_basin": candidate.trial.source_basin,
            "expected_value": candidate.expected_value,
            "obstruction_penalty": candidate.obstruction_penalty,
        }
        for index, candidate in enumerate(selected)
    ]


def build_candidate_pool(
    parents: Sequence[Polynomial],
    target_pair: Pair,
    laws,
    obstructions,
    known_hashes: set[str],
    candidate_count: int,
    cycle_id: str,
    seed: int,
) -> list[PortfolioCandidate]:
    if laws:
        return generate_candidates(parents, target_pair, laws, obstructions, known_hashes, candidate_count, cycle_id, seed)
    return fallback_atlas_candidates(parents, target_pair, candidate_count, cycle_id, seed)


def run_cycle(config: CycleConfig) -> dict[str, Any]:
    cycle_id = f"{config.version}_{utc_stamp()}"
    out = config.root / config.version
    cycle_dir = out / "cycles" / cycle_id
    cycle_dir.mkdir(parents=True, exist_ok=True)

    ledger = TrialLedger(out / "trials.jsonl")
    trials_before = ledger.load()
    memory = build_memory(trials_before)
    laws_before = learn_laws(trials_before)
    obstructions_before = learn_obstructions(trials_before)
    parents = parents_from_trials(trials_before)

    candidates = build_candidate_pool(
        parents,
        config.target_pair,
        laws_before,
        obstructions_before,
        memory.known_child_hashes,
        config.candidate_count,
        cycle_id,
        config.seed,
    )
    selected = select_portfolio(candidates)
    ledger.append_many([candidate.trial for candidate in selected])
    polynomials = [candidate.polynomial for candidate in selected]
    submission_text = build_submission(polynomials)
    submission_json = {"payload": {"polynomials": submission_text.splitlines()}}

    for directory in (out, cycle_dir):
        (directory / "submission.txt").write_text(submission_text, encoding="utf-8")
        write_json(directory / "submission.json", submission_json)
        write_csv(directory / "selected_top100.csv", selected_rows(selected))
        write_csv(directory / "candidate_pool.csv", candidate_rows(candidates))
        LawBook(tuple(laws_before), tuple(obstructions_before)).save(directory / "lawbook.json")
        write_json(directory / "obstructions.json", [asdict(item) for item in obstructions_before])

    report = {
        "cycle_id": cycle_id,
        "version": config.version,
        "candidate_count_requested": config.candidate_count,
        "generated_candidates": len(candidates),
        "selected": len(selected),
        "expected_value": sum(candidate.expected_value for candidate in selected),
        "target_pair_distribution": dict(Counter(str(candidate.trial.predicted_target_pair) for candidate in selected)),
        "mutation_type_distribution": dict(Counter(candidate.trial.mutation_type for candidate in selected)),
        "law_trust_distribution": dict(Counter(law.trust_label for law in laws_before)),
        "obstruction_penalties_applied": sum(candidate.obstruction_penalty < 1 for candidate in selected),
        "novelty_rate": sum(candidate.trial.child_hash not in memory.known_child_hashes for candidate in selected) / max(1, len(selected)),
        "submission_bytes": len(submission_text.encode()),
        "submitted": False,
    }
    write_json(out / "run_report.json", report)
    write_json(cycle_dir / "run_report.json", report)
    print(json.dumps(report, indent=2))
    if not config.submit:
        print("recommended next action: inspect dry-run portfolio; set SAIR_SUBMIT=1 to submit")
        return report

    key = get_api_key(prompt=True)
    client = SairClient(key, base_url=config.api_base)
    response = client.submit(polynomials, f"MathGraph {config.version} cycle {cycle_id}")
    submission_id = str(response.get("submissionId") or response.get("id") or "")
    if not submission_id:
        raise RuntimeError("submission response lacks submissionId")
    ledger.register_submission([candidate.trial for candidate in selected], submission_id)
    report.update({"submitted": True, "submission_id": submission_id})
    if not config.poll:
        write_json(out / "run_report.json", report)
        write_json(cycle_dir / "run_report.json", report)
        return report

    final = client.poll(submission_id)
    write_json(out / "api_results.json", final)
    write_json(cycle_dir / "api_results.json", final)
    verified, failed = api_rows(final)
    for row in verified + failed:
        index = row.get("polynomialIndex", row.get("polynomial_index", row.get("index")))
        if index is not None:
            join_result(submission_id, int(index), row, ledger)

    verified_fields = sorted(set().union(*(row.keys() for row in verified))) if verified else ["polynomialIndex", "status", "computed_label", "computed_r"]
    failed_fields = sorted(set().union(*(row.keys() for row in failed))) if failed else ["polynomialIndex", "status", "reason"]
    for directory in (out, cycle_dir):
        write_csv(directory / "verified_from_api.csv", verified, verified_fields)
        write_csv(directory / "failed_from_api.csv", failed, failed_fields)

    actual_pairs: Counter[Pair] = Counter()
    for row in verified:
        label = str(row.get("computed_label", ""))
        digits = "".join(ch if ch.isdigit() else " " for ch in label).split()
        t = row.get("t", int(digits[-1]) if digits else None)
        r = row.get("r", row.get("computed_r"))
        if t is not None and r is not None:
            actual_pairs[(int(t), int(r))] += 1
    pair_summary = [
        {"t": pair[0], "r": pair[1], "count": count, "bad_attractor": pair in DEFAULT_HARMFUL_PAIRS}
        for pair, count in actual_pairs.items()
    ]
    for directory in (out, cycle_dir):
        write_csv(directory / "verified_pairs_summary.csv", pair_summary)

    trials_after = ledger.load()
    laws_after = learn_laws(trials_after)
    obstructions_after = learn_obstructions(trials_after)
    LawBook(tuple(laws_after), tuple(obstructions_after)).save(out / "updated_lawbook.json")
    LawBook(tuple(laws_after), tuple(obstructions_after)).save(cycle_dir / "updated_lawbook.json")
    write_json(out / "updated_obstructions.json", [asdict(item) for item in obstructions_after])
    write_json(cycle_dir / "updated_obstructions.json", [asdict(item) for item in obstructions_after])

    next_rows = []
    for parent in parents_from_trials(trials_after)[:20]:
        rec = recommend(parent, config.target_pair, laws_after, obstructions_after)
        next_rows.append({
            "recommendation_id": rec.recommendation_id,
            "parent_hash": rec.parent_hash,
            "law_id": rec.law.law_id,
            "expected_child_hash": rec.expected_child_hash,
            "posterior": rec.target_pair_posterior,
            "confidence_lower": rec.confidence_interval[0],
            "support": rec.support_count,
            "failures": rec.failure_count,
            "rationale": " | ".join(rec.rationale),
        })
    write_csv(out / "next_recommendations.csv", next_rows)
    write_csv(cycle_dir / "next_recommendations.csv", next_rows)

    before = {law.law_id: law for law in laws_before}
    report.update({
        "actual_pair_distribution": {str(pair): count for pair, count in actual_pairs.items()},
        "score_candidate_pairs": {str(pair): count for pair, count in actual_pairs.items() if pair not in DEFAULT_HARMFUL_PAIRS},
        "bad_attractor_pairs": {str(pair): count for pair, count in actual_pairs.items() if pair in DEFAULT_HARMFUL_PAIRS},
        "laws_strengthened": [law.law_id for law in laws_after if law.law_id not in before or law.success_count > before[law.law_id].success_count],
        "laws_weakened": [law.law_id for law in laws_after if law.law_id in before and law.failure_count > before[law.law_id].failure_count],
        "new_obstructions": len(obstructions_after) - len(obstructions_before),
        "recommended_next_action": "apply highest-evidence unobstructed laws in the next cycle",
    })
    write_json(out / "run_report.json", report)
    write_json(cycle_dir / "run_report.json", report)
    print(json.dumps(report, indent=2))
    return report


__all__ = ["CycleConfig", "run_cycle", "parents_from_trials", "build_candidate_pool"]
