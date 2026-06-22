from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from .fingerprints import basin_id
from .models import (
    MutationTrial,
    Outcome,
    Pair,
    Polynomial,
    TRUST_GENERATED,
    TRUST_HELD_OUT_REPLICATION,
    TRUST_OBSERVED_TRIAL,
)
from .polynomial import hash_poly, normalize_poly


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def create_trial(
    parent: Sequence[int],
    child: Sequence[int],
    mutation_type: str,
    mutation_parameters: dict[str, Any],
    target_pair: Optional[Pair],
    *,
    pair_before: Optional[Pair] = None,
    generator_version: str = "v102",
    random_seed: int = 0,
) -> MutationTrial:
    parent_poly, child_poly = normalize_poly(parent), normalize_poly(child)
    payload = {
        "parent_hash": hash_poly(parent_poly),
        "child_hash": hash_poly(child_poly),
        "mutation_type": mutation_type,
        "mutation_parameters": mutation_parameters,
        "generator_version": generator_version,
        "random_seed": random_seed,
        "target_pair": target_pair,
    }
    trial_id = "TR" + hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()[:24]
    return MutationTrial(
        trial_id=trial_id,
        parent_hash=payload["parent_hash"],
        child_hash=payload["child_hash"],
        parent_poly=parent_poly,
        child_poly=child_poly,
        source_basin=basin_id(parent_poly),
        destination_basin=basin_id(child_poly),
        mutation_type=mutation_type,
        mutation_parameters=dict(mutation_parameters),
        target_pair=target_pair,
        pair_before=pair_before,
        pair_after=None,
        submission_id="",
        polynomial_index=-1,
        status="GENERATED",
        generator_version=generator_version,
        random_seed=random_seed,
        trust_label=TRUST_GENERATED,
        timestamp=utc_now(),
    )


def trial_from_dict(row: dict[str, Any]) -> MutationTrial:
    def pair(name: str) -> Optional[Pair]:
        value = row.get(name)
        return tuple(map(int, value)) if isinstance(value, (list, tuple)) and len(value) == 2 else None

    return MutationTrial(
        trial_id=str(row["trial_id"]), parent_hash=str(row["parent_hash"]), child_hash=str(row["child_hash"]),
        parent_poly=tuple(map(int, row["parent_poly"])), child_poly=tuple(map(int, row["child_poly"])),
        source_basin=str(row["source_basin"]), destination_basin=str(row["destination_basin"]),
        mutation_type=str(row["mutation_type"]), mutation_parameters=dict(row.get("mutation_parameters", {})),
        target_pair=pair("target_pair"), pair_before=pair("pair_before"), pair_after=pair("pair_after"),
        submission_id=str(row.get("submission_id", "")), polynomial_index=int(row.get("polynomial_index", -1)),
        status=str(row.get("status", "GENERATED")), generator_version=str(row.get("generator_version", "v102")),
        random_seed=int(row.get("random_seed", 0)), trust_label=str(row.get("trust_label", TRUST_GENERATED)),
        timestamp=str(row.get("timestamp", "")), outcome=dict(row.get("outcome", {})),
    )


class TrialLedger:
    """Append-only JSONL ledger; later records supersede earlier trial states."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, trial: MutationTrial) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(stable_json(trial.to_dict()) + "\n")

    def load(self) -> list[MutationTrial]:
        latest: dict[str, MutationTrial] = {}
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    trial = trial_from_dict(json.loads(line))
                    latest[trial.trial_id] = trial
        return list(latest.values())

    def find(self, submission_id: str, polynomial_index: int) -> MutationTrial:
        matches = [
            trial for trial in self.load()
            if trial.submission_id == submission_id and trial.polynomial_index == polynomial_index
        ]
        if not matches:
            raise KeyError(f"no trial for submission={submission_id!r}, index={polynomial_index}")
        return matches[-1]


def parse_outcome(value: Outcome | dict[str, Any]) -> Outcome:
    if isinstance(value, Outcome):
        return value
    pair_value = value.get("pair")
    if pair_value is None:
        t = value.get("t", value.get("computed_t"))
        r = value.get("r", value.get("computed_r"))
        if t is None and value.get("computed_label") is not None:
            labels = re.findall(r"\d+", str(value["computed_label"]))
            t = int(labels[-1]) if labels else None
        pair_value = (int(t), int(r)) if t is not None and r is not None else None
    pair = tuple(map(int, pair_value)) if pair_value is not None else None
    return Outcome(status=str(value.get("status", "unknown")), pair=pair, payload=dict(value))


def join_result(
    submission_id: str,
    polynomial_index: int,
    outcome: Outcome | dict[str, Any],
    ledger: TrialLedger | str | Path | None = None,
) -> MutationTrial:
    if ledger is None:
        ledger = TrialLedger(Path("artifacts") / "trials.jsonl")
    elif not isinstance(ledger, TrialLedger):
        ledger = TrialLedger(ledger)
    trial = ledger.find(submission_id, polynomial_index)
    parsed = parse_outcome(outcome)
    is_replay = "law_id" in trial.mutation_parameters
    observed = replace(
        trial,
        pair_after=parsed.pair,
        status=parsed.status,
        trust_label=TRUST_HELD_OUT_REPLICATION if is_replay else TRUST_OBSERVED_TRIAL,
        timestamp=utc_now(),
        outcome=parsed.payload,
    )
    ledger.append(observed)
    return observed
