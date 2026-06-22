from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from .fingerprints import basin_id, fingerprint
from .models import MutationTrial, Outcome, Pair, TRUST_GENERATED, TRUST_OBSERVED_TRIAL, TRUST_SUBMITTED
from .polynomial import hash_poly, normalize_poly


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _fingerprint_delta(before: tuple[Any, ...], after: tuple[Any, ...]) -> dict[str, Any]:
    names = ("support", "height_bin", "even_support", "odd_support", "center_index", "center_peak", "support_pattern", "max_gap", "entropy", "center_mass", "lacunarity")
    result: dict[str, Any] = {}
    for name, left, right in zip(names, before, after):
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            result[name] = right - left
        elif left != right:
            result[name] = {"before": left, "after": right}
    return result


def create_trial(
    parent: Sequence[int], child: Sequence[int], mutation_type: str,
    mutation_parameters: dict[str, Any], target_pair: Optional[Pair], *,
    pair_before: Optional[Pair] = None, generator_version: str = "v102",
    random_seed: int = 0, cycle_id: str = "",
) -> MutationTrial:
    parent_poly, child_poly = normalize_poly(parent), normalize_poly(child)
    before, after = fingerprint(parent_poly), fingerprint(child_poly)
    payload = {
        "parent_hash": hash_poly(parent_poly), "child_hash": hash_poly(child_poly),
        "mutation_type": mutation_type, "mutation_parameters": mutation_parameters,
        "target_pair": target_pair, "generator_version": generator_version,
        "random_seed": random_seed, "cycle_id": cycle_id,
    }
    now = utc_now()
    return MutationTrial(
        trial_id="TR" + hashlib.sha256(stable_json(payload).encode()).hexdigest()[:24],
        parent_hash=payload["parent_hash"], child_hash=payload["child_hash"],
        parent_polynomial=parent_poly, child_polynomial=child_poly,
        source_basin=basin_id(parent_poly), predicted_target_pair=target_pair,
        mutation_type=mutation_type, mutation_parameters=dict(mutation_parameters),
        coefficient_delta=tuple(b - a for a, b in zip(parent_poly, child_poly)),
        fingerprint_before=before, fingerprint_after=after,
        fingerprint_delta=_fingerprint_delta(before, after),
        generator_version=generator_version, random_seed=random_seed, cycle_id=cycle_id,
        submission_id="", polynomial_index=-1, pair_before=pair_before, pair_after=None,
        outcome_status="GENERATED", api_label="", api_t=None, api_r=None, api_reason="",
        trust_label=TRUST_GENERATED, created_at=now, updated_at=now,
    )


def trial_from_dict(row: dict[str, Any]) -> MutationTrial:
    def pair(name: str) -> Optional[Pair]:
        value = row.get(name)
        return tuple(map(int, value)) if isinstance(value, (list, tuple)) and len(value) == 2 else None
    parent = row.get("parent_polynomial", row.get("parent_poly"))
    child = row.get("child_polynomial", row.get("child_poly"))
    return MutationTrial(
        trial_id=str(row["trial_id"]), parent_hash=str(row["parent_hash"]), child_hash=str(row["child_hash"]),
        parent_polynomial=tuple(map(int, parent)), child_polynomial=tuple(map(int, child)),
        source_basin=str(row["source_basin"]), predicted_target_pair=pair("predicted_target_pair") or pair("target_pair"),
        mutation_type=str(row["mutation_type"]), mutation_parameters=dict(row.get("mutation_parameters", {})),
        coefficient_delta=tuple(map(int, row.get("coefficient_delta", []))),
        fingerprint_before=tuple(row.get("fingerprint_before", [])), fingerprint_after=tuple(row.get("fingerprint_after", [])),
        fingerprint_delta=dict(row.get("fingerprint_delta", {})), generator_version=str(row.get("generator_version", "v102")),
        random_seed=int(row.get("random_seed", 0)), cycle_id=str(row.get("cycle_id", "")),
        submission_id=str(row.get("submission_id", "")), polynomial_index=int(row.get("polynomial_index", -1)),
        pair_before=pair("pair_before"), pair_after=pair("pair_after"),
        outcome_status=str(row.get("outcome_status", row.get("status", "GENERATED"))),
        api_label=str(row.get("api_label", "")), api_t=row.get("api_t"), api_r=row.get("api_r"),
        api_reason=str(row.get("api_reason", "")), trust_label=str(row.get("trust_label", TRUST_GENERATED)),
        created_at=str(row.get("created_at", row.get("timestamp", ""))), updated_at=str(row.get("updated_at", row.get("timestamp", ""))),
    )


class TrialLedger:
    """Append-only JSONL ledger. Every state update is durable and replayable."""
    def __init__(self, path: str | Path): self.path = Path(path)

    def append(self, trial: MutationTrial) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(stable_json(trial.to_dict()) + "\n")

    def append_many(self, trials: Sequence[MutationTrial]) -> None:
        for trial in trials: self.append(trial)

    def load(self) -> list[MutationTrial]:
        latest: dict[str, MutationTrial] = {}
        if not self.path.exists(): return []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                trial = trial_from_dict(json.loads(line)); latest[trial.trial_id] = trial
        return list(latest.values())

    def find(self, submission_id: str, polynomial_index: int) -> MutationTrial:
        matches = [t for t in self.load() if t.submission_id == submission_id and t.polynomial_index == polynomial_index]
        if not matches: raise KeyError(f"no trial for submission={submission_id!r}, index={polynomial_index}")
        return matches[-1]

    def register_submission(self, trials: Sequence[MutationTrial], submission_id: str) -> list[MutationTrial]:
        registered = []
        for index, trial in enumerate(trials):
            updated = replace(trial, submission_id=submission_id, polynomial_index=index,
                              outcome_status="SUBMITTED", trust_label=TRUST_SUBMITTED, updated_at=utc_now())
            self.append(updated); registered.append(updated)
        return registered


def parse_outcome(value: Outcome | dict[str, Any]) -> Outcome:
    if isinstance(value, Outcome): return value
    t = value.get("t", value.get("computed_t")); r = value.get("r", value.get("computed_r"))
    label = str(value.get("computed_label", value.get("label", "")))
    if t is None and label:
        numbers = re.findall(r"\d+", label); t = int(numbers[-1]) if numbers else None
    pair = (int(t), int(r)) if t is not None and r is not None else None
    return Outcome(str(value.get("status", "unknown")), pair, label, str(value.get("reason", "")), dict(value))


def join_result(submission_id: str, polynomial_index: int, outcome: Outcome | dict[str, Any], ledger: TrialLedger | str | Path | None = None) -> MutationTrial:
    store = ledger if isinstance(ledger, TrialLedger) else TrialLedger(ledger or Path("artifacts") / "trials.jsonl")
    trial = store.find(submission_id, polynomial_index); parsed = parse_outcome(outcome); now = utc_now()
    observed = replace(
        trial, pair_after=parsed.pair, outcome_status=parsed.status, api_label=parsed.label,
        api_t=parsed.pair[0] if parsed.pair else None, api_r=parsed.pair[1] if parsed.pair else None,
        api_reason=parsed.reason, trust_label=TRUST_OBSERVED_TRIAL, updated_at=now,
    )
    store.append(observed); return observed
