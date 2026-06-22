from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .models import BasinLaw, Evidence, Obstruction


def evidence_from_dict(row: dict[str, Any]) -> Evidence:
    return Evidence(
        successes=int(row.get("successes", 0)), failures=int(row.get("failures", 0)),
        confidence_lower=float(row.get("confidence_lower", 0.0)),
        confidence_upper=float(row.get("confidence_upper", 1.0)),
        trial_ids=tuple(map(str, row.get("trial_ids", []))),
        replay_successes=int(row.get("replay_successes", 0)),
        replay_failures=int(row.get("replay_failures", 0)),
    )


def law_from_dict(row: dict[str, Any]) -> BasinLaw:
    return BasinLaw(
        law_id=str(row["law_id"]), source_basin=str(row["source_basin"]),
        target_basin=str(row["target_basin"]), target_pair=tuple(map(int, row["target_pair"])),
        coefficient_deltas=tuple(map(int, row["coefficient_deltas"])),
        preconditions=dict(row.get("preconditions", {})), evidence=evidence_from_dict(row.get("evidence", {})),
        trust_label=str(row.get("trust_label", "EMPIRICAL_LAW")), version=str(row.get("version", "1")),
    )


def obstruction_from_dict(row: dict[str, Any]) -> Obstruction:
    return Obstruction(
        obstruction_id=str(row["obstruction_id"]), source_basin=str(row["source_basin"]),
        mutation_signature=str(row["mutation_signature"]), harmful_pair=tuple(map(int, row["harmful_pair"])),
        support=int(row["support"]), failure_rate=float(row["failure_rate"]), confidence=float(row["confidence"]),
        example_trial_ids=tuple(map(str, row.get("example_trial_ids", []))),
        trust_label=str(row.get("trust_label", "NAMED_OBSTRUCTION")),
    )


@dataclass(frozen=True)
class LawBook:
    laws: tuple[BasinLaw, ...]
    obstructions: tuple[Obstruction, ...] = ()
    schema_version: str = "1"

    @classmethod
    def load(cls, path: str | Path) -> "LawBook":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            laws=tuple(law_from_dict(row) for row in payload.get("laws", [])),
            obstructions=tuple(obstruction_from_dict(row) for row in payload.get("obstructions", [])),
            schema_version=str(payload.get("schema_version", "1")),
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({
            "schema_version": self.schema_version,
            "laws": [asdict(law) for law in self.laws],
            "obstructions": [asdict(item) for item in self.obstructions],
        }, indent=2, sort_keys=True), encoding="utf-8")

