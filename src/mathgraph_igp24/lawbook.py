from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .models import BasinLaw, Evidence, Obstruction


def evidence_from_dict(row: dict[str, Any]) -> Evidence:
    return Evidence(int(row.get("successes", 0)), int(row.get("failures", 0)),
                    float(row.get("confidence_lower", 0)), float(row.get("confidence_upper", 1)),
                    tuple(row.get("trial_ids", [])), int(row.get("replay_successes", 0)), int(row.get("replay_failures", 0)))


def law_from_dict(row: dict[str, Any]) -> BasinLaw:
    return BasinLaw(
        law_id=row["law_id"], source_basin=row["source_basin"], target_basin=row["target_basin"],
        target_pair=tuple(row["target_pair"]), mutation_type=row["mutation_type"],
        coefficient_deltas=tuple(row["coefficient_deltas"]), preconditions=dict(row.get("preconditions", {})),
        evidence=evidence_from_dict(row.get("evidence", {})), mutation_parameters_summary=dict(row.get("mutation_parameters_summary", {})),
        coefficient_delta_mean=tuple(row.get("coefficient_delta_mean", [])), coefficient_delta_median=tuple(row.get("coefficient_delta_median", [])),
        coefficient_delta_mode=tuple(row.get("coefficient_delta_mode", [])), fingerprint_delta_mean=dict(row.get("fingerprint_delta_mean", {})),
        fingerprint_delta_median=dict(row.get("fingerprint_delta_median", {})), success_rate=float(row.get("success_rate", 0)),
        lift_over_baseline=float(row.get("lift_over_baseline", 0)), matched_failure_count=int(row.get("matched_failure_count", 0)),
        novelty_score=float(row.get("novelty_score", 1)), submission_ids=tuple(row.get("submission_ids", [])),
        cycle_ids=tuple(row.get("cycle_ids", [])), trust_label=row.get("trust_label", "GENERATED"), version=row.get("version", "2"),
    )


def obstruction_from_dict(row: dict[str, Any]) -> Obstruction:
    return Obstruction(
        obstruction_id=row["obstruction_id"], source_basin=row["source_basin"], mutation_type=row["mutation_type"],
        harmful_pair=tuple(row["harmful_pair"]), feature_signature=row["feature_signature"],
        failure_count=int(row["failure_count"]), matched_success_count=int(row["matched_success_count"]),
        confidence=float(row["confidence"]), avoided_by=tuple(row.get("avoided_by", [])),
        example_trial_ids=tuple(row.get("example_trial_ids", [])), trust_label=row.get("trust_label", "OBSTRUCTED"),
    )


@dataclass(frozen=True)
class LawBook:
    laws: tuple[BasinLaw, ...]; obstructions: tuple[Obstruction, ...] = (); schema_version: str = "2"

    @classmethod
    def load(cls, path: str | Path) -> "LawBook":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(tuple(law_from_dict(row) for row in payload.get("laws", [])),
                   tuple(obstruction_from_dict(row) for row in payload.get("obstructions", [])), str(payload.get("schema_version", "2")))

    def save(self, path: str | Path) -> None:
        target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"schema_version": self.schema_version, "laws": [asdict(law) for law in self.laws],
                                      "obstructions": [asdict(item) for item in self.obstructions]}, indent=2, sort_keys=True), encoding="utf-8")

