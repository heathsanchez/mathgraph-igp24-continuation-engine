from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .leaderboard import HARD_BLACKLIST, LeaderboardContext, classify_pair


@dataclass(frozen=True)
class SurvivorObstruction:
    obstruction_id: str
    label: str
    evidence_count: int
    affected_basin_ids: tuple[str, ...]
    affected_constructor_families: tuple[str, ...]
    affected_pairs: tuple[str, ...]
    support_patterns: tuple[str, ...]
    severity: float
    suggested_escape_operator: str
    examples: tuple[str, ...]


ESCAPES = {
    "reciprocal_collapse": "reciprocal_breaker",
    "low_r_gravel": "high_r_lift",
    "crowded_attractor": "quotient_escape",
    "irreducibility_failure": "support_transport",
    "duplicate_signature": "virgin_support_probe",
    "baseline_dead_zone": "quotient_escape",
    "overfit_known_lane": "separatrix_perturb",
    "false_virginity": "lane_exploit",
    "high_r_failure": "high_r_lift",
    "support_singularity": "support_transport",
}


def _pair(record: dict[str, Any]):
    if record.get("pair"):
        return tuple(record["pair"]) if not isinstance(record["pair"], str) else None
    if record.get("t") is not None and record.get("r") is not None:
        return int(record["t"]), int(record["r"])
    return None


def _record_labels(record: dict[str, Any], leaderboard: LeaderboardContext | None = None) -> set[str]:
    labels: set[str] = set()
    pair = _pair(record)
    family = str(record.get("constructor_family", record.get("mutation_type", ""))).lower()
    reciprocal = float(record.get("reciprocal_score", 0) or 0)
    status = str(record.get("status", "")).lower()
    reason = str(record.get("scoringReason", record.get("api_reason", ""))).lower()
    if pair in {(25000, 0), (25000, 2), (25000, 4), (25000, 6), (25000, 8), (24979, 0), (24979, 2), (24979, 4), (24979, 6), (24979, 8)} and (reciprocal > 0.7 or "reciprocal" in family):
        labels.add("reciprocal_collapse")
    if pair and pair[1] in {0, 4} and (pair in HARD_BLACKLIST or (leaderboard and classify_pair(pair, leaderboard).target_value <= 10)):
        labels.add("low_r_gravel")
    if pair and (pair in HARD_BLACKLIST or (leaderboard and classify_pair(pair, leaderboard).score_class in {"CROWDED", "DEAD"})):
        labels.add("crowded_attractor")
    if "reduc" in reason or "irreduc" in reason or ("fail" in status and "factor" in reason):
        labels.add("irreducibility_failure")
    if str(record.get("scoreable", "")).lower() == "false" or "baseline" in reason:
        labels.add("baseline_dead_zone")
    if pair and leaderboard and pair not in HARD_BLACKLIST and classify_pair(pair, leaderboard).score_class == "DEAD":
        labels.add("false_virginity")
    if pair and pair[1] < 12 and "high_r" in family:
        labels.add("high_r_failure")
    return labels


def learn_obstruction_atlas(
    records: Sequence[dict[str, Any]],
    basin_summary: Sequence[dict[str, Any]] | None = None,
    leaderboard: LeaderboardContext | None = None,
) -> list[SurvivorObstruction]:
    evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_pairs_by_run: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        pair = _pair(record)
        if pair:
            seen_pairs_by_run[str(record.get("run_id", ""))][pair] += 1
        for label in _record_labels(record, leaderboard):
            evidence[label].append(record)
    for run_id, counts in seen_pairs_by_run.items():
        duplicate_pairs = {pair for pair, count in counts.items() if count > 1}
        for record in records:
            if str(record.get("run_id", "")) == run_id and _pair(record) in duplicate_pairs:
                evidence["duplicate_signature"].append(record)
    for basin in basin_summary or []:
        if float(basin.get("survival_rate", 0) or 0) > 0.3 and float(basin.get("crowded_pair_share", 0) or 0) > 0.5:
            evidence["overfit_known_lane"].append({"basin_id": basin["basin_id"], "constructor_family": basin.get("constructor_family", "unknown"), "support": basin.get("support_pattern", ""), "line": ""})
        if float(basin.get("high_r_score", 0) or 0) == 0 and int(basin.get("count_total", 0) or 0) >= 3:
            evidence["high_r_failure"].append({"basin_id": basin["basin_id"], "constructor_family": basin.get("constructor_family", "unknown"), "support": basin.get("support_pattern", ""), "line": ""})
        pair_counts = basin.get("pair_counts", {})
        if isinstance(pair_counts, dict) and pair_counts:
            top = max(pair_counts.values())
            if top / max(1, sum(pair_counts.values())) > 0.7:
                evidence["support_singularity"].append({"basin_id": basin["basin_id"], "constructor_family": basin.get("constructor_family", "unknown"), "support": basin.get("support_pattern", ""), "line": ""})
    result = []
    for label, rows in sorted(evidence.items()):
        if not rows:
            continue
        basins = tuple(sorted({str(row.get("basin_id", "")) for row in rows if row.get("basin_id")}))
        families = tuple(sorted({str(row.get("constructor_family", row.get("mutation_type", "unknown"))) for row in rows}))
        pairs = tuple(sorted({str(_pair(row)) for row in rows if _pair(row)}))
        supports = tuple(sorted({str(row.get("support", "")) for row in rows if row.get("support")})[:20])
        severity = min(1.0, len(rows) / max(3, len(records)))
        result.append(SurvivorObstruction(
            obstruction_id=f"OBS-{label}-{abs(hash((label, basins, pairs))) % 10_000_000:07d}",
            label=label,
            evidence_count=len(rows),
            affected_basin_ids=basins,
            affected_constructor_families=families,
            affected_pairs=pairs,
            support_patterns=supports,
            severity=severity,
            suggested_escape_operator=ESCAPES.get(label, "separatrix_perturb"),
            examples=tuple(str(row.get("line", "")) for row in rows[:5]),
        ))
    return sorted(result, key=lambda item: (item.severity, item.evidence_count), reverse=True)


def write_obstruction_atlas(path_root: str | Path, obstructions: Sequence[SurvivorObstruction]) -> None:
    root = Path(path_root)
    root.mkdir(parents=True, exist_ok=True)
    rows = [asdict(item) for item in obstructions]
    fields = list(rows[0].keys()) if rows else ["obstruction_id", "label", "evidence_count", "severity"]
    with (root / "obstruction_atlas.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (root / "obstruction_atlas.json").write_text(json.dumps(rows, indent=2, sort_keys=True, default=str), encoding="utf-8")


__all__ = ["SurvivorObstruction", "learn_obstruction_atlas", "write_obstruction_atlas"]
