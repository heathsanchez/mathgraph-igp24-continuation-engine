from __future__ import annotations

import csv
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .constructors import CONSTRUCTORS, ConstructedCandidate, choose_constructor
from .leaderboard import LeaderboardContext, classify_pair
from .polynomial import poly_to_line, valid_poly
from .submission import MAX_BYTES, MAX_POLYNOMIALS, build_submission
from .survivor_atlas import basin_id_from_signature, basin_signature, height_bucket, support_tuple
from .survivor_router import Route

Polynomial = tuple[int, ...]


@dataclass(frozen=True)
class SurvivorCandidate:
    constructed: ConstructedCandidate
    route: Route
    predicted_pair: tuple[int, int] | None
    predicted_score_class: str
    route_score: float
    novelty_score: float
    estimated_height: int

    @property
    def line(self) -> str:
        return poly_to_line(self.constructed.child)

    @property
    def child_hash(self) -> str:
        return self.constructed.child_hash

    @property
    def constructor_name(self) -> str:
        return self.constructed.constructor_name

    def to_meta(self, index: int | None = None) -> dict[str, Any]:
        row = self.constructed.to_dict()
        row.update({
            "index": index,
            "line": self.line,
            "route_id": self.route.route_id,
            "route_score": self.route_score,
            "predicted_pair": self.predicted_pair,
            "predicted_score_class": self.predicted_score_class,
            "novelty_score": self.novelty_score,
            "support": support_tuple(self.constructed.child),
            "support_count": len(support_tuple(self.constructed.child)),
            "basin_id": basin_id_from_signature(basin_signature(self.constructed.child, self.constructor_name)),
            "height_bucket": height_bucket(self.constructed.child),
            "obstruction_avoided": self.constructed.obstruction_avoided,
            "trust_label": "EMPIRICAL_ROUTE",
        })
        return row


def seed_parent() -> Polynomial:
    return tuple([1, 1] + [0] * 22 + [1])


def parents_from_records(records: Sequence[dict[str, Any]], limit: int = 50) -> list[Polynomial]:
    parents: list[Polynomial] = []
    seen: set[str] = set()
    for record in reversed(records):
        coeffs = record.get("coefficients")
        if coeffs and record.get("line") not in seen:
            poly = tuple(int(value) for value in coeffs)
            if valid_poly(poly):
                parents.append(poly)
                seen.add(record.get("line"))
        if len(parents) >= limit:
            break
    return parents or [seed_parent()]


def constructor_for_route(route: Route, index: int, mode: str) -> str:
    if mode == "high_r":
        return "high_r_lift"
    if mode == "virgin":
        return "virgin_support_probe"
    if mode == "lane_24648":
        return "lane_exploit"
    if mode == "discriminant_minimize":
        return "discriminant_minimizer"
    if mode == "mixed":
        cycle = [
            "quotient_escape", "reciprocal_breaker", "asymmetry_injection", "support_transport",
            "high_r_lift", "parity_bridge", "separatrix_perturb", "virgin_support_probe",
            "lane_exploit", "discriminant_minimizer", "boundary_microshift", "odd_lane_probe",
            "height_trim_probe", "wide_support_probe", "center_shear_probe", "mixed_parity_probe",
            "quotient_jitter_probe", "separatrix_wide_probe", "asymmetric_height_probe",
        ]
        return cycle[index % len(cycle)]
    if route.intended_escape in CONSTRUCTORS:
        return route.intended_escape
    cycle = ["quotient_escape", "reciprocal_breaker", "asymmetry_injection", "support_transport", "high_r_lift", "parity_bridge", "separatrix_perturb", "virgin_support_probe", "lane_exploit"]
    return cycle[index % len(cycle)]


def generate_survivor_candidates(
    parents: Sequence[Polynomial],
    routes: Sequence[Route],
    leaderboard: LeaderboardContext,
    candidate_count: int,
    seed: int = 109,
    mode: str = "mixed",
) -> list[SurvivorCandidate]:
    rng = random.Random(seed)
    parents = list(parents or [seed_parent()])
    routes = list(routes or [])
    if not routes:
        from .survivor_router import Route

        routes = [Route("ROUTE-seed", "seed", "virgin_support_probe", "virgin_boundary", (), 1.0, {"fallback": True}, "virgin_support_probe")]
    result: list[SurvivorCandidate] = []
    seen: set[str] = set()
    for index in range(candidate_count):
        route = routes[index % len(routes)]
        parent = parents[(index // max(1, len(routes))) % len(parents)]
        name = constructor_for_route(route, index, mode)
        constructor = choose_constructor(name)
        built = constructor.generate(parent, rng, seed + index, route.target_profile)
        if built.child_hash in seen:
            continue
        seen.add(built.child_hash)
        predicted_pair = None
        if name == "lane_exploit":
            predicted_pair = (24648 if index % 2 == 0 else 16055, 4 if index % 3 else 2)
        pair_value = classify_pair(predicted_pair, leaderboard)
        novelty = 1.0 if predicted_pair is None or predicted_pair not in leaderboard.known_pairs else 0.2
        score = route.route_score * max(0.01, pair_value.target_value / 1000.0) * novelty
        result.append(SurvivorCandidate(
            constructed=built,
            route=route,
            predicted_pair=predicted_pair,
            predicted_score_class=pair_value.score_class,
            route_score=score,
            novelty_score=novelty,
            estimated_height=max(abs(value) for value in built.child),
        ))
    return result


def select_survivor_portfolio(candidates: Sequence[SurvivorCandidate], limit: int = MAX_POLYNOMIALS, lane_exploit: bool = False) -> list[SurvivorCandidate]:
    selected: list[SurvivorCandidate] = []
    lines: set[str] = set()
    predicted_pairs: set[tuple[int, int]] = set()
    support_counts: Counter = Counter()
    family_counts: Counter = Counter()
    basin_counts: Counter = Counter()
    bytes_used = 0
    family_cap = 20 if lane_exploit else 8
    for candidate in sorted(candidates, key=lambda item: (item.route_score, -item.estimated_height), reverse=True):
        if not valid_poly(candidate.constructed.child):
            continue
        if candidate.predicted_score_class in {"CROWDED", "DEAD", "BASELINE"}:
            continue
        line = candidate.line
        support = support_tuple(candidate.constructed.child)
        basin = basin_id_from_signature(basin_signature(candidate.constructed.child, candidate.constructor_name))
        encoded = len((line + "\n").encode("utf-8"))
        if line in lines or bytes_used + encoded > MAX_BYTES:
            continue
        if candidate.predicted_pair and candidate.predicted_pair in predicted_pairs:
            continue
        if support_counts[support] >= 2 or family_counts[candidate.constructor_name] >= family_cap or basin_counts[basin] >= 10:
            continue
        selected.append(candidate)
        lines.add(line)
        if candidate.predicted_pair:
            predicted_pairs.add(candidate.predicted_pair)
        support_counts[support] += 1
        family_counts[candidate.constructor_name] += 1
        basin_counts[basin] += 1
        bytes_used += encoded
        if len(selected) >= limit:
            break
    return selected


def write_portfolio(root: str | Path, selected: Sequence[SurvivorCandidate], candidates: Sequence[SurvivorCandidate]) -> dict[str, Any]:
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    api_dir = path / "api_submit_cleaned"
    api_dir.mkdir(parents=True, exist_ok=True)
    text = build_submission([item.constructed.child for item in selected])
    (path / "submission.txt").write_text(text, encoding="utf-8")
    (api_dir / "submitted_valid_polys.txt").write_text(text, encoding="utf-8")
    selected_meta = [candidate.to_meta(index) for index, candidate in enumerate(selected)]
    (path / "selected_meta.json").write_text(json.dumps(selected_meta, indent=2, sort_keys=True, default=str), encoding="utf-8")
    pool_rows = [candidate.to_meta(index) for index, candidate in enumerate(candidates)]
    fields = list(pool_rows[0].keys()) if pool_rows else ["index", "line", "constructor_name"]
    with (path / "candidate_pool.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(pool_rows)
    report = {
        "selected": len(selected),
        "candidate_pool": len(candidates),
        "constructor_distribution": dict(Counter(item.constructor_name for item in selected)),
        "support_distribution": {str(key): value for key, value in Counter(support_tuple(item.constructed.child) for item in selected).items()},
        "basin_distribution": dict(Counter(item.to_meta()["basin_id"] for item in selected)),
        "obstruction_avoidance_count": dict(Counter(item.constructed.obstruction_avoided for item in selected)),
        "predicted_score_class_distribution": dict(Counter(item.predicted_score_class for item in selected)),
        "novelty_score": sum(item.novelty_score for item in selected) / max(1, len(selected)),
        "expected_high_r_share": sum("high_r" in item.route.target_profile for item in selected) / max(1, len(selected)),
        "expected_low_k_share": sum(item.predicted_score_class in {"VIRGIN_CANDIDATE", "LOW_K"} for item in selected) / max(1, len(selected)),
        "constructor_rationale": {name: CONSTRUCTORS[name].expected_effect for name in sorted({item.constructor_name for item in selected}) if name in CONSTRUCTORS},
    }
    (path / "portfolio_report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return report


__all__ = ["SurvivorCandidate", "generate_survivor_candidates", "select_survivor_portfolio", "write_portfolio", "parents_from_records"]
