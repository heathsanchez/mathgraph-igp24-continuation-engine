from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .leaderboard import LeaderboardContext
from .obstruction_atlas import SurvivorObstruction


@dataclass(frozen=True)
class Route:
    route_id: str
    source_basin_id: str
    constructor_family: str
    target_profile: str
    avoid_obstructions: tuple[str, ...]
    route_score: float
    evidence: dict[str, Any]
    intended_escape: str


def _boundary_bonus(basin: dict[str, Any]) -> float:
    unique_t = int(basin.get("unique_t", 0) or 0)
    unique_r = int(basin.get("unique_r", 0) or 0)
    survival = float(basin.get("survival_rate", 0) or 0)
    banned = float(basin.get("banned_pair_share", 0) or 0)
    if unique_t >= 2 or unique_r >= 2 or (survival > 0 and banned > 0):
        return 1.5
    return 1.0


def recommend_routes(
    basin_summary: Sequence[dict[str, Any]],
    leaderboard_context: LeaderboardContext,
    obstructions: Sequence[SurvivorObstruction] = (),
    target_count: int = 100,
    mode: str = "mixed",
) -> list[Route]:
    obstruction_by_basin: dict[str, list[SurvivorObstruction]] = {}
    for obstruction in obstructions:
        for basin_id in obstruction.affected_basin_ids:
            obstruction_by_basin.setdefault(basin_id, []).append(obstruction)
    routes: list[Route] = []
    for basin in basin_summary:
        basin_id = str(basin.get("basin_id", "unknown"))
        survival = float(basin.get("survival_rate", 0) or 0)
        novelty = float(basin.get("novelty_score", 0.5) or 0.5)
        low_k = float(basin.get("low_k_proxy_score", 0.5) or 0.5)
        high_r = float(basin.get("high_r_score", 0) or 0)
        crowded = float(basin.get("crowded_pair_share", 0) or 0)
        banned = float(basin.get("banned_pair_share", 0) or 0)
        boundary = _boundary_bonus(basin)
        attached = obstruction_by_basin.get(basin_id, [])
        penalty = 1.0
        for obstruction in attached:
            penalty *= max(0.05, 1.0 - obstruction.severity)
        diversity = 1.0 / max(1.0, float(basin.get("count_total", 1) or 1) ** 0.25)
        mode_bonus = 1.0
        if mode == "high_r":
            mode_bonus += high_r
        elif mode == "virgin":
            mode_bonus += novelty
        elif mode == "lane_24648":
            mode_bonus += 0.5 if "24648" in json.dumps(basin.get("pair_counts", {})) else 0.0
        score = (survival + 0.1) * (novelty + 0.1) * (low_k + 0.1) * (high_r + 0.2)
        score *= diversity * penalty * boundary * mode_bonus
        score *= max(0.01, 1.0 - crowded) * max(0.01, 1.0 - banned)
        family = str(basin.get("constructor_family", "unknown"))
        target = "high_r_low_k" if high_r else "virgin_boundary" if boundary > 1 else "survivor_escape"
        escape = attached[0].suggested_escape_operator if attached else ("high_r_lift" if mode == "high_r" else "virgin_support_probe")
        routes.append(Route(
            route_id=f"ROUTE-{basin_id}-{len(routes):04d}",
            source_basin_id=basin_id,
            constructor_family=family,
            target_profile=target,
            avoid_obstructions=tuple(item.label for item in attached),
            route_score=score,
            evidence={
                "survival_rate": survival,
                "novelty_score": novelty,
                "low_k_proxy_score": low_k,
                "high_r_score": high_r,
                "diversity_score": diversity,
                "obstruction_penalty": penalty,
                "boundary_bonus": boundary,
                "crowded_pair_share": crowded,
                "banned_pair_share": banned,
            },
            intended_escape=escape,
        ))
    if not routes:
        routes.append(Route("ROUTE-seed-0000", "seed", "virgin_support_probe", "virgin_boundary", (), 1.0, {"fallback": True}, "virgin_support_probe"))
    return sorted(routes, key=lambda route: route.route_score, reverse=True)[:target_count]


def write_routes(root: str | Path, routes: Sequence[Route]) -> None:
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    rows = [asdict(route) for route in routes]
    fields = list(rows[0].keys()) if rows else ["route_id", "route_score"]
    with (path / "route_scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    boundaries = [row for row in rows if float(row["evidence"].get("boundary_bonus", 1)) > 1]
    with (path / "phase_boundaries.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(boundaries)
    (path / "selected_routes.json").write_text(json.dumps(rows, indent=2, sort_keys=True, default=str), encoding="utf-8")


__all__ = ["Route", "recommend_routes", "write_routes"]
