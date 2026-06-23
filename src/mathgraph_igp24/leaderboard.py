from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

Pair = tuple[int, int]

SCORE_VIRGIN = "VIRGIN_CANDIDATE"
SCORE_LOW_K = "LOW_K"
SCORE_MID_K = "MID_K"
SCORE_CROWDED = "CROWDED"
SCORE_DEAD = "DEAD"
SCORE_BASELINE = "BASELINE"
SCORE_UNKNOWN = "UNKNOWN"

HARD_BLACKLIST: set[Pair] = {
    (25000, 0), (25000, 2), (25000, 4), (25000, 6), (25000, 8),
    (24979, 0), (24979, 2), (24979, 4), (24979, 6), (24979, 8),
    (24970, 0), (24970, 4),
    (9993, 0), (9993, 4),
    (143, 0), (143, 4),
    (6763, 0), (6763, 4),
    (21844, 0), (21844, 4),
    (52, 0), (53, 0), (54, 4), (68, 0), (45, 0), (38, 0),
    (1475, 0), (2711, 0), (2711, 4), (687, 0), (688, 4), (689, 0),
}


@dataclass(frozen=True)
class PairValue:
    pair: Pair
    k_estimate: Optional[int]
    public_score: Optional[float]
    score_class: str
    target_value: float
    estimated: bool = False


@dataclass
class LeaderboardContext:
    known_pairs: set[Pair]
    pair_k: dict[Pair, int]
    pair_score: dict[Pair, float]
    known_t: set[int]
    discovered_t_count: int
    pair_density_by_t: dict[int, int]
    pair_density_by_r: dict[int, int]
    blacklist: set[Pair]
    low_k_pairs: set[Pair]
    source_paths: list[str]

    def value(self, pair: Optional[Pair]) -> PairValue:
        return classify_pair(pair, self)


def parse_pair(value: Any) -> Optional[Pair]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
    text = str(value)
    numbers = re.findall(r"\d+", text)
    if len(numbers) >= 2:
        return int(numbers[-2]), int(numbers[-1])
    return None


def _row_pair(row: dict[str, Any]) -> Optional[Pair]:
    if "pair" in row:
        pair = parse_pair(row["pair"])
        if pair:
            return pair
    t = row.get("t", row.get("T", row.get("computed_t")))
    r = row.get("r", row.get("R", row.get("computed_r")))
    label = row.get("computed_label", row.get("label", row.get("group", "")))
    if t is None and label:
        digits = re.findall(r"\d+", str(label))
        if digits:
            t = digits[-1]
    try:
        if t is not None and r is not None:
            return int(t), int(r)
    except (TypeError, ValueError):
        return None
    return parse_pair(label)


def _float(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _rows_from_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("rows", "data", "leaderboard", "submissions", "pairs"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        return [payload]
    return []


def _rows_from_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def find_leaderboard_files(root: str | Path, explicit: str | Path | None = None) -> list[Path]:
    if explicit:
        path = Path(explicit)
        return [path] if path.exists() else []
    roots = [Path(root), Path(root) / "leaderboard", Path("/content/drive/MyDrive/MathGraph_IGP24")]
    names = {"igp24_leaderboard.csv", "igp24_leaderboard.json", "igp24_leaderboard_by_member.csv"}
    paths: list[Path] = []
    for base in roots:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and (path.name in names or ("leaderboard" in path.name.lower() and path.suffix.lower() in {".csv", ".json"})):
                paths.append(path)
    return sorted(dict.fromkeys(paths))


def estimate_k_from_score(score: Optional[float]) -> Optional[int]:
    if score is None or score <= 0:
        return None
    return max(1, int(round(1 - math.log(score, 2))))


def load_leaderboard_context(root: str | Path, leaderboard_path: str | Path | None = None) -> LeaderboardContext:
    pair_score: dict[Pair, float] = {}
    pair_k: dict[Pair, int] = {}
    source_paths: list[str] = []
    for path in find_leaderboard_files(root, leaderboard_path):
        source_paths.append(str(path))
        rows = _rows_from_json(path) if path.suffix.lower() == ".json" else _rows_from_csv(path)
        for row in rows:
            pair = _row_pair(row)
            if not pair:
                continue
            score = _float(row.get("score", row.get("public_score", row.get("pair_score"))))
            k_raw = row.get("k", row.get("count", row.get("pair_k")))
            try:
                k = int(k_raw) if k_raw not in (None, "") else estimate_k_from_score(score)
            except (TypeError, ValueError):
                k = estimate_k_from_score(score)
            if score is not None:
                pair_score[pair] = max(score, pair_score.get(pair, 0.0))
            if k is not None:
                pair_k[pair] = max(1, min(k, pair_k.get(pair, k)))
    known_pairs = set(pair_score) | set(pair_k)
    known_t = {pair[0] for pair in known_pairs}
    density_t: dict[int, int] = {}
    density_r: dict[int, int] = {}
    for t, r in known_pairs:
        density_t[t] = density_t.get(t, 0) + 1
        density_r[r] = density_r.get(r, 0) + 1
    low_k = {pair for pair, k in pair_k.items() if k <= 3}
    return LeaderboardContext(
        known_pairs=known_pairs,
        pair_k=pair_k,
        pair_score=pair_score,
        known_t=known_t,
        discovered_t_count=len(known_t),
        pair_density_by_t=density_t,
        pair_density_by_r=density_r,
        blacklist=set(HARD_BLACKLIST),
        low_k_pairs=low_k,
        source_paths=source_paths,
    )


def classify_pair(pair: Optional[Pair], context: LeaderboardContext | None = None) -> PairValue:
    if pair is None:
        return PairValue((-1, -1), None, None, SCORE_UNKNOWN, 100.0)
    context = context or LeaderboardContext(set(), {}, {}, set(), 0, {}, {}, set(HARD_BLACKLIST), set(), [])
    if pair in context.blacklist:
        return PairValue(pair, context.pair_k.get(pair), context.pair_score.get(pair), SCORE_DEAD, -1000.0)
    if pair[0] == 1 or pair == (25000, 0):
        return PairValue(pair, context.pair_k.get(pair), context.pair_score.get(pair), SCORE_BASELINE, -500.0)
    if pair not in context.known_pairs:
        return PairValue(pair, None, None, SCORE_VIRGIN, 1000.0)
    k = context.pair_k.get(pair)
    score = context.pair_score.get(pair)
    estimated = False
    if k is None:
        k = estimate_k_from_score(score)
        estimated = k is not None
    if k is None:
        return PairValue(pair, None, score, SCORE_UNKNOWN, 300.0)
    if k <= 1:
        return PairValue(pair, k, score, SCORE_LOW_K, 1000.0, estimated)
    if k == 2:
        return PairValue(pair, k, score, SCORE_LOW_K, 500.0, estimated)
    if k == 3:
        return PairValue(pair, k, score, SCORE_MID_K, 250.0, estimated)
    if k <= 8:
        return PairValue(pair, k, score, SCORE_CROWDED, max(10.0, 250.0 / (2 ** (k - 3))), estimated)
    return PairValue(pair, k, score, SCORE_DEAD, 0.0, estimated)


def context_rows(context: LeaderboardContext) -> list[dict[str, Any]]:
    return [
        {
            "t": pair[0],
            "r": pair[1],
            "k_estimate": context.pair_k.get(pair),
            "public_score": context.pair_score.get(pair),
            "score_class": context.value(pair).score_class,
            "target_value": context.value(pair).target_value,
        }
        for pair in sorted(context.known_pairs | context.blacklist)
    ]


__all__ = ["PairValue", "LeaderboardContext", "HARD_BLACKLIST", "load_leaderboard_context", "classify_pair", "context_rows"]
