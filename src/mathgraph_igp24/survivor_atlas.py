from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from .leaderboard import HARD_BLACKLIST, LeaderboardContext, classify_pair
from .polynomial import parse_poly as _parse_poly, poly_to_line, valid_poly

Polynomial = tuple[int, ...]
Pair = tuple[int, int]


def parse_poly(value: str | Iterable[int]) -> Polynomial:
    return _parse_poly(value)


def support_tuple(poly: Polynomial) -> tuple[int, ...]:
    return tuple(index for index, value in enumerate(poly) if value)


def sign_pattern(poly: Polynomial) -> str:
    return "".join("+" if value > 0 else "-" if value < 0 else "0" for value in poly)


def parity_signature(poly: Polynomial) -> str:
    support = support_tuple(poly)
    even = sum(index % 2 == 0 for index in support)
    odd = len(support) - even
    if odd == 0:
        family = "even_only"
    elif even == 0:
        family = "odd_only"
    else:
        family = "mixed"
    return f"{family}:E{even}:O{odd}"


def coefficient_shape(poly: Polynomial) -> str:
    height = max(abs(value) for value in poly)
    return f"support={len(support_tuple(poly))}:height={height_bucket(poly)}:sign={sign_pattern(poly)}"


def reciprocal_score(poly: Polynomial) -> float:
    diffs = [abs(poly[index] - poly[24 - index]) for index in range(25)]
    scale = sum(abs(value) for value in poly) + 1
    return max(0.0, 1.0 - (sum(diffs) / scale))


def palindromic_score(poly: Polynomial) -> float:
    return reciprocal_score(poly)


def asymmetry_score(poly: Polynomial) -> float:
    return 1.0 - reciprocal_score(poly)


def central_mass(poly: Polynomial) -> float:
    total = sum(abs(value) for value in poly) + 1
    return sum(abs(poly[index]) for index in range(8, 17)) / total


def height_bucket(poly: Polynomial) -> str:
    height = max(abs(value) for value in poly)
    if height <= 4:
        return "h<=4"
    if height <= 16:
        return "h<=16"
    if height <= 64:
        return "h<=64"
    if height <= 256:
        return "h<=256"
    return "h>256"


def support_gap_signature(poly: Polynomial) -> str:
    support = support_tuple(poly)
    gaps = tuple(right - left for left, right in zip(support, support[1:]))
    return ".".join(map(str, gaps)) or "singleton"


def symmetry_class(poly: Polynomial) -> str:
    rec = reciprocal_score(poly)
    if rec >= 0.95:
        return "reciprocal"
    if rec >= 0.75:
        return "near_reciprocal"
    return "asymmetric"


def family_signature(row: dict[str, Any]) -> str:
    for key in ("constructor_family", "constructor_name", "mutation_type", "tag", "family"):
        if row.get(key):
            return str(row[key])
    return "unknown"


def basin_signature(poly: Polynomial, constructor_family: str = "unknown") -> tuple[Any, ...]:
    return (
        len(support_tuple(poly)),
        parity_signature(poly),
        support_gap_signature(poly),
        symmetry_class(poly),
        height_bucket(poly),
        round(central_mass(poly), 1),
        constructor_family,
    )


def basin_id_from_signature(signature: tuple[Any, ...]) -> str:
    import hashlib

    raw = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    return "SB" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _read_csv(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except UnicodeError:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            return list(csv.DictReader(handle))


def _read_json_any(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def _rows_from_json(path: Path) -> list[dict[str, Any]]:
    payload = _read_json_any(path)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("verifiedPolynomials", "failedPolynomials", "rows", "data", "selected", "candidates"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        return [payload]
    return []


def _pair_from_row(row: dict[str, Any]) -> Optional[Pair]:
    t = row.get("t", row.get("T", row.get("api_t")))
    r = row.get("r", row.get("R", row.get("api_r", row.get("computed_r"))))
    label = str(row.get("computed_label", row.get("api_label", row.get("label", ""))))
    if t is None and label:
        import re

        digits = re.findall(r"\d+", label)
        if digits:
            t = digits[-1]
    try:
        if t is not None and r is not None:
            return int(t), int(r)
    except (TypeError, ValueError):
        return None
    return None


def _line_from_row(row: dict[str, Any]) -> Optional[str]:
    for key in ("line", "polynomial", "poly", "coefficients"):
        if row.get(key):
            try:
                return poly_to_line(parse_poly(row[key]))
            except Exception:
                return str(row[key])
    return None


def _status_from_row(row: dict[str, Any], pair: Optional[Pair]) -> str:
    raw = str(row.get("status", row.get("outcome_status", ""))).lower()
    if pair:
        return "accepted"
    if "fail" in raw or "reject" in raw or row.get("reason"):
        return "rejected"
    if "queue" in raw:
        return "queued"
    return "unknown"


def _record_from_line(run_id: str, index: int, line: str, source: Path, row: dict[str, Any] | None = None) -> dict[str, Any]:
    row = row or {}
    poly = parse_poly(line)
    support = support_tuple(poly)
    family = family_signature(row)
    signature = basin_signature(poly, family)
    pair = _pair_from_row(row)
    status = _status_from_row(row, pair)
    return {
        "run_id": run_id,
        "polynomial_index": int(row.get("polynomial_index", row.get("polynomialIndex", row.get("index", index)))),
        "line": poly_to_line(poly),
        "coefficients": poly,
        "support": support,
        "support_count": len(support),
        "parity_signature": parity_signature(poly),
        "symmetry_class": symmetry_class(poly),
        "constructor_family": family,
        "mutation_type": row.get("mutation_type", family),
        "tag": row.get("tag", ""),
        "height": max(abs(value) for value in poly),
        "height_bucket": height_bucket(poly),
        "c12": poly[12],
        "center_mass": central_mass(poly),
        "reciprocal_score": reciprocal_score(poly),
        "palindromic_score": palindromic_score(poly),
        "asymmetry_score": asymmetry_score(poly),
        "odd_support_count": sum(index % 2 == 1 for index in support),
        "even_support_count": sum(index % 2 == 0 for index in support),
        "support_gap_signature": support_gap_signature(poly),
        "basin_signature": signature,
        "basin_id": basin_id_from_signature(signature),
        "t": pair[0] if pair else None,
        "r": pair[1] if pair else None,
        "pair": pair,
        "status": status,
        "accepted": status == "accepted",
        "scoringStatus": row.get("scoringStatus", ""),
        "scoreable": row.get("scoreable", ""),
        "scoringReason": row.get("scoringReason", row.get("reason", "")),
        "discriminant": row.get("discriminant", row.get("D", "")),
        "source_path": str(source),
    }


def _ingest_submission(path: Path) -> list[dict[str, Any]]:
    run_id = path.parent.name
    rows = []
    for index, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        stripped = line.strip()
        if stripped:
            try:
                rows.append(_record_from_line(run_id, index, stripped, path))
            except Exception:
                continue
    return rows


def _ingest_tabular(path: Path) -> list[dict[str, Any]]:
    run_id = path.parent.name
    rows = _rows_from_json(path) if path.suffix.lower() == ".json" else _read_csv(path)
    records = []
    for index, row in enumerate(rows):
        line = _line_from_row(row)
        if not line:
            continue
        try:
            records.append(_record_from_line(run_id, index, line, path, row))
        except Exception:
            continue
    return records


def load_survivor_records(root: str | Path) -> list[dict[str, Any]]:
    base = Path(root)
    if not base.exists():
        return []
    wanted = {
        "submission.txt", "selected_meta.json", "verified_joined.json", "trials.jsonl",
        "submission_final.json", "selected_top100.csv", "candidate_pool.csv",
    }
    records: dict[tuple[str, int, str], dict[str, Any]] = {}
    for path in base.rglob("*"):
        if not path.is_file() or path.name not in wanted:
            continue
        if path.name == "trials.jsonl":
            rows = []
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            tmp = []
            for index, row in enumerate(rows):
                poly = row.get("child_polynomial", row.get("child_poly"))
                if poly:
                    row = dict(row)
                    row.setdefault("polynomial", ",".join(map(str, poly)))
                    row.setdefault("index", row.get("polynomial_index", index))
                    tmp.append(row)
            loaded = []
            for index, row in enumerate(tmp):
                line = _line_from_row(row)
                if line:
                    loaded.append(_record_from_line(path.parent.name, index, line, path, row))
        elif path.name == "submission.txt":
            loaded = _ingest_submission(path)
        else:
            loaded = _ingest_tabular(path)
        for record in loaded:
            key = (record["run_id"], int(record["polynomial_index"]), record["line"])
            previous = records.get(key, {})
            merged = {**previous, **{k: v for k, v in record.items() if v not in (None, "", "unknown")}}
            records[key] = merged
    return list(records.values())


def summarize_basins(records: list[dict[str, Any]], leaderboard: LeaderboardContext | None = None) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["basin_id"]].append(record)
    result = []
    for basin_id, rows in grouped.items():
        pairs = [row["pair"] for row in rows if row.get("pair")]
        verified = [row for row in rows if row.get("accepted")]
        failed = [row for row in rows if row.get("status") == "rejected"]
        pair_counts = Counter(map(str, pairs))
        banned = sum(pair in HARD_BLACKLIST for pair in pairs)
        crowded = 0
        if leaderboard:
            crowded = sum(classify_pair(pair, leaderboard).target_value <= 10 for pair in pairs)
        unique_pairs = len(set(pairs))
        high_r = sum((pair[1] in {12, 16, 18, 20, 24}) for pair in pairs)
        novelty = unique_pairs / max(1, len(pairs))
        result.append({
            "basin_id": basin_id,
            "count_total": len(rows),
            "count_verified": len(verified),
            "count_failed": len(failed),
            "survival_rate": len(verified) / max(1, len(rows)),
            "pair_counts": dict(pair_counts),
            "t_counts": dict(Counter(row["t"] for row in rows if row.get("t") is not None)),
            "r_counts": dict(Counter(row["r"] for row in rows if row.get("r") is not None)),
            "unique_pairs": unique_pairs,
            "unique_t": len({pair[0] for pair in pairs}),
            "unique_r": len({pair[1] for pair in pairs}),
            "crowded_pair_share": crowded / max(1, len(pairs)),
            "banned_pair_share": banned / max(1, len(pairs)),
            "novelty_score": novelty,
            "high_r_score": high_r / max(1, len(pairs)),
            "low_k_proxy_score": novelty * (1 - banned / max(1, len(pairs))),
            "obstruction_labels": [],
            "top_examples": [row["line"] for row in rows[:3]],
            "constructor_family": Counter(row.get("constructor_family", "unknown") for row in rows).most_common(1)[0][0],
            "support_pattern": str(rows[0].get("support")),
        })
    return sorted(result, key=lambda row: (row["low_k_proxy_score"], row["survival_rate"], row["unique_pairs"]), reverse=True)


def write_survivor_atlas(root: str | Path, records: list[dict[str, Any]], basin_summary: list[dict[str, Any]]) -> None:
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    record_fields = [
        "run_id", "polynomial_index", "line", "support", "support_count", "parity_signature",
        "symmetry_class", "constructor_family", "height", "c12", "center_mass", "reciprocal_score",
        "palindromic_score", "odd_support_count", "even_support_count", "t", "r", "pair", "status",
        "scoringStatus", "scoreable", "scoringReason", "discriminant", "basin_id", "source_path",
    ]
    with (base / "survivor_records.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=record_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    summary_fields = list(basin_summary[0].keys()) if basin_summary else ["basin_id", "count_total"]
    with (base / "basin_atlas.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(basin_summary)
    (base / "basin_summary.json").write_text(json.dumps(basin_summary, indent=2, sort_keys=True, default=str), encoding="utf-8")


def build_survivor_atlas(root: str | Path, output_root: str | Path | None = None, leaderboard: LeaderboardContext | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = load_survivor_records(root)
    summary = summarize_basins(records, leaderboard)
    if output_root:
        write_survivor_atlas(output_root, records, summary)
    return records, summary


__all__ = [
    "build_survivor_atlas", "load_survivor_records", "summarize_basins", "write_survivor_atlas",
    "parse_poly", "support_tuple", "sign_pattern", "parity_signature", "coefficient_shape",
    "reciprocal_score", "palindromic_score", "asymmetry_score", "central_mass",
    "height_bucket", "support_gap_signature", "family_signature", "basin_signature",
    "basin_id_from_signature",
]
