# START FULL SCRIPT
# MathGraph IGP24 v101 — Provenance-Bearing Continuation Engine

"""Colab-ready basin-atlas candidate farmer for degree-24 IGP24 submissions.

The program learns empirical fingerprint basins from earlier runs, estimates
pair posteriors and transitions, generates one million candidates in a
bounded-memory stream, and writes a diversified 100-polynomial portfolio.
Empirical labels remain observations; local validation does not claim proof.
"""

from __future__ import annotations

import csv
import getpass
import hashlib
import json
import math
import os
import random
import re
import statistics
import time
from datetime import datetime, timezone
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from heapq import heappush, heapreplace
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

try:
    import requests
except Exception:
    requests = None


ROOT = Path(os.environ.get("MATHGRAPH_ROOT", "/content/drive/MyDrive/MathGraph_IGP24"))
OUT = ROOT / "v101_provenance_bearing_continuation_engine"
API_OUT = OUT / "api_submit_cleaned"
LEDGER = OUT / "trials.jsonl"
CYCLES_DIR = OUT / "cycles"

DEGREE = 24
COEFFICIENT_COUNT = 25
MAX_SUBMISSIONS = 100
MAX_BYTES = 100_000
GENERATION_COUNT = int(os.environ.get("MATHGRAPH_CANDIDATES", "1000000"))
SEED = int(os.environ.get("MATHGRAPH_SEED", "782130100"))
RNG = random.Random(SEED)
TARGET_PAIR = (14010, 8)
GENERATOR_VERSION = "v101"
LAW_ROUTING_SHARE = float(os.environ.get("MATHGRAPH_LAW_ROUTING_SHARE", "0.80"))
COMPETITION_ID = os.environ.get("SAIR_COMPETITION_ID", "igp24")
API_BASE = os.environ.get("SAIR_API_BASE", "https://api.sair.foundation/api/public/v1").rstrip("/")
AUTORUN = os.environ.get("SAIR_AUTORUN", "0") == "1"
MAX_CYCLES = int(os.environ.get("MATHGRAPH_MAX_CYCLES", "1"))
CYCLE_SLEEP_SECONDS = int(os.environ.get("MATHGRAPH_CYCLE_SLEEP_SECONDS", "900"))
POLL_SECONDS = int(os.environ.get("MATHGRAPH_POLL_SECONDS", "30"))
MAX_POLLS = int(os.environ.get("MATHGRAPH_MAX_POLLS", "160"))

RUN_DIRS = [
    "v46_targeted_score_candidate_exploiter",
    "v47_basin_fingerprint_cartographer",
    "v44_score_candidate_basin_harvester",
    "v43_actual_winner_basin_exploiter",
    "v42_empirical_label_router_farmer",
    "v41_score_aware_pair_suppression_farmer",
    "v40_verified_basin_replication_farmer",
    "v39_novel_pair_suppression_farmer",
    "v38_hot_basin_amplifier_point_farmer",
    "v38_basin_amplifier",
    "v37_klpb_shadow_1000pt_one_shot",
    "v36a_fast_transition_factory",
    "v33_transition_rarity_explorer",
    "v30_root_basin_lawbook_amplifier",
]
INPUT_NAMES = [
    "verified_from_api.csv", "submission_final.json",
    "submitted_valid_polys.txt", "selected_top100.csv",
    "candidate_pool.csv", "submission.txt",
]

POSITIVE_WEIGHTS = {
    (14010, 8): 8.0, (7208, 8): 8.0, (24970, 20): 20.0,
    (24970, 24): 24.0, (25000, 16): 16.0, (24970, 16): 16.0,
    (24970, 12): 12.0,
}
NEGATIVE_PAIRS = {
    (24979, 2), (24979, 4), (24979, 6), (24979, 8),
    (25000, 2), (25000, 4), (25000, 6), (25000, 8),
}
FAMILY_WEIGHTS = [
    ("basin_guided", 0.50), ("basin_clone", 0.20),
    ("transition_mutation", 0.15), ("structured", 0.10),
    ("random_exploration", 0.05),
]
PAIR_RE = re.compile(r"[-+]?\d+")


def banner(text: str) -> None:
    print("\n" + "=" * 76)
    print(text)
    print("=" * 76)


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    API_OUT.mkdir(parents=True, exist_ok=True)
    CYCLES_DIR.mkdir(parents=True, exist_ok=True)


def safe_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        if isinstance(value, float) and not value.is_integer():
            return None
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def poly_to_line(poly: Sequence[int]) -> str:
    return ",".join(str(int(x)) for x in poly)


def parse_poly(value: Any) -> Optional[tuple[int, ...]]:
    if isinstance(value, (list, tuple)):
        nums = [safe_int(x) for x in value]
    elif isinstance(value, str):
        nums = [safe_int(x) for x in PAIR_RE.findall(value)]
    else:
        return None
    if len(nums) != COEFFICIENT_COUNT or any(x is None for x in nums):
        return None
    return tuple(int(x) for x in nums if x is not None)


def row_poly(row: dict[str, Any]) -> Optional[tuple[int, ...]]:
    for key in ("poly", "polynomial", "coefficients", "coefficient_line", "line"):
        p = parse_poly(row.get(key))
        if p is not None:
            return p
    indexed = []
    for prefix in ("a", "c", "coef", "coeff"):
        indexed = [safe_int(row.get(f"{prefix}{i}")) for i in range(COEFFICIENT_COUNT)]
        if all(x is not None for x in indexed):
            return tuple(int(x) for x in indexed if x is not None)
    return None


def gcd_many(values: Iterable[int]) -> int:
    g = 0
    for value in values:
        g = math.gcd(g, abs(int(value)))
    return g


def valid_poly(poly: Sequence[int]) -> bool:
    return (
        len(poly) == COEFFICIENT_COUNT
        and all(isinstance(x, int) and not isinstance(x, bool) for x in poly)
        and poly[-1] == 1
        and poly[0] != 0
        and gcd_many(poly) == 1
    )


def normalize_poly(poly: Sequence[int]) -> Optional[tuple[int, ...]]:
    if len(poly) != COEFFICIENT_COUNT:
        return None
    p = [int(round(x)) for x in poly]
    p[-1] = 1
    if p[0] == 0:
        p[0] = RNG.choice((-1, 1))
    g = gcd_many(p)
    if g > 1:
        p = [x // g for x in p]
        p[-1] = 1
    result = tuple(p)
    return result if valid_poly(result) else None


def body_bytes(lines: Sequence[str]) -> int:
    return len(("\n".join(lines) + ("\n" if lines else "")).encode("utf-8"))


def line_hash(line: str) -> str:
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


def entropy(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    n = len(values)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def quantize(value: float, step: float, low: int = -999, high: int = 999) -> int:
    if not math.isfinite(value):
        return 0
    return max(low, min(high, int(round(value / step))))


def fingerprint(poly: Sequence[int]) -> tuple[Any, ...]:
    support = [i for i, x in enumerate(poly) if x]
    nonzero = [poly[i] for i in support]
    sparse = len(support)
    h = max(abs(x) for x in poly)
    even = sum(1 for i in support if i % 2 == 0)
    odd = sparse - even
    center = max(range(7, 18), key=lambda i: abs(poly[i]))
    center_peak = abs(poly[center]) / max(1, h)
    support_pattern = "".join("1" if x else "0" for x in poly)
    sign_pattern = "".join("+" if x > 0 else "-" if x < 0 else "0" for x in poly)
    magnitude_pattern = tuple(0 if x == 0 else min(7, int(math.log2(abs(x))) + 1) for x in poly)
    gaps = [b - a - 1 for a, b in zip(support, support[1:])]
    support_gaps = (max(gaps, default=0), sum(g > 0 for g in gaps), quantize(statistics.mean(gaps) if gaps else 0, 0.5))
    symmetry = sum(1 for i in range(12) if poly[i] == poly[24 - i]) / 12.0
    weights = [abs(x) for x in poly]
    total = sum(weights) or 1
    center_mass = sum(i * w for i, w in enumerate(weights)) / total
    lacunarity = (statistics.pvariance(weights) / ((statistics.mean(weights) or 1) ** 2)) if len(weights) > 1 else 0
    return (
        sparse, min(12, int(math.log10(h + 1))), even, odd, center,
        quantize(center_peak, 0.1, 0, 10), support_pattern,
        sign_pattern, magnitude_pattern, support_gaps,
        quantize(symmetry, 0.1, 0, 10), quantize(entropy(nonzero), 0.25, 0, 40),
        quantize(lacunarity, 0.5, 0, 40), quantize(center_mass, 1.0, 0, 24),
    )


def fingerprint_key(fp: tuple[Any, ...]) -> str:
    raw = json.dumps(fp, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(raw.encode("ascii")).hexdigest()[:16]


def coarse_basin_key(poly: Sequence[int]) -> tuple[Any, ...]:
    fp = fingerprint(poly)
    return (fp[0], fp[1], fp[2] // 3, fp[5], fp[9], fp[10], fp[12], fp[13])


def extract_pair(row: dict[str, Any]) -> Optional[tuple[int, int]]:
    key_pairs = [
        ("t", "r"), ("trace", "rank"), ("actual_t", "actual_r"),
        ("actualT", "actualR"), ("T", "R"), ("first", "second"),
        ("computed_t", "computed_r"), ("computedT", "computedR"),
    ]
    for ka, kb in key_pairs:
        a, b = safe_int(row.get(ka)), safe_int(row.get(kb))
        if a is not None and b is not None:
            return (a, b)
    for key in ("pair", "actual_pair", "label", "verdict"):
        nums = PAIR_RE.findall(str(row.get(key, "")))
        if len(nums) >= 2:
            return (int(nums[0]), int(nums[1]))
    label = str(row.get("computed_label", ""))
    r = safe_int(row.get("computed_r"))
    labels = PAIR_RE.findall(label)
    if labels and r is not None:
        return (int(labels[-1]), r)
    return None


def extract_index(row: dict[str, Any]) -> Optional[int]:
    for key in ("polynomialIndex", "polynomial_index", "index", "idx"):
        value = safe_int(row.get(key))
        if value is not None:
            return value
    return None


def candidate_paths(run_dir: str, name: str) -> list[Path]:
    base = ROOT / run_dir
    return [base / name, base / "api_submit_cleaned" / name]


def read_lines(path: Path) -> list[str]:
    try:
        return [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    except OSError:
        return []


def nearby_submissions(path: Path) -> list[tuple[int, ...]]:
    candidates = [
        path.parent / "submitted_valid_polys.txt",
        path.parent / "submission.txt",
        path.parent.parent / "submitted_valid_polys.txt",
        path.parent.parent / "submission.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            polys = [p for p in (parse_poly(x) for x in read_lines(candidate)) if p is not None]
            if polys:
                return polys
    return []


@dataclass
class VerifiedExample:
    poly: tuple[int, ...]
    pair: tuple[int, int]
    source: str
    index: int = -1
    sequence: int = -1


TRUST_GENERATED = "GENERATED"
TRUST_SUBMITTED = "SUBMITTED"
TRUST_OBSERVED_TRIAL = "OBSERVED_TRIAL"
TRUST_HELD_OUT_REPLICATION = "HELD_OUT_REPLICATION"
TRUST_EMPIRICAL_LAW = "EMPIRICAL_LAW"
TRUST_REPLICATED_LAW = "HELD_OUT_REPLICATED_LAW"
TRUST_NAMED_OBSTRUCTION = "NAMED_OBSTRUCTION"
TRUST_ROUTING_POLICY = "ROUTING_POLICY"


@dataclass
class MutationTrial:
    trial_id: str
    parent_poly: tuple[int, ...]
    child_poly: tuple[int, ...]
    parent_hash: str
    child_hash: str
    parent_basin: str
    child_basin: str
    mutation_type: str
    mutation_spec: dict[str, Any]
    coefficient_delta: tuple[int, ...]
    fingerprint_before: tuple[Any, ...]
    fingerprint_after: tuple[Any, ...]
    fingerprint_delta: dict[str, Any]
    target_pair: Optional[tuple[int, int]]
    pair_before: Optional[tuple[int, int]]
    pair_after: Optional[tuple[int, int]]
    successful: Optional[bool]
    submission_id: str
    polynomial_index: int
    cycle_id: str
    generator_version: str
    random_seed: int
    status: str = "CANDIDATE"
    trust_label: str = TRUST_GENERATED
    api_result: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    observed_at: str = ""


@dataclass
class BasinLaw:
    law_id: str
    source_basin: str
    target_basin: str
    mutation_type: str
    delta_signature: str
    coefficient_changes: dict[str, Any]
    fingerprint_changes: dict[str, Any]
    source_pair: Optional[tuple[int, int]]
    destination_pair: tuple[int, int]
    success_count: int
    failure_count: int
    trial_count: int
    posterior_mean: float
    confidence_lower: float
    confidence_upper: float
    submission_count: int
    examples: list[str]
    failure_examples: list[str]
    replay_successes: int
    replay_failures: int
    trust_label: str


@dataclass
class Obstruction:
    obstruction_id: str
    name: str
    source_pair: Optional[tuple[int, int]]
    destination_pair: tuple[int, int]
    source_basin: str
    mutation_type: str
    triggering_delta: str
    support: int
    trial_count: int
    failure_rate: float
    confidence: float
    examples: list[str]
    trust_label: str = TRUST_NAMED_OBSTRUCTION


@dataclass
class InverseRoute:
    route_id: str
    current_pair: Optional[tuple[int, int]]
    desired_pair: tuple[int, int]
    source_basin: str
    law_ids: list[str]
    estimated_probability: float
    known_obstructions: list[str]
    confidence: float
    trust_label: str = TRUST_ROUTING_POLICY


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def polynomial_hash(poly: Sequence[int]) -> str:
    return line_hash(poly_to_line(poly))


def fingerprint_delta(before: tuple[Any, ...], after: tuple[Any, ...]) -> dict[str, Any]:
    names = [
        "sparse_count", "height_bin", "even_count", "odd_count", "center_peak_index",
        "center_peak", "support_pattern", "sign_pattern", "magnitude_pattern", "support_gaps",
        "symmetry_score", "coefficient_entropy", "lacunarity", "center_mass",
    ]
    result: dict[str, Any] = {}
    for index, name in enumerate(names):
        left, right = before[index], after[index]
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            result[name] = right - left
        elif left != right:
            result[name] = {"before": left, "after": right}
    return result


def mutation_specification(parent: Sequence[int], child: Sequence[int], mutation_type: str) -> dict[str, Any]:
    edits = [
        {"coefficient": f"a{i}", "index": i, "before": int(a), "after": int(b), "delta": int(b - a)}
        for i, (a, b) in enumerate(zip(parent, child)) if a != b
    ]
    return {"mutation_type": mutation_type, "edits": edits, "edit_count": len(edits)}


def delta_signature(delta: Sequence[int]) -> str:
    changed = [(i, 1 if d > 0 else -1, min(6, int(math.log2(abs(d))) + 1)) for i, d in enumerate(delta) if d]
    return stable_json(changed)


def make_trial(parent: Sequence[int], child: Sequence[int], mutation_type: str, parent_basin: str, target_pair: Optional[tuple[int, int]], cycle_id: str, serial: int) -> MutationTrial:
    parent_t, child_t = tuple(parent), tuple(child)
    before, after = fingerprint(parent_t), fingerprint(child_t)
    spec = mutation_specification(parent_t, child_t, mutation_type)
    payload = {
        "parent_hash": polynomial_hash(parent_t), "child_hash": polynomial_hash(child_t),
        "mutation_spec": spec, "cycle_id": cycle_id, "serial": serial,
    }
    trial_id = "TR" + hashlib.sha256(stable_json(payload).encode()).hexdigest()[:20]
    return MutationTrial(
        trial_id=trial_id, parent_poly=parent_t, child_poly=child_t,
        parent_hash=payload["parent_hash"], child_hash=payload["child_hash"],
        parent_basin=parent_basin, child_basin="B" + hashlib.sha1(repr(coarse_basin_key(child_t)).encode()).hexdigest()[:10],
        mutation_type=mutation_type, mutation_spec=spec,
        coefficient_delta=tuple(b - a for a, b in zip(parent_t, child_t)),
        fingerprint_before=before, fingerprint_after=after,
        fingerprint_delta=fingerprint_delta(before, after), target_pair=target_pair,
        pair_before=None, pair_after=None, successful=None, submission_id="",
        polynomial_index=-1, cycle_id=cycle_id, generator_version=GENERATOR_VERSION,
        random_seed=SEED + serial, created_at=utc_now(),
    )


def is_observed_trial(trial: MutationTrial) -> bool:
    return trial.trust_label in {TRUST_OBSERVED_TRIAL, TRUST_HELD_OUT_REPLICATION}


@dataclass
class EmpiricalMemory:
    pair_counts: Counter = field(default_factory=Counter)
    known_submitted_lines: set[str] = field(default_factory=set)
    verified_examples: dict[tuple[int, int], list[VerifiedExample]] = field(default_factory=lambda: defaultdict(list))
    transition_counts: Counter = field(default_factory=Counter)
    actual_pair_history: list[tuple[int, int]] = field(default_factory=list)
    all_verified: list[VerifiedExample] = field(default_factory=list)


def load_csv_examples(path: Path, memory: EmpiricalMemory, seen: set[tuple[str, tuple[int, int]]]) -> None:
    submitted = nearby_submissions(path)
    try:
        with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return
    for sequence, row in enumerate(rows):
        pair = extract_pair(row)
        poly = row_poly(row)
        idx = extract_index(row)
        if poly is None and idx is not None and 0 <= idx < len(submitted):
            poly = submitted[idx]
        if pair is None:
            continue
        memory.pair_counts[pair] += 1
        memory.actual_pair_history.append(pair)
        if poly is None or not valid_poly(poly):
            continue
        marker = (poly_to_line(poly), pair)
        if marker in seen:
            continue
        seen.add(marker)
        ex = VerifiedExample(poly, pair, str(path.relative_to(ROOT)), idx if idx is not None else -1, sequence)
        memory.verified_examples[pair].append(ex)
        memory.all_verified.append(ex)


def walk_json_rows(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_json_rows(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_json_rows(child)


def load_json_examples(path: Path, memory: EmpiricalMemory, seen: set[tuple[str, tuple[int, int]]]) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, json.JSONDecodeError):
        return
    submitted = nearby_submissions(path)
    for sequence, row in enumerate(walk_json_rows(data)):
        pair = extract_pair(row)
        if pair is None:
            continue
        poly = row_poly(row)
        idx = extract_index(row)
        if poly is None and idx is not None and 0 <= idx < len(submitted):
            poly = submitted[idx]
        if poly is None or not valid_poly(poly):
            continue
        marker = (poly_to_line(poly), pair)
        if marker in seen:
            continue
        seen.add(marker)
        memory.pair_counts[pair] += 1
        memory.actual_pair_history.append(pair)
        ex = VerifiedExample(poly, pair, str(path.relative_to(ROOT)), idx if idx is not None else -1, sequence)
        memory.verified_examples[pair].append(ex)
        memory.all_verified.append(ex)


def collect_polys_from_csv(path: Path) -> Iterator[tuple[int, ...]]:
    try:
        with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as handle:
            for row in csv.DictReader(handle):
                p = row_poly(row)
                if p is not None and valid_poly(p):
                    yield p
    except (OSError, csv.Error):
        return


def load_empirical_memory() -> EmpiricalMemory:
    memory = EmpiricalMemory()
    seen_examples: set[tuple[str, tuple[int, int]]] = set()
    seen_paths: set[Path] = set()
    for run_dir in RUN_DIRS:
        for name in INPUT_NAMES:
            for path in candidate_paths(run_dir, name):
                if path in seen_paths or not path.exists():
                    continue
                seen_paths.add(path)
                if name == "verified_from_api.csv":
                    load_csv_examples(path, memory, seen_examples)
                elif name == "submission_final.json":
                    load_json_examples(path, memory, seen_examples)
                elif path.suffix == ".csv":
                    for poly in collect_polys_from_csv(path):
                        memory.known_submitted_lines.add(poly_to_line(poly))
                else:
                    for text in read_lines(path):
                        poly = parse_poly(text)
                        if poly is not None and valid_poly(poly):
                            memory.known_submitted_lines.add(poly_to_line(poly))
    for ex in memory.all_verified:
        memory.known_submitted_lines.add(poly_to_line(ex.poly))
    return memory


@dataclass
class Basin:
    basin_id: str
    fingerprint: tuple[Any, ...]
    members: list[VerifiedExample] = field(default_factory=list)
    pair_counts: Counter = field(default_factory=Counter)
    quality: float = 0.0
    examples: list[tuple[int, ...]] = field(default_factory=list)
    transitions: Counter = field(default_factory=Counter)

    @property
    def total(self) -> int:
        return sum(self.pair_counts.values())


def pair_value(pair: tuple[int, int]) -> float:
    if pair in NEGATIVE_PAIRS:
        return -max(8.0, float(pair[1]) * 2.0)
    return POSITIVE_WEIGHTS.get(pair, max(0.1, min(12.0, pair[1] / 2.0)))


def quality_from_counts(counts: Counter) -> float:
    total = sum(counts.values())
    if not total:
        return 0.0
    weighted = sum(pair_value(pair) * count for pair, count in counts.items())
    confidence = total / (total + 5.0)
    target_share = (counts[TARGET_PAIR] + 1.0) / (total + len(counts) + 2.0)
    bad_share = sum(count for pair, count in counts.items() if pair in NEGATIVE_PAIRS) / total
    return confidence * (weighted / total) * (0.75 + target_share) * (1.0 - 0.85 * bad_share)


def build_basins(memory: EmpiricalMemory, provenance_trials: Sequence[MutationTrial] = ()) -> dict[str, Basin]:
    basins: dict[str, Basin] = {}
    key_to_id: dict[tuple[Any, ...], str] = {}
    for ex in memory.all_verified:
        key = coarse_basin_key(ex.poly)
        basin_id = key_to_id.setdefault(key, "B" + hashlib.sha1(repr(key).encode()).hexdigest()[:10])
        basin = basins.setdefault(basin_id, Basin(basin_id, key))
        basin.members.append(ex)
        basin.pair_counts[ex.pair] += 1
        if len(basin.examples) < 24:
            basin.examples.append(ex.poly)
    for basin in basins.values():
        basin.quality = quality_from_counts(basin.pair_counts)
    for trial in provenance_trials:
        if not is_observed_trial(trial):
            continue
        a = key_to_id.get(coarse_basin_key(trial.parent_poly), trial.parent_basin)
        b = key_to_id.get(coarse_basin_key(trial.child_poly), trial.child_basin)
        if a in basins and b in basins and a != b:
            basins[a].transitions[b] += 1
    return basins


def pair_posteriors(basins: dict[str, Basin], memory: EmpiricalMemory) -> list[dict[str, Any]]:
    pair_totals = Counter(memory.pair_counts)
    rows = []
    all_pairs = sorted(pair_totals)
    for basin in basins.values():
        denom_b = basin.total + max(1, len(all_pairs))
        for pair in all_pairs:
            count = basin.pair_counts[pair]
            p_pair_given_basin = (count + 1.0) / denom_b
            p_basin_given_pair = (count + 1.0) / (pair_totals[pair] + len(basins))
            rows.append({
                "basin_id": basin.basin_id, "t": pair[0], "r": pair[1],
                "count": count, "p_pair_given_basin": p_pair_given_basin,
                "p_basin_given_pair": p_basin_given_pair,
            })
    return rows


def inverse_route(basins: dict[str, Basin], target_pair: tuple[int, int], top_n: int = 24) -> list[Basin]:
    def route_score(basin: Basin) -> float:
        posterior = (basin.pair_counts[target_pair] + 1.0) / (basin.total + len(basin.pair_counts) + 2.0)
        return basin.quality + 10.0 * posterior + math.log1p(basin.total)
    active = [b for b in basins.values() if b.examples and not all(p in NEGATIVE_PAIRS for p in b.pair_counts)]
    return sorted(active, key=route_score, reverse=True)[:top_n]


def clamp_coefficient(value: int, limit: int = 10**12) -> int:
    return max(-limit, min(limit, int(value)))


def random_delta(scale: int = 8) -> int:
    choices = [-scale, -scale // 2, -2, -1, 1, 2, scale // 2, scale]
    return RNG.choice([x for x in choices if x])


def mutate_micro(poly: Sequence[int], intensity: int = 1) -> tuple[int, ...]:
    p = list(poly)
    for _ in range(max(1, intensity)):
        i = RNG.randrange(0, DEGREE)
        p[i] = clamp_coefficient(p[i] + random_delta(max(2, min(1000, abs(p[i]) // 20 + 4))))
    return normalize_poly(p) or tuple(poly)


def mutate_a0_sweep(poly: Sequence[int]) -> tuple[int, ...]:
    p = list(poly)
    base = max(2, abs(p[0]) // 12 + 2)
    p[0] = clamp_coefficient(p[0] + random_delta(base))
    if p[0] == 0:
        p[0] = RNG.choice((-1, 1))
    return normalize_poly(p) or tuple(poly)


def mutate_sparse_keep_shape(poly: Sequence[int]) -> tuple[int, ...]:
    p = list(poly)
    support = [i for i in range(DEGREE) if p[i]] or [0]
    for _ in range(RNG.randint(1, 3)):
        i = RNG.choice(support)
        p[i] = clamp_coefficient(p[i] + random_delta(max(2, abs(p[i]) // 16 + 2)))
        if i == 0 and p[i] == 0:
            p[i] = 1
    return normalize_poly(p) or tuple(poly)


def mutate_basin_clone(poly: Sequence[int]) -> tuple[int, ...]:
    p = list(poly)
    scale_num, scale_den = RNG.choice([(9, 10), (19, 20), (21, 20), (11, 10)])
    for i in range(DEGREE):
        if p[i] and RNG.random() < 0.35:
            p[i] = clamp_coefficient(round(p[i] * scale_num / scale_den) + RNG.choice((-1, 0, 1)))
    return normalize_poly(p) or mutate_micro(poly)


def mutate_basin_guided(poly: Sequence[int], target_fp: tuple[Any, ...]) -> tuple[int, ...]:
    p = list(poly)
    desired_sparse = int(target_fp[0])
    support = [i for i in range(DEGREE) if p[i]]
    if len(support) > desired_sparse:
        removable = [i for i in support if i != 0]
        for i in RNG.sample(removable, min(len(removable), len(support) - desired_sparse)):
            p[i] = 0
    elif len(support) < desired_sparse:
        zeros = [i for i in range(1, DEGREE) if p[i] == 0]
        for i in RNG.sample(zeros, min(len(zeros), desired_sparse - len(support))):
            p[i] = RNG.choice((-1, 1)) * RNG.randint(1, 64)
    center_hint = target_fp[4] if len(target_fp) > 4 else 12
    center = min(23, max(1, int(center_hint) if isinstance(center_hint, (int, float)) else 12))
    if RNG.random() < 0.5:
        p[center] = clamp_coefficient(p[center] + random_delta(max(4, abs(p[center]) // 10 + 4)))
    return normalize_poly(p) or mutate_micro(poly)


def make_high_sparse(direct: bool = False) -> tuple[int, ...]:
    p = [0] * COEFFICIENT_COUNT
    p[-1] = 1
    p[0] = RNG.choice((-1, 1)) * RNG.randint(1, 10**5 if direct else 10**3)
    count = RNG.randint(4, 8)
    for i in RNG.sample(range(1, DEGREE), count - 2):
        p[i] = RNG.choice((-1, 1)) * RNG.randint(1, 10**6 if direct else 10**4)
    return normalize_poly(p) or tuple([1] + [0] * 23 + [1])


def make_center_peak(direct: bool = False) -> tuple[int, ...]:
    p = [0] * COEFFICIENT_COUNT
    p[-1] = 1
    p[0] = RNG.choice((-1, 1)) * RNG.randint(1, 500)
    center = RNG.randint(9, 15)
    p[center] = RNG.choice((-1, 1)) * RNG.randint(10**5 if direct else 10**3, 10**8 if direct else 10**6)
    for i in RNG.sample([x for x in range(1, DEGREE) if x != center], RNG.randint(2, 7)):
        p[i] = RNG.choice((-1, 1)) * RNG.randint(1, max(2, abs(p[center]) // 50))
    return normalize_poly(p) or make_high_sparse()


def make_even_lacunary(direct: bool = False) -> tuple[int, ...]:
    p = [0] * COEFFICIENT_COUNT
    p[-1] = 1
    p[0] = RNG.choice((-1, 1)) * RNG.randint(1, 1000)
    positions = RNG.sample(range(2, DEGREE, 2), RNG.randint(3, 8))
    for i in positions:
        p[i] = RNG.choice((-1, 1)) * RNG.randint(1, 10**7 if direct else 10**4)
    return normalize_poly(p) or make_high_sparse()


def random_exploration() -> tuple[int, ...]:
    p = [RNG.randint(-5000, 5000) if RNG.random() < 0.45 else 0 for _ in range(DEGREE)] + [1]
    if not p[0]:
        p[0] = RNG.choice((-1, 1)) * RNG.randint(1, 5000)
    return normalize_poly(p) or make_high_sparse()


def replay_law(poly: Sequence[int], law: BasinLaw) -> tuple[int, ...]:
    p = list(poly)
    for name, summary in law.coefficient_changes.items():
        index = safe_int(name[1:]) if name.startswith("a") else None
        if index is None or not 0 <= index < DEGREE:
            continue
        delta = summary.get("median", 0) if isinstance(summary, dict) else 0
        p[index] = clamp_coefficient(p[index] + int(round(float(delta))))
    return normalize_poly(p) or tuple(poly)


def apply_basin_law(poly: Sequence[int], law: BasinLaw) -> tuple[int, ...]:
    return replay_law(poly, law)


def transition_mutation(poly: Sequence[int], source: Basin, basins: dict[str, Basin]) -> tuple[int, ...]:
    choices = [(bid, count) for bid, count in source.transitions.items() if bid in basins]
    if not choices:
        return mutate_basin_guided(poly, source.fingerprint)
    target_id = RNG.choices([x[0] for x in choices], weights=[x[1] for x in choices], k=1)[0]
    target = basins[target_id]
    p = list(mutate_basin_guided(poly, target.fingerprint))
    exemplar = RNG.choice(target.examples)
    for i in RNG.sample(range(DEGREE), RNG.randint(2, 6)):
        if (p[i] == 0) != (exemplar[i] == 0) or RNG.random() < 0.4:
            p[i] = exemplar[i] if RNG.random() < 0.35 else (p[i] + exemplar[i]) // 2
    return normalize_poly(p) or mutate_micro(poly)


def atlas_walk(poly: Sequence[int], basin: Basin, basins: dict[str, Basin], steps: int = 2) -> tuple[int, ...]:
    p = tuple(poly)
    current = basin
    for _ in range(max(1, steps)):
        p = transition_mutation(p, current, basins)
        outgoing = [bid for bid in current.transitions if bid in basins]
        if outgoing:
            current = basins[RNG.choice(outgoing)]
    return p


@dataclass(order=True)
class Candidate:
    expected_value: float
    serial: int
    poly: tuple[int, ...] = field(compare=False)
    family: str = field(compare=False)
    basin_id: str = field(compare=False)
    expected_pair: tuple[int, int] = field(compare=False)
    expected_score: float = field(compare=False)
    novelty: float = field(compare=False)
    basin_quality: float = field(compare=False)
    diversity: float = field(compare=False)
    transition_bonus: float = field(compare=False)
    penalty: float = field(compare=False)
    fp_key: str = field(compare=False)
    trial: Optional[MutationTrial] = field(default=None, compare=False)


def predicted_pair(basin: Basin) -> tuple[int, int]:
    if not basin.pair_counts:
        return TARGET_PAIR
    return max(basin.pair_counts, key=lambda pair: (basin.pair_counts[pair], pair_value(pair)))


def evaluate_candidate(poly: tuple[int, ...], family: str, basin: Basin, memory: EmpiricalMemory, serial: int, trial: Optional[MutationTrial] = None) -> Candidate:
    fp = fingerprint(poly)
    fpkey = fingerprint_key(fp)
    pair = predicted_pair(basin)
    total = basin.total + len(basin.pair_counts) + 2
    posterior_score = sum((count + 1) * max(0.0, pair_value(p)) for p, count in basin.pair_counts.items()) / total
    expected_score = max(0.05, posterior_score + 0.35 * max(0.0, basin.quality))
    line = poly_to_line(poly)
    novelty = 0.08 if line in memory.known_submitted_lines else 1.0
    support_ratio = len(set(i for i, x in enumerate(poly) if x)) / COEFFICIENT_COUNT
    diversity = 0.75 + 0.5 * support_ratio + 0.05 * min(8.0, entropy([x for x in poly if x]))
    transition_bonus = 1.0 + min(0.35, math.log1p(sum(basin.transitions.values())) / 20.0)
    penalty = 1.0
    if pair in NEGATIVE_PAIRS:
        penalty *= 0.04
    if max(abs(x) for x in poly) > 10**11:
        penalty *= 0.75
    if fp[0] <= 2:
        penalty *= 0.5
    ev = expected_score * novelty * diversity * transition_bonus * penalty
    return Candidate(ev, serial, poly, family, basin.basin_id, pair, expected_score, novelty, basin.quality, diversity, transition_bonus, penalty, fpkey, trial)


class CandidateReservoir:
    def __init__(self, per_bucket: int = 1200, global_cap: int = 30000):
        self.per_bucket = per_bucket
        self.global_cap = global_cap
        self.heaps: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
        self.seen_lines: set[str] = set()
        self.generated = 0
        self.valid = 0

    def offer(self, candidate: Candidate) -> None:
        self.generated += 1
        if not valid_poly(candidate.poly):
            return
        line = poly_to_line(candidate.poly)
        if line in self.seen_lines:
            return
        self.valid += 1
        key = (candidate.family, candidate.basin_id)
        heap = self.heaps[key]
        if len(heap) < self.per_bucket:
            heappush(heap, candidate)
            self.seen_lines.add(line)
        elif candidate > heap[0]:
            removed = heapreplace(heap, candidate)
            self.seen_lines.discard(poly_to_line(removed.poly))
            self.seen_lines.add(line)

    def values(self) -> list[Candidate]:
        values = [c for heap in self.heaps.values() for c in heap]
        return sorted(values, reverse=True)[: self.global_cap]


def choose_family(u: float) -> str:
    cumulative = 0.0
    for family, weight in FAMILY_WEIGHTS:
        cumulative += weight
        if u < cumulative:
            return family
    return FAMILY_WEIGHTS[-1][0]


def structured_candidate(serial: int) -> tuple[tuple[int, ...], str]:
    mode = serial % 6
    if mode == 0:
        return make_high_sparse(False), "high_sparse"
    if mode == 1:
        return make_high_sparse(True), "high_sparse_direct"
    if mode == 2:
        return make_center_peak(False), "center_peak"
    if mode == 3:
        return make_center_peak(True), "center_peak_direct"
    if mode == 4:
        return make_even_lacunary(False), "even_lacunary"
    return make_even_lacunary(True), "even_lacunary_direct"


def generate_candidates(memory: EmpiricalMemory, basins: dict[str, Basin], active: list[Basin], count: int, cycle_id: str, laws: Sequence[BasinLaw] = ()) -> CandidateReservoir:
    reservoir = CandidateReservoir()
    # Five routes are the minimum needed to fill 100 slots under the 20/basin cap.
    # Cold starts and very small histories therefore receive reproducible,
    # structurally distinct exploration basins without pretending they are verified.
    synthetic_makers = [
        lambda: make_high_sparse(False), lambda: make_high_sparse(True),
        lambda: make_center_peak(False), lambda: make_even_lacunary(False),
        random_exploration,
    ]
    synthetic_index = 0
    while len(active) < 5:
        example = synthetic_makers[synthetic_index % len(synthetic_makers)]()
        basin_id = f"Bsynthetic{synthetic_index + 1}"
        synthetic = Basin(basin_id, coarse_basin_key(example), quality=0.1, examples=[example])
        basins[basin_id] = synthetic
        active.append(synthetic)
        synthetic_index += 1
    weights = [max(0.1, b.quality + 2.0) * math.log2(b.total + 2.0) for b in active]
    routed_laws = [law for law in laws if law.destination_pair == TARGET_PAIR and law.source_basin in basins]
    for serial in range(count):
        family = choose_family(RNG.random())
        basin = RNG.choices(active, weights=weights, k=1)[0]
        seed = RNG.choice(basin.examples) if basin.examples else make_high_sparse()
        applied_law: Optional[BasinLaw] = None
        if routed_laws and RNG.random() < max(0.0, min(1.0, LAW_ROUTING_SHARE)):
            applied_law = RNG.choices(routed_laws, weights=[max(0.01, law.confidence_lower) for law in routed_laws], k=1)[0]
            basin = basins[applied_law.source_basin]
            seed = RNG.choice(basin.examples) if basin.examples else seed
            poly, routed_family = apply_basin_law(seed, applied_law), "law_apply"
        elif family == "basin_guided":
            submode = serial % 5
            if submode == 0:
                poly, routed_family = mutate_micro(seed), "micro"
            elif submode == 1:
                poly, routed_family = mutate_a0_sweep(seed), "a0_sweep"
            elif submode == 2:
                poly, routed_family = mutate_sparse_keep_shape(seed), "sparse_keep_shape"
            elif submode == 3:
                poly, routed_family = mutate_basin_guided(seed, basin.fingerprint), "basin_guided"
            else:
                poly, routed_family = atlas_walk(seed, basin, basins), "atlas_walk"
        elif family == "basin_clone":
            poly, routed_family = mutate_basin_clone(seed), "basin_clone"
        elif family == "transition_mutation":
            poly, routed_family = transition_mutation(seed, basin, basins), "transition_mutation"
        elif family == "structured":
            poly, routed_family = structured_candidate(serial)
        else:
            poly, routed_family = random_exploration(), "random_exploration"
        trial = make_trial(seed, poly, routed_family, basin.basin_id, predicted_pair(basin), cycle_id, serial)
        if applied_law is not None:
            trial.mutation_spec["law_id"] = applied_law.law_id
        for example in basin.members:
            if example.poly == tuple(seed):
                trial.pair_before = example.pair
                break
        candidate = evaluate_candidate(poly, routed_family, basin, memory, serial, trial)
        reservoir.offer(candidate)
        if count >= 100_000 and (serial + 1) % 100_000 == 0:
            print(f"generated {serial + 1:,}/{count:,}; retained {sum(len(h) for h in reservoir.heaps.values()):,}")
    return reservoir


def select_portfolio(candidates: list[Candidate]) -> list[Candidate]:
    selected: list[Candidate] = []
    basin_counts: Counter = Counter()
    family_counts: Counter = Counter()
    fp_counts: Counter = Counter()
    support_counts: Counter = Counter()
    law_counts: Counter = Counter()
    used: set[str] = set()
    remaining = sorted(candidates, reverse=True)
    while remaining and len(selected) < MAX_SUBMISSIONS:
        best_index, best_score = -1, -math.inf
        for i, c in enumerate(remaining[: min(8000, len(remaining))]):
            line = poly_to_line(c.poly)
            law_id = c.trial.mutation_spec.get("law_id", "") if c.trial is not None else ""
            family_limit = 80 if c.family == "law_apply" else 20
            if (line in used or basin_counts[c.basin_id] >= 20 or family_counts[c.family] >= family_limit
                    or (law_id and law_counts[law_id] >= 20) or fp_counts[c.fp_key] >= 4):
                continue
            support = tuple(i for i, x in enumerate(c.poly) if x)
            marginal = c.expected_value
            marginal *= 1.0 / (1.0 + 0.06 * basin_counts[c.basin_id])
            marginal *= 1.0 / (1.0 + 0.05 * family_counts[c.family])
            marginal *= 1.0 / (1.0 + 0.08 * support_counts[support])
            trial_lines = [poly_to_line(x.poly) for x in selected] + [line]
            if body_bytes(trial_lines) > MAX_BYTES:
                continue
            if marginal > best_score:
                best_index, best_score = i, marginal
        if best_index < 0:
            break
        chosen = remaining.pop(best_index)
        selected.append(chosen)
        used.add(poly_to_line(chosen.poly))
        basin_counts[chosen.basin_id] += 1
        family_counts[chosen.family] += 1
        chosen_law_id = chosen.trial.mutation_spec.get("law_id", "") if chosen.trial is not None else ""
        if chosen_law_id:
            law_counts[chosen_law_id] += 1
        fp_counts[chosen.fp_key] += 1
        support_counts[tuple(i for i, x in enumerate(chosen.poly) if x)] += 1
    return selected


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def candidate_row(c: Candidate, rank: int = -1) -> dict[str, Any]:
    return {
        "rank": rank, "polynomial": poly_to_line(c.poly), "family": c.family,
        "basin_id": c.basin_id, "expected_t": c.expected_pair[0],
        "expected_r": c.expected_pair[1], "expected_score": c.expected_score,
        "expected_value": c.expected_value, "novelty": c.novelty,
        "basin_quality": c.basin_quality, "diversity": c.diversity,
        "transition_bonus": c.transition_bonus, "penalty": c.penalty,
        "fingerprint": c.fp_key, "sha256": line_hash(poly_to_line(c.poly)),
    }


def trial_dict(trial: MutationTrial) -> dict[str, Any]:
    row = asdict(trial)
    row["parent_poly"] = list(trial.parent_poly)
    row["child_poly"] = list(trial.child_poly)
    row["coefficient_delta"] = list(trial.coefficient_delta)
    row["fingerprint_before"] = list(trial.fingerprint_before)
    row["fingerprint_after"] = list(trial.fingerprint_after)
    return row


def trial_from_dict(row: dict[str, Any]) -> Optional[MutationTrial]:
    try:
        def pair(name: str) -> Optional[tuple[int, int]]:
            value = row.get(name)
            return tuple(map(int, value)) if isinstance(value, (list, tuple)) and len(value) == 2 else None
        return MutationTrial(
            trial_id=str(row["trial_id"]), parent_poly=tuple(map(int, row["parent_poly"])),
            child_poly=tuple(map(int, row["child_poly"])), parent_hash=str(row["parent_hash"]),
            child_hash=str(row["child_hash"]), parent_basin=str(row["parent_basin"]),
            child_basin=str(row["child_basin"]), mutation_type=str(row["mutation_type"]),
            mutation_spec=dict(row.get("mutation_spec", {})), coefficient_delta=tuple(map(int, row["coefficient_delta"])),
            fingerprint_before=tuple(row.get("fingerprint_before", [])), fingerprint_after=tuple(row.get("fingerprint_after", [])),
            fingerprint_delta=dict(row.get("fingerprint_delta", {})), target_pair=pair("target_pair"),
            pair_before=pair("pair_before"), pair_after=pair("pair_after"),
            successful=row.get("successful"), submission_id=str(row.get("submission_id", "")),
            polynomial_index=int(row.get("polynomial_index", -1)), cycle_id=str(row.get("cycle_id", "")),
            generator_version=str(row.get("generator_version", GENERATOR_VERSION)), random_seed=int(row.get("random_seed", SEED)),
            status=str(row.get("status", "CANDIDATE")), trust_label=str(row.get("trust_label", TRUST_GENERATED)),
            api_result=dict(row.get("api_result", {})), created_at=str(row.get("created_at", "")),
            observed_at=str(row.get("observed_at", "")),
        )
    except (KeyError, TypeError, ValueError):
        return None


def load_trials(path: Path = LEDGER) -> list[MutationTrial]:
    trials: dict[str, MutationTrial] = {}
    for text in read_lines(path):
        try:
            trial = trial_from_dict(json.loads(text))
        except json.JSONDecodeError:
            trial = None
        if trial is not None:
            trials[trial.trial_id] = trial
    return list(trials.values())


def augment_memory_from_trials(memory: EmpiricalMemory, trials: Sequence[MutationTrial]) -> None:
    seen = {(poly_to_line(ex.poly), ex.pair) for ex in memory.all_verified}
    for sequence, trial in enumerate(trials):
        if not is_observed_trial(trial) or trial.pair_after is None or not valid_poly(trial.child_poly):
            continue
        marker = (poly_to_line(trial.child_poly), trial.pair_after)
        if marker in seen:
            continue
        seen.add(marker)
        ex = VerifiedExample(trial.child_poly, trial.pair_after, f"v101:{trial.submission_id}", trial.polynomial_index, sequence)
        memory.all_verified.append(ex)
        memory.verified_examples[trial.pair_after].append(ex)
        memory.pair_counts[trial.pair_after] += 1
        memory.actual_pair_history.append(trial.pair_after)
        memory.known_submitted_lines.add(marker[0])


def append_trials(trials: Sequence[MutationTrial], path: Path = LEDGER) -> None:
    if not trials:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for trial in trials:
            handle.write(stable_json(trial_dict(trial)) + "\n")


def write_cycle_trials(path: Path, trials: Sequence[MutationTrial]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for trial in trials:
            handle.write(stable_json(trial_dict(trial)) + "\n")


def join_api_results(trials: Sequence[MutationTrial], final_json: dict[str, Any], submission_id: str) -> list[MutationTrial]:
    by_index = {trial.polynomial_index: trial for trial in trials}
    observed: list[MutationTrial] = []
    verified = final_json.get("verifiedPolynomials", []) or []
    failed = final_json.get("failedPolynomials", []) or []
    for result in list(verified) + list(failed):
        index = extract_index(result)
        if index is None or index not in by_index:
            continue
        trial = by_index[index]
        trial.submission_id = submission_id
        trial.api_result = dict(result)
        trial.pair_after = extract_pair(result)
        trial.status = str(result.get("status", "ok" if result in verified else "failed"))
        trial.successful = trial.pair_after == trial.target_pair if trial.target_pair is not None else trial.pair_after is not None
        trial.trust_label = TRUST_HELD_OUT_REPLICATION if trial.mutation_spec.get("law_id") else TRUST_OBSERVED_TRIAL
        trial.observed_at = utc_now()
        observed.append(trial)
    return observed


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials <= 0:
        return (0.0, 1.0)
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * trials)) / trials) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def summarize_changes(trials: Sequence[MutationTrial]) -> tuple[dict[str, Any], dict[str, Any]]:
    coeff: dict[str, Any] = {}
    for index in range(COEFFICIENT_COUNT):
        values = [trial.coefficient_delta[index] for trial in trials if trial.coefficient_delta[index] != 0]
        if values:
            coeff[f"a{index}"] = {"median": statistics.median(values), "min": min(values), "max": max(values), "count": len(values)}
    numeric: dict[str, list[float]] = defaultdict(list)
    for trial in trials:
        for name, value in trial.fingerprint_delta.items():
            if isinstance(value, (int, float)):
                numeric[name].append(float(value))
    fp = {name: {"median": statistics.median(values), "mean": statistics.mean(values)} for name, values in numeric.items()}
    return coeff, fp


def learn_basin_laws(trials: Sequence[MutationTrial], min_trials: int = 3) -> list[BasinLaw]:
    observed = [trial for trial in trials if is_observed_trial(trial) and trial.pair_after is not None]
    contexts: dict[tuple[str, str, str, Optional[tuple[int, int]]], list[MutationTrial]] = defaultdict(list)
    for trial in observed:
        key = (trial.parent_basin, trial.mutation_type, delta_signature(trial.coefficient_delta), trial.pair_before)
        contexts[key].append(trial)
    laws: list[BasinLaw] = []
    for (source_basin, mutation_type, signature, source_pair), context_trials in contexts.items():
        if len(context_trials) < min_trials:
            continue
        destinations: dict[tuple[int, int], list[MutationTrial]] = defaultdict(list)
        for trial in context_trials:
            if trial.pair_after is not None:
                destinations[trial.pair_after].append(trial)
        for destination, successes in destinations.items():
            lower, upper = wilson_interval(len(successes), len(context_trials))
            failures = [trial for trial in context_trials if trial.pair_after != destination]
            submissions = {trial.submission_id for trial in successes if trial.submission_id}
            coeff, fp = summarize_changes(successes)
            identity = stable_json([source_basin, mutation_type, signature, source_pair, destination])
            law_id = "LAW" + hashlib.sha1(identity.encode()).hexdigest()[:12]
            replay_trials = [trial for trial in observed if trial.mutation_spec.get("law_id") == law_id]
            replay_successes = sum(trial.pair_after == destination for trial in replay_trials)
            replay_failures = len(replay_trials) - replay_successes
            replay_submissions = {trial.submission_id for trial in replay_trials if trial.submission_id}
            laws.append(BasinLaw(
                law_id=law_id,
                source_basin=source_basin, target_basin=Counter(t.child_basin for t in successes).most_common(1)[0][0],
                mutation_type=mutation_type, delta_signature=signature,
                coefficient_changes=coeff, fingerprint_changes=fp, source_pair=source_pair,
                destination_pair=destination, success_count=len(successes), failure_count=len(failures),
                trial_count=len(context_trials),
                posterior_mean=(len(successes) + 1) / (len(context_trials) + 2),
                confidence_lower=lower, confidence_upper=upper, submission_count=len(submissions),
                examples=[trial.trial_id for trial in successes[:20]],
                failure_examples=[trial.trial_id for trial in failures[:20]],
                replay_successes=replay_successes, replay_failures=replay_failures,
                trust_label=(TRUST_REPLICATED_LAW
                             if len(replay_submissions) >= 2 and replay_successes >= 3 and replay_successes > replay_failures
                             else TRUST_EMPIRICAL_LAW),
            ))
    return sorted(laws, key=lambda law: (law.trust_label == TRUST_REPLICATED_LAW, law.confidence_lower, law.success_count), reverse=True)


def obstruction_name(trials: Sequence[MutationTrial]) -> str:
    deltas = Counter()
    for trial in trials:
        for name, value in trial.fingerprint_delta.items():
            if isinstance(value, (int, float)) and value:
                deltas[(name, "increase" if value > 0 else "decrease")] += 1
    if not deltas:
        return "Unresolved Basin Collapse"
    (name, direction), _ = deltas.most_common(1)[0]
    pretty = name.replace("_", " ").title()
    return f"{pretty} {direction.title()}"


def learn_obstructions(trials: Sequence[MutationTrial], min_trials: int = 3) -> list[Obstruction]:
    harmful: dict[tuple[Optional[tuple[int, int]], tuple[int, int], str, str, str], list[MutationTrial]] = defaultdict(list)
    for trial in trials:
        if not is_observed_trial(trial) or trial.pair_after not in NEGATIVE_PAIRS:
            continue
        signature = delta_signature(trial.coefficient_delta)
        harmful[(trial.pair_before, trial.pair_after, trial.parent_basin, trial.mutation_type, signature)].append(trial)
    result = []
    for (source_pair, destination, basin, mutation_type, signature), examples in harmful.items():
        if len(examples) < min_trials:
            continue
        context_count = sum(
            1 for trial in trials
            if trial.parent_basin == basin and trial.mutation_type == mutation_type and delta_signature(trial.coefficient_delta) == signature
        )
        confidence = (len(examples) + 1) / (context_count + 2)
        identity = stable_json([source_pair, destination, basin, mutation_type, signature])
        result.append(Obstruction(
            obstruction_id="OBS" + hashlib.sha1(identity.encode()).hexdigest()[:12],
            name=obstruction_name(examples), source_pair=source_pair, destination_pair=destination,
            source_basin=basin, mutation_type=mutation_type, triggering_delta=signature,
            support=len(examples), trial_count=context_count,
            failure_rate=len(examples) / max(1, context_count), confidence=confidence,
            examples=[trial.trial_id for trial in examples[:20]],
        ))
    return sorted(result, key=lambda item: (item.confidence, item.trial_count), reverse=True)


def build_inverse_routes(laws: Sequence[BasinLaw], obstructions: Sequence[Obstruction], desired_pair: tuple[int, int] = TARGET_PAIR) -> list[InverseRoute]:
    routes = []
    for law in laws:
        if law.destination_pair != desired_pair:
            continue
        blocked = [obs.obstruction_id for obs in obstructions if obs.source_basin == law.source_basin and obs.mutation_type == law.mutation_type]
        probability = law.posterior_mean * math.prod(1.0 - min(0.95, obs.confidence) for obs in obstructions if obs.obstruction_id in blocked)
        identity = stable_json([law.source_pair, desired_pair, law.source_basin, law.law_id])
        routes.append(InverseRoute(
            route_id="ROUTE" + hashlib.sha1(identity.encode()).hexdigest()[:12],
            current_pair=law.source_pair, desired_pair=desired_pair, source_basin=law.source_basin,
            law_ids=[law.law_id], estimated_probability=probability,
            known_obstructions=blocked, confidence=law.confidence_lower,
        ))
    return sorted(routes, key=lambda route: (route.estimated_probability, route.confidence), reverse=True)


def write_lawbook(trials: Sequence[MutationTrial], laws: Sequence[BasinLaw], obstructions: Sequence[Obstruction], routes: Sequence[InverseRoute]) -> None:
    write_csv(OUT / "trial_certificates.csv", [
        {
            "trial_id": t.trial_id, "parent_hash": t.parent_hash, "mutation_spec": stable_json(t.mutation_spec),
            "child_hash": t.child_hash, "submission_id": t.submission_id, "polynomial_index": t.polynomial_index,
            "pair_before": stable_json(t.pair_before), "pair_after": stable_json(t.pair_after),
            "successful": t.successful, "status": t.status, "trust_label": t.trust_label,
        } for t in trials
    ], ["trial_id", "parent_hash", "mutation_spec", "child_hash", "submission_id", "polynomial_index", "pair_before", "pair_after", "successful", "status", "trust_label"])
    law_rows = [asdict(law) for law in laws]
    for row in law_rows:
        for key in ("coefficient_changes", "fingerprint_changes", "source_pair", "destination_pair", "examples", "failure_examples"):
            row[key] = stable_json(row[key])
    write_csv(OUT / "basin_laws.csv", law_rows)
    obstruction_rows = [asdict(item) for item in obstructions]
    for row in obstruction_rows:
        for key in ("source_pair", "destination_pair", "examples"):
            row[key] = stable_json(row[key])
    write_csv(OUT / "obstructions.csv", obstruction_rows)
    route_rows = [asdict(route) for route in routes]
    for row in route_rows:
        for key in ("current_pair", "desired_pair", "law_ids", "known_obstructions"):
            row[key] = stable_json(row[key])
    write_csv(OUT / "inverse_routes.csv", route_rows)
    lawbook = {
        "generated_at": utc_now(), "trust_boundary": "Empirical routing objects are not verified mathematical theorems.",
        "trial_count": len(trials), "observed_trial_count": sum(is_observed_trial(t) for t in trials),
        "laws": [asdict(law) for law in laws], "obstructions": [asdict(item) for item in obstructions],
        "inverse_routes": [asdict(route) for route in routes],
    }
    (OUT / "lawbook.json").write_text(json.dumps(lawbook, indent=2, sort_keys=True), encoding="utf-8")


def write_outputs(selected: list[Candidate], candidates: list[Candidate], basins: dict[str, Basin], posteriors: list[dict[str, Any]], memory: EmpiricalMemory, reservoir: CandidateReservoir, elapsed: float, cycle_id: str) -> tuple[dict[str, Any], list[MutationTrial]]:
    ensure_dirs()
    lines = [poly_to_line(c.poly) for c in selected]
    if len(lines) > MAX_SUBMISSIONS or len(lines) != len(set(lines)) or body_bytes(lines) > MAX_BYTES:
        raise RuntimeError("portfolio violates submission limits")
    if not all(valid_poly(parse_poly(line) or ()) for line in lines):
        raise RuntimeError("portfolio contains a locally invalid polynomial")
    (OUT / "submission.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    (API_OUT / "submitted_valid_polys.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    submission = {
        "version": "v101_provenance_bearing_continuation_engine", "cycle_id": cycle_id, "degree": DEGREE,
        "count": len(lines), "bytes": body_bytes(lines), "polynomials": [list(c.poly) for c in selected],
    }
    (OUT / "submission.json").write_text(json.dumps(submission, indent=2), encoding="utf-8")
    write_csv(OUT / "selected_top100.csv", [candidate_row(c, i + 1) for i, c in enumerate(selected)])
    write_csv(OUT / "candidate_pool.csv", [candidate_row(c, i + 1) for i, c in enumerate(candidates)])
    write_csv(OUT / "expected_scores.csv", [candidate_row(c, i + 1) for i, c in enumerate(selected)])
    basin_rows = []
    for basin in sorted(basins.values(), key=lambda b: b.quality, reverse=True):
        basin_rows.append({
            "basin_id": basin.basin_id, "fingerprint": json.dumps(basin.fingerprint),
            "members": len(basin.members), "quality": basin.quality,
            "pair_counts": json.dumps({f"{a}:{b}": n for (a, b), n in basin.pair_counts.items()}, sort_keys=True),
            "example_count": len(basin.examples), "transition_count": sum(basin.transitions.values()),
        })
    write_csv(OUT / "basin_memory.csv", basin_rows)
    edge_rows = [
        {"source_basin": b.basin_id, "target_basin": target, "count": count}
        for b in basins.values() for target, count in b.transitions.items()
    ]
    write_csv(OUT / "transition_graph.csv", edge_rows, ["source_basin", "target_basin", "count"])
    write_csv(OUT / "pair_posteriors.csv", posteriors, ["basin_id", "t", "r", "count", "p_pair_given_basin", "p_basin_given_pair"])
    selected_trials = []
    for index, candidate in enumerate(selected):
        trial = candidate.trial or make_trial(candidate.poly, candidate.poly, candidate.family, candidate.basin_id, candidate.expected_pair, cycle_id, candidate.serial)
        trial.polynomial_index = index
        trial.status = "SELECTED"
        selected_trials.append(trial)
    cycle_dir = CYCLES_DIR / cycle_id
    cycle_dir.mkdir(parents=True, exist_ok=True)
    (cycle_dir / "submission.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    write_cycle_trials(cycle_dir / "trials.jsonl", selected_trials)
    report = {
        "version": "v101_provenance_bearing_continuation_engine", "cycle_id": cycle_id, "seed": SEED,
        "generated_requested": GENERATION_COUNT, "generated_offered": reservoir.generated,
        "valid_unique_offers": reservoir.valid, "candidate_pool": len(candidates),
        "selected": len(selected), "submission_bytes": body_bytes(lines),
        "verified_rows": len(memory.all_verified), "known_submissions": len(memory.known_submitted_lines),
        "basins": len(basins), "active_basins": len([b for b in basins.values() if b.examples]),
        "transition_edges": len(edge_rows), "expected_score": sum(c.expected_score for c in selected),
        "expected_value": sum(c.expected_value for c in selected),
        "target_distribution": {f"{a}:{b}": n for (a, b), n in Counter(c.expected_pair for c in selected).items()},
        "family_distribution": dict(Counter(c.family for c in selected)),
        "basin_distribution": dict(Counter(c.basin_id for c in selected)),
        "elapsed_seconds": elapsed,
        "terminal_status": "CANDIDATE_PORTFOLIO_NOT_A_VERIFIED_CLAIM",
    }
    (OUT / "run_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (cycle_dir / "run_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report, selected_trials


def sanitize_api_key(raw: str) -> str:
    return "".join(ch for ch in raw.strip() if 32 < ord(ch) < 127 and not ch.isspace())


def api_headers(api_key: str, json_content: bool = False) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {sanitize_api_key(api_key)}"}
    if json_content:
        headers["Content-Type"] = "application/json"
    return headers


def api_get(api_key: str, path: str) -> dict[str, Any]:
    if requests is None:
        raise RuntimeError("requests is required for SAIR API access")
    response = requests.get(f"{API_BASE}/{path.lstrip('/')}", headers=api_headers(api_key), timeout=180)
    response.raise_for_status()
    envelope = response.json()
    return envelope.get("data", envelope)


def read_api_state(api_key: str) -> dict[str, Any]:
    competition = api_get(api_key, f"competitions/{COMPETITION_ID}")
    eligibility = api_get(api_key, f"competitions/{COMPETITION_ID}/me")
    spec = competition.get("submissionSpec", {}) or {}
    if spec.get("kind") and spec.get("kind") != "igp24-polynomial":
        raise RuntimeError(f"unexpected submission kind: {spec.get('kind')}")
    return {"competition": competition, "eligibility": eligibility, "checked_at": utc_now()}


def submission_history() -> list[dict[str, Any]]:
    path = OUT / "submission_history.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def record_submission(submission_id: str, cycle_id: str) -> None:
    history = submission_history()
    history.append({"submission_id": submission_id, "cycle_id": cycle_id, "submitted_at": utc_now()})
    (OUT / "submission_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def allowed_to_submit(api_state: dict[str, Any]) -> tuple[bool, str]:
    eligibility = api_state.get("eligibility", {}) or {}
    can_submit = eligibility.get("canSubmit")
    if can_submit is False:
        return False, str(eligibility.get("submitBlockedReason") or "API reports canSubmit=false")
    today = datetime.now(timezone.utc).date().isoformat()
    used_today = sum(str(item.get("submitted_at", "")).startswith(today) for item in submission_history())
    explicit_remaining = None
    for key in ("submissionsRemainingToday", "remainingSubmissionsToday", "dailySubmissionsRemaining"):
        value = safe_int(eligibility.get(key))
        if value is not None:
            explicit_remaining = value
            break
    if explicit_remaining is not None and explicit_remaining <= 0:
        return False, "API eligibility reports no submissions remaining today"
    if can_submit is True:
        return True, "API reports canSubmit=true"
    # If canSubmit is absent, use the conservative documented daily limit.
    credited = safe_int(eligibility.get("distinctScoreablePairs")) or safe_int(eligibility.get("creditedPairCount")) or 0
    conservative_cap = 100 if credited >= 5 else 5
    if used_today >= conservative_cap and explicit_remaining is None:
        return False, f"local UTC submission ledger reached conservative daily cap {conservative_cap}"
    return True, "eligible"


def submit_to_sair(lines: Sequence[str], description: str = "MathGraph v101 provenance-bearing continuation engine") -> Optional[dict[str, Any]]:
    if requests is None:
        print("requests is not installed; skipping API submission")
        return None
    if not lines or len(lines) > MAX_SUBMISSIONS or body_bytes(lines) > MAX_BYTES:
        raise ValueError("submission violates count or byte limits")
    if not all(valid_poly(parse_poly(line) or ()) for line in lines):
        raise ValueError("submission contains invalid polynomials")
    api_key = sanitize_api_key(os.environ.get("SAIR_API_KEY", ""))
    if not api_key and AUTORUN:
        try:
            api_key = sanitize_api_key(getpass.getpass("Paste SAIR API key: "))
        except Exception:
            api_key = ""
    if not api_key:
        print("No SAIR_API_KEY supplied; skipping API submission")
        return None
    headers = api_headers(api_key, json_content=True)
    payload = {
        "payload": {"polynomials": list(lines)},
        "meta": {"description": description[:500]},
    }
    response = requests.post(f"{API_BASE}/competitions/{COMPETITION_ID}/submissions", headers=headers, json=payload, timeout=180)
    if response.status_code == 429:
        print("Rate limited. Retry-After:", response.headers.get("Retry-After"))
        return None
    response.raise_for_status()
    envelope = response.json()
    result = envelope.get("data", envelope)
    result["_api_key"] = api_key
    return result


def poll_submission(api_key: str, submission_id: str, max_polls: int = 80, poll_seconds: int = 30) -> dict[str, Any]:
    if requests is None:
        raise RuntimeError("requests is required for polling")
    headers = api_headers(api_key)
    last: dict[str, Any] = {}
    for attempt in range(max_polls):
        response = requests.get(f"{API_BASE}/competitions/{COMPETITION_ID}/submissions/{submission_id}", headers=headers, timeout=180)
        if response.status_code == 404:
            response = requests.get(f"{API_BASE}/competitions/{COMPETITION_ID}/submissions/me", headers=headers, timeout=180)
        if response.status_code == 429:
            print("Rate limited while polling. Retry-After:", response.headers.get("Retry-After"))
            time.sleep(poll_seconds)
            continue
        response.raise_for_status()
        envelope = response.json()
        last = envelope.get("data", envelope)
        verified = last.get("verifiedPolynomials", []) or []
        failed = last.get("failedPolynomials", []) or []
        queued = (last.get("payload", {}) or {}).get("queuedPolynomials", []) or []
        print(f"poll {attempt + 1}/{max_polls}: verified={len(verified)} failed={len(failed)} queued={len(queued)}")
        if not queued:
            return last
        time.sleep(poll_seconds)
    return last


def save_api_results(final_json: dict[str, Any], cycle_id: str = "") -> None:
    ensure_dirs()
    clean = {k: v for k, v in final_json.items() if k != "_api_key"}
    (API_OUT / "submission_final.json").write_text(json.dumps(clean, indent=2), encoding="utf-8")
    (OUT / "api_results.json").write_text(json.dumps(clean, indent=2), encoding="utf-8")
    if cycle_id:
        cycle_dir = CYCLES_DIR / cycle_id
        cycle_dir.mkdir(parents=True, exist_ok=True)
        (cycle_dir / "api_results.json").write_text(json.dumps(clean, indent=2), encoding="utf-8")
    verified = clean.get("verifiedPolynomials", []) or []
    failed = clean.get("failedPolynomials", []) or []
    verified_fields = sorted(set().union(*(row.keys() for row in verified))) if verified else ["polynomialIndex", "status", "label", "t", "r"]
    failed_fields = sorted(set().union(*(row.keys() for row in failed))) if failed else ["polynomialIndex", "status", "reason"]
    write_csv(API_OUT / "verified_from_api.csv", verified, verified_fields)
    write_csv(API_OUT / "failed_from_api.csv", failed, failed_fields)
    counts = Counter()
    for row in verified:
        pair = extract_pair(row)
        if pair is not None:
            counts[pair] += 1
    summary = [
        {"t": pair[0], "r": pair[1], "count": count, "value": pair_value(pair), "banned": pair in NEGATIVE_PAIRS}
        for pair, count in counts.most_common()
    ]
    write_csv(API_OUT / "verified_pairs_summary.csv", summary, ["t", "r", "count", "value", "banned"])


def print_report(report: dict[str, Any]) -> None:
    banner("STARTUP / RUN SUMMARY")
    labels = [
        ("verified rows", "verified_rows"), ("known submissions", "known_submissions"),
        ("basins", "basins"), ("active basins", "active_basins"),
        ("transition edges", "transition_edges"), ("candidate count", "candidate_pool"),
        ("selected count", "selected"), ("expected score", "expected_score"),
        ("target distribution", "target_distribution"), ("family distribution", "family_distribution"),
        ("basin distribution", "basin_distribution"),
    ]
    for label, key in labels:
        print(f"{label}: {report.get(key)}")


def cycle_identifier(number: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"cycle_{number:06d}_{stamp}"


def run_cycle(number: int, api_key: str = "") -> dict[str, Any]:
    started = time.time()
    cycle_id = cycle_identifier(number)
    banner(f"v101 CONTINUATION CYCLE {cycle_id}")
    historical_trials = load_trials()
    laws = learn_basin_laws(historical_trials)
    obstructions = learn_obstructions(historical_trials)
    routes = build_inverse_routes(laws, obstructions)
    memory = load_empirical_memory()
    augment_memory_from_trials(memory, historical_trials)
    basins = build_basins(memory, historical_trials)
    active = inverse_route(basins, TARGET_PAIR, top_n=24)
    posteriors = pair_posteriors(basins, memory)
    print(f"observed trials: {sum(is_observed_trial(t) for t in historical_trials):,}")
    print(f"empirical laws: {len(laws):,}; obstructions: {len(obstructions):,}; routes: {len(routes):,}")
    print(f"verified rows: {len(memory.all_verified):,}; known submissions: {len(memory.known_submitted_lines):,}")
    reservoir = generate_candidates(memory, basins, active, GENERATION_COUNT, cycle_id, laws)
    candidates = reservoir.values()
    selected = select_portfolio(candidates)
    report, selected_trials = write_outputs(
        selected, candidates, basins, posteriors, memory, reservoir, time.time() - started, cycle_id
    )
    # Persist the exact parent/spec/child records before any external action.
    append_trials(selected_trials)
    write_lawbook(load_trials(), laws, obstructions, routes)
    report["autorun"] = AUTORUN
    report["submission_attempted"] = False
    if not AUTORUN:
        print("dry-run complete; set SAIR_AUTORUN=1 with SAIR_API_KEY to enable cyclic submission")
        print_report(report)
        return report
    if not api_key:
        raise RuntimeError("SAIR_AUTORUN=1 requires SAIR_API_KEY")
    api_state = read_api_state(api_key)
    allowed, reason = allowed_to_submit(api_state)
    (CYCLES_DIR / cycle_id / "api_state.json").write_text(json.dumps(api_state, indent=2), encoding="utf-8")
    if not allowed:
        report["submission_blocked_reason"] = reason
        print("submission blocked:", reason)
        return report
    lines = read_lines(OUT / "submission.txt")
    result = submit_to_sair(lines, f"MathGraph v101 cycle {cycle_id}")
    report["submission_attempted"] = True
    if not result:
        report["submission_blocked_reason"] = "submission returned no result"
        return report
    submission_id = str(result.get("submissionId") or result.get("id") or result.get("submission_id") or "")
    returned_key = str(result.pop("_api_key", api_key))
    if not submission_id:
        raise RuntimeError("SAIR submission response did not include submissionId")
    record_submission(submission_id, cycle_id)
    for trial in selected_trials:
        trial.submission_id = submission_id
        trial.status = "SUBMITTED"
        trial.trust_label = TRUST_SUBMITTED
    append_trials(selected_trials)
    final = poll_submission(returned_key, submission_id, MAX_POLLS, POLL_SECONDS)
    save_api_results(final, cycle_id)
    observed = join_api_results(selected_trials, final, submission_id)
    append_trials(observed)
    all_trials = load_trials()
    laws = learn_basin_laws(all_trials)
    obstructions = learn_obstructions(all_trials)
    routes = build_inverse_routes(laws, obstructions)
    write_lawbook(all_trials, laws, obstructions, routes)
    report.update({
        "submission_id": submission_id, "observed_results": len(observed),
        "empirical_laws": len(laws), "named_obstructions": len(obstructions),
        "inverse_routes": len(routes),
    })
    (CYCLES_DIR / cycle_id / "cycle_final.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print_report(report)
    return report


def main() -> None:
    ensure_dirs()
    banner("MathGraph IGP24 v101 — Provenance-Bearing Continuation Engine")
    api_key = sanitize_api_key(os.environ.get("SAIR_API_KEY", ""))
    cycle = 1
    while MAX_CYCLES == 0 or cycle <= MAX_CYCLES:
        try:
            report = run_cycle(cycle, api_key)
            print(f"cycle {cycle} complete: selected={report.get('selected', 0)} observed={report.get('observed_results', 0)}")
        except KeyboardInterrupt:
            print("continuation engine stopped by user")
            break
        except Exception as exc:
            error = {"cycle": cycle, "time": utc_now(), "error": type(exc).__name__, "message": str(exc)}
            with (OUT / "errors.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(stable_json(error) + "\n")
            print(f"cycle {cycle} failed: {type(exc).__name__}: {exc}")
        if MAX_CYCLES != 0 and cycle >= MAX_CYCLES:
            break
        print(f"sleeping {CYCLE_SLEEP_SECONDS}s before next eligibility check")
        time.sleep(max(60, CYCLE_SLEEP_SECONDS))
        cycle += 1


if __name__ == "__main__":
    main()

# END FULL SCRIPT
