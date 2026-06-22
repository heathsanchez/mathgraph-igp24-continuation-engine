from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from typing import Any, Sequence


def entropy(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def fingerprint(poly: Sequence[int]) -> tuple[Any, ...]:
    support = tuple(index for index, coefficient in enumerate(poly) if coefficient)
    weights = tuple(abs(value) for value in poly)
    total_weight = sum(weights) or 1
    height = max(weights)
    center_index = max(range(7, 18), key=lambda index: weights[index])
    gaps = tuple(right - left - 1 for left, right in zip(support, support[1:]))
    return (
        len(support),
        min(12, int(math.log10(height + 1))),
        sum(index % 2 == 0 for index in support),
        sum(index % 2 == 1 for index in support),
        center_index,
        round(weights[center_index] / max(1, height), 2),
        "".join("1" if value else "0" for value in poly),
        max(gaps, default=0),
        round(entropy(tuple(value for value in poly if value)), 3),
        round(sum(index * weight for index, weight in enumerate(weights)) / total_weight, 3),
        round(statistics.pvariance(weights) / ((statistics.mean(weights) or 1) ** 2), 3),
    )


def basin_id(poly: Sequence[int]) -> str:
    raw = json.dumps(fingerprint(poly), separators=(",", ":"), ensure_ascii=True)
    return "B" + hashlib.sha256(raw.encode("ascii")).hexdigest()[:16]

