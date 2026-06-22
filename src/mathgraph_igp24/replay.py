from __future__ import annotations

from typing import Sequence

from .fingerprints import basin_id
from .models import BasinLaw, Polynomial
from .polynomial import hash_poly, normalize_poly


def replay_law(poly: Sequence[int], law: BasinLaw) -> Polynomial:
    parent = normalize_poly(poly)
    if basin_id(parent) != law.source_basin:
        raise ValueError(f"law {law.law_id} source-basin precondition failed")
    edits = tuple(value + delta for value, delta in zip(parent, law.coefficient_deltas))
    child = normalize_poly(edits)
    required_parent = law.preconditions.get("parent_hash")
    if required_parent and hash_poly(parent) != required_parent:
        raise ValueError(f"law {law.law_id} parent-hash precondition failed")
    return child
