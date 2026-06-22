from __future__ import annotations

import hashlib
from typing import Optional, Sequence

from .fingerprints import basin_id
from .models import (
    BasinLaw,
    Evidence,
    Obstruction,
    Pair,
    RouteRecommendation,
    TRUST_GENERATED,
)
from .polynomial import hash_poly, normalize_poly
from .replay import replay_law
from .trials import stable_json


def fallback_law(poly: Sequence[int], target_pair: Pair) -> BasinLaw:
    parent = normalize_poly(poly)
    deltas = [0] * 25
    deltas[12] = 18 if parent[12] <= 0 else max(1, abs(parent[12]) // 20)
    evidence = Evidence(0, 0, 0.0, 1.0)
    identity = stable_json([basin_id(parent), target_pair, deltas, "fallback"])
    return BasinLaw(
        law_id="GEN" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16],
        source_basin=basin_id(parent), target_basin="UNKNOWN", target_pair=target_pair,
        coefficient_deltas=tuple(deltas),
        preconditions={"source_basin": basin_id(parent), "strategy": "center-mass exploration"},
        evidence=evidence, trust_label=TRUST_GENERATED,
    )


def law_score(law: BasinLaw, obstructions: Sequence[Obstruction]) -> float:
    penalty = 1.0
    for obstruction in obstructions:
        if obstruction.source_basin == law.source_basin:
            penalty *= 1.0 - min(0.95, obstruction.confidence)
    replay_factor = (law.evidence.replay_successes + 1) / (law.evidence.replay_successes + law.evidence.replay_failures + 2)
    raw = law.evidence.posterior_mean * (0.25 + law.evidence.confidence_lower) * (0.5 + replay_factor) * penalty
    return max(0.0, min(1.0, raw))


def recommend(
    poly: Sequence[int],
    target_pair: Pair,
    laws: Optional[Sequence[BasinLaw]] = None,
    obstructions: Optional[Sequence[Obstruction]] = None,
) -> RouteRecommendation:
    parent = normalize_poly(poly)
    source = basin_id(parent)
    known_laws = [law for law in (laws or ()) if law.target_pair == target_pair and law.source_basin == source]
    known_obstructions = tuple(item for item in (obstructions or ()) if item.source_basin == source)
    law = max(known_laws, key=lambda item: law_score(item, known_obstructions)) if known_laws else fallback_law(parent, target_pair)
    child = replay_law(parent, law)
    expected_hash = hash_poly(child)
    probability = law_score(law, known_obstructions) if known_laws else 0.0
    rationale = (
        f"selected {law.law_id} for source basin {source}",
        f"evidence: {law.evidence.successes} successes, {law.evidence.failures} failures",
        f"held-out replay: {law.evidence.replay_successes} successes, {law.evidence.replay_failures} failures",
        f"known obstructions avoided: {len(known_obstructions)}",
    )
    identity = stable_json([hash_poly(parent), target_pair, law.law_id, expected_hash])
    return RouteRecommendation(
        recommendation_id="REC" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20],
        parent_hash=hash_poly(parent), target_pair=target_pair, law=law,
        expected_child_hash=expected_hash, expected_probability=probability,
        evidence=law.evidence, obstructions=known_obstructions, rationale=rationale,
    )
