from __future__ import annotations

import hashlib
from typing import Optional, Sequence

from .fingerprints import basin_id
from .models import BasinLaw, Evidence, Obstruction, Pair, RouteRecommendation, TRUST_DEMOTED, TRUST_GENERATED, TRUST_OBSTRUCTED
from .polynomial import hash_poly, normalize_poly
from .replay import replay_law
from .trials import stable_json


PAIR_VALUES = {(14010, 8): 8.0, (7208, 8): 8.0, (24970, 20): 20.0, (24970, 24): 24.0, (25000, 16): 16.0}


def fallback_law(poly: Sequence[int], target_pair: Pair) -> BasinLaw:
    parent = normalize_poly(poly); deltas = [0] * 25
    deltas[12] = 18 if parent[12] <= 0 else max(1, abs(parent[12]) // 20)
    identity = stable_json([basin_id(parent), target_pair, deltas, "fallback"])
    return BasinLaw(
        law_id="GEN" + hashlib.sha256(identity.encode()).hexdigest()[:16], source_basin=basin_id(parent),
        target_basin="UNKNOWN", target_pair=target_pair, mutation_type="center_mass_push",
        coefficient_deltas=tuple(deltas), preconditions={"source_basin": basin_id(parent)},
        evidence=Evidence(0, 0, 0.0, 1.0), mutation_parameters_summary={"index": 12, "delta": deltas[12]},
        trust_label=TRUST_GENERATED,
    )

def obstruction_penalty(law: BasinLaw, obstructions: Sequence[Obstruction]) -> tuple[float, tuple[Obstruction, ...]]:
    relevant = tuple(item for item in obstructions if item.source_basin == law.source_basin and item.mutation_type == law.mutation_type)
    penalty = 1.0
    for item in relevant: penalty *= max(0.02, 1.0 - item.confidence * item.failure_rate)
    return penalty, relevant


def law_score(law: BasinLaw, obstructions: Sequence[Obstruction]) -> float:
    if law.trust_label in {TRUST_DEMOTED, TRUST_OBSTRUCTED}: return 0.0
    penalty, _ = obstruction_penalty(law, obstructions)
    pair_value = PAIR_VALUES.get(law.target_pair, max(0.1, law.target_pair[1] / 2))
    posterior = law.evidence.posterior_mean
    lift = max(0.01, min(5.0, law.lift_over_baseline or 1.0))
    replay = (law.replay_successes + 1) / (law.replay_successes + law.replay_failures + 2)
    return pair_value * posterior * law.confidence_lower * lift * law.novelty_score * penalty * replay


def _edits(law: BasinLaw) -> tuple[dict[str, int], ...]:
    return tuple({"index": index, "before": 0, "delta": delta} for index, delta in enumerate(law.coefficient_deltas) if delta)


def recommend(poly: Sequence[int], target_pair: Pair, laws: Optional[Sequence[BasinLaw]] = None, obstructions: Optional[Sequence[Obstruction]] = None) -> RouteRecommendation:
    parent = normalize_poly(poly); source = basin_id(parent); obstruction_list = tuple(obstructions or ())
    eligible = [law for law in (laws or ()) if law.target_pair == target_pair and law.source_basin == source and law.trust_label not in {TRUST_DEMOTED, TRUST_OBSTRUCTED}]
    empirical = bool(eligible); law = max(eligible, key=lambda item: law_score(item, obstruction_list)) if empirical else fallback_law(parent, target_pair)
    child = replay_law(parent, law); expected_hash = hash_poly(child); penalty, relevant = obstruction_penalty(law, obstruction_list)
    posterior = law.evidence.posterior_mean if empirical else 0.0
    rationale = (
        (f"selected {law.law_id} from {law.trial_count} observed trials", f"obstruction penalty={penalty:.4f}",
         f"replay evidence={law.replay_successes}/{law.replay_successes + law.replay_failures}")
        if empirical else
        ("no empirical law exists for this source basin and target pair", "returned deterministic GENERATED fallback", "confidence is zero")
    )
    identity = stable_json([hash_poly(parent), target_pair, law.law_id, expected_hash])
    edits = tuple({"index": index, "before": parent[index], "after": child[index], "delta": child[index] - parent[index]}
                  for index in range(25) if parent[index] != child[index])
    return RouteRecommendation(
        recommendation_id="REC" + hashlib.sha256(identity.encode()).hexdigest()[:20], parent_hash=hash_poly(parent),
        target_pair=target_pair, law=law, coefficient_edits=edits, mutation_type=law.mutation_type,
        mutation_parameters={**law.mutation_parameters_summary, "law_id": law.law_id, "coefficient_deltas": law.coefficient_deltas},
        expected_child_hash=expected_hash, expected_child_polynomial=child, predicted_destination_basin=basin_id(child),
        target_pair_posterior=posterior, confidence_interval=(law.confidence_lower, law.confidence_upper) if empirical else (0.0, 0.0),
        support_count=law.success_count, failure_count=law.failure_count,
        replay_evidence={"successes": law.replay_successes, "failures": law.replay_failures},
        relevant_obstructions=relevant, rationale=rationale,
        trust_label=law.trust_label if empirical else TRUST_GENERATED,
    )
