from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Optional, Sequence

from .models import (
    BasinLaw,
    Evidence,
    MutationTrial,
    Pair,
    TRUST_EMPIRICAL_LAW,
    TRUST_HELD_OUT_REPLICATION,
    TRUST_OBSERVED_TRIAL,
    TRUST_REPLICATED_LAW,
)
from .trials import stable_json


def coefficient_delta(trial: MutationTrial) -> tuple[int, ...]:
    return tuple(child - parent for parent, child in zip(trial.parent_poly, trial.child_poly))


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials <= 0:
        return (0.0, 1.0)
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt((proportion * (1 - proportion) + z * z / (4 * trials)) / trials) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def learn_laws(trials: Sequence[MutationTrial], minimum_trials: int = 3) -> list[BasinLaw]:
    observed = [
        trial for trial in trials
        if trial.trust_label in {TRUST_OBSERVED_TRIAL, TRUST_HELD_OUT_REPLICATION} and trial.pair_after is not None
    ]
    contexts: dict[tuple[str, tuple[int, ...], Optional[Pair]], list[MutationTrial]] = defaultdict(list)
    for trial in observed:
        contexts[(trial.source_basin, coefficient_delta(trial), trial.pair_before)].append(trial)

    laws: list[BasinLaw] = []
    for (source_basin, delta, pair_before), context in contexts.items():
        if len(context) < minimum_trials:
            continue
        destinations: dict[Pair, list[MutationTrial]] = defaultdict(list)
        for trial in context:
            destinations[trial.pair_after].append(trial)  # type: ignore[index]
        for target_pair, successes in destinations.items():
            failures = [trial for trial in context if trial.pair_after != target_pair]
            identity = stable_json([source_basin, delta, pair_before, target_pair])
            law_id = "LAW" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
            replays = [trial for trial in observed if trial.mutation_parameters.get("law_id") == law_id]
            replay_successes = sum(trial.pair_after == target_pair for trial in replays)
            replay_failures = len(replays) - replay_successes
            replay_submissions = {trial.submission_id for trial in replays if trial.submission_id}
            lower, upper = wilson_interval(len(successes), len(context))
            evidence = Evidence(
                successes=len(successes), failures=len(failures), confidence_lower=lower,
                confidence_upper=upper, trial_ids=tuple(trial.trial_id for trial in context[:50]),
                replay_successes=replay_successes, replay_failures=replay_failures,
            )
            laws.append(BasinLaw(
                law_id=law_id, source_basin=source_basin,
                target_basin=max((trial.destination_basin for trial in successes), key=lambda value: sum(t.destination_basin == value for t in successes)),
                target_pair=target_pair, coefficient_deltas=delta,
                preconditions={"source_basin": source_basin, "pair_before": pair_before},
                evidence=evidence,
                trust_label=(TRUST_REPLICATED_LAW
                             if len(replay_submissions) >= 2 and replay_successes >= 3 and replay_successes > replay_failures
                             else TRUST_EMPIRICAL_LAW),
            ))
    return sorted(laws, key=lambda law: (law.trust_label == TRUST_REPLICATED_LAW, law.evidence.confidence_lower, law.evidence.successes), reverse=True)

