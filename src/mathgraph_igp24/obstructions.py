from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Sequence

from .laws import coefficient_delta
from .models import (
    MutationTrial,
    Obstruction,
    Pair,
    TRUST_HELD_OUT_REPLICATION,
    TRUST_OBSERVED_TRIAL,
)
from .trials import stable_json

DEFAULT_HARMFUL_PAIRS: frozenset[Pair] = frozenset({
    (24979, 2), (24979, 4), (24979, 6), (24979, 8),
    (25000, 2), (25000, 4), (25000, 6), (25000, 8),
})


def mutation_signature(trial: MutationTrial) -> str:
    changes = [(index, value) for index, value in enumerate(coefficient_delta(trial)) if value]
    return stable_json(changes)


def learn_obstructions(
    trials: Sequence[MutationTrial],
    harmful_pairs: frozenset[Pair] = DEFAULT_HARMFUL_PAIRS,
    minimum_support: int = 3,
) -> list[Obstruction]:
    observed = [
        trial for trial in trials
        if trial.trust_label in {TRUST_OBSERVED_TRIAL, TRUST_HELD_OUT_REPLICATION} and trial.pair_after is not None
    ]
    contexts: dict[tuple[str, str], list[MutationTrial]] = defaultdict(list)
    for trial in observed:
        contexts[(trial.source_basin, mutation_signature(trial))].append(trial)
    result = []
    for (source_basin, signature), context in contexts.items():
        harmful_by_pair: dict[Pair, list[MutationTrial]] = defaultdict(list)
        for trial in context:
            if trial.pair_after in harmful_pairs:
                harmful_by_pair[trial.pair_after].append(trial)  # type: ignore[index]
        for harmful_pair, failures in harmful_by_pair.items():
            if len(failures) < minimum_support:
                continue
            rate = len(failures) / len(context)
            identity = stable_json([source_basin, signature, harmful_pair])
            result.append(Obstruction(
                obstruction_id="OBS" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16],
                source_basin=source_basin, mutation_signature=signature, harmful_pair=harmful_pair,
                support=len(failures), failure_rate=rate,
                confidence=(len(failures) + 1) / (len(context) + 2),
                example_trial_ids=tuple(trial.trial_id for trial in failures[:50]),
            ))
    return sorted(result, key=lambda obstruction: (obstruction.confidence, obstruction.support), reverse=True)

