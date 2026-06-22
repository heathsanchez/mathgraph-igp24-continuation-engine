from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Sequence

from .models import MutationTrial, Obstruction, Pair, TRUST_OBSERVED_TRIAL
from .trials import stable_json

DEFAULT_HARMFUL_PAIRS: frozenset[Pair] = frozenset({
    (24979, 2), (24979, 4), (24979, 6), (24979, 8),
    (25000, 2), (25000, 4), (25000, 6), (25000, 8),
})


def feature_signature(trial: MutationTrial) -> str:
    numeric = {key: round(float(value), 3) for key, value in trial.fingerprint_delta.items() if isinstance(value, (int, float)) and value}
    delta = [(index, value) for index, value in enumerate(trial.coefficient_delta) if value]
    return stable_json({"fingerprint": numeric, "coefficient_delta": delta})


def learn_obstructions(trials: Sequence[MutationTrial], harmful_pairs: frozenset[Pair] = DEFAULT_HARMFUL_PAIRS, minimum_failures: int = 3) -> list[Obstruction]:
    observed = [trial for trial in trials if trial.trust_label == TRUST_OBSERVED_TRIAL and trial.pair_after is not None]
    contexts: dict[tuple[str, str, str], list[MutationTrial]] = defaultdict(list)
    source_successes: dict[str, Counter] = defaultdict(Counter)
    for trial in observed:
        contexts[(trial.source_basin, trial.mutation_type, feature_signature(trial))].append(trial)
        if trial.pair_after not in harmful_pairs: source_successes[trial.source_basin][trial.mutation_type] += 1
    result = []
    for (source, mutation_type, signature), context in contexts.items():
        for harmful_pair in sorted(harmful_pairs):
            failures = [trial for trial in context if trial.pair_after == harmful_pair]
            if len(failures) < minimum_failures: continue
            matched_successes = [trial for trial in context if trial.pair_after != harmful_pair]
            confidence = (len(failures) + 1) / (len(context) + 2)
            avoided = tuple(name for name, _ in source_successes[source].most_common(3) if name != mutation_type)
            identity = stable_json([source, mutation_type, harmful_pair, signature])
            result.append(Obstruction(
                obstruction_id="OBS" + hashlib.sha256(identity.encode()).hexdigest()[:16], source_basin=source,
                mutation_type=mutation_type, harmful_pair=harmful_pair, feature_signature=signature,
                failure_count=len(failures), matched_success_count=len(matched_successes), confidence=confidence,
                avoided_by=avoided, example_trial_ids=tuple(trial.trial_id for trial in failures[:100]),
            ))
    return sorted(result, key=lambda item: (item.confidence, item.failure_count), reverse=True)

