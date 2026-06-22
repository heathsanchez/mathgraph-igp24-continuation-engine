from __future__ import annotations

import hashlib
import math
import statistics
from collections import Counter, defaultdict
from typing import Any, Optional, Sequence

from .models import BasinLaw, Evidence, MutationTrial, Pair, TRUST_DEMOTED, TRUST_EMPIRICAL_LAW, TRUST_OBSERVED_TRIAL, TRUST_REPLICATED_LAW
from .mutations import replay_mutation
from .polynomial import hash_poly
from .trials import stable_json


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials <= 0: return (0.0, 1.0)
    p = successes / trials; denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * trials)) / trials) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _vector_stat(vectors: Sequence[Sequence[int]], function) -> tuple[float, ...]:
    return tuple(float(function([vector[index] for vector in vectors])) for index in range(25)) if vectors else (0.0,) * 25


def _mode(values: Sequence[int]) -> int:
    return Counter(values).most_common(1)[0][0]


def _parameter_summary(trials: Sequence[MutationTrial]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    keys = set().union(*(trial.mutation_parameters.keys() for trial in trials)) if trials else set()
    for key in sorted(keys - {"seed", "description", "edits"}):
        values = [trial.mutation_parameters.get(key) for trial in trials if key in trial.mutation_parameters]
        if values:
            summaries[key] = Counter(stable_json(value) for value in values).most_common(1)[0][0]
    return summaries


def _fingerprint_stats(trials: Sequence[MutationTrial], function) -> dict[str, float]:
    numeric: dict[str, list[float]] = defaultdict(list)
    for trial in trials:
        for key, value in trial.fingerprint_delta.items():
            if isinstance(value, (int, float)): numeric[key].append(float(value))
    return {key: float(function(values)) for key, values in numeric.items() if values}


def _replays(trials: Sequence[MutationTrial]) -> tuple[int, int]:
    successes = failures = 0
    for trial in trials:
        try:
            child = replay_mutation(trial.parent_polynomial, trial.mutation_type, trial.mutation_parameters)
            successes += hash_poly(child) == trial.child_hash
            failures += hash_poly(child) != trial.child_hash
        except Exception:
            failures += 1
    return successes, failures


def learn_laws(trials: Sequence[MutationTrial], minimum_trials: int = 5) -> list[BasinLaw]:
    observed = [trial for trial in trials if trial.trust_label == TRUST_OBSERVED_TRIAL and trial.pair_after is not None]
    source_totals: dict[str, list[MutationTrial]] = defaultdict(list)
    contexts: dict[tuple[str, str, Optional[Pair]], list[MutationTrial]] = defaultdict(list)
    for trial in observed:
        source_totals[trial.source_basin].append(trial)
        contexts[(trial.source_basin, trial.mutation_type, trial.pair_before)].append(trial)
    laws: list[BasinLaw] = []
    for (source_basin, mutation_type, pair_before), context in contexts.items():
        destination_pairs = sorted({trial.pair_after for trial in context if trial.pair_after is not None})
        for target_pair in destination_pairs:
            successes = [trial for trial in context if trial.pair_after == target_pair]
            failures = [trial for trial in context if trial.pair_after != target_pair]
            total = len(context); success_rate = len(successes) / total
            source_rows = source_totals[source_basin]
            baseline = sum(trial.pair_after == target_pair for trial in source_rows) / max(1, len(source_rows))
            lift = success_rate / max(1e-9, baseline)
            lower, upper = wilson_interval(len(successes), total)
            replay_successes, replay_failures = _replays(context)
            submissions = tuple(sorted({trial.submission_id for trial in successes if trial.submission_id}))
            cycles = tuple(sorted({trial.cycle_id for trial in successes if trial.cycle_id}))
            meaningful_margin = lower > baseline + 0.05
            empirical = total >= minimum_trials and len(successes) >= 2 and (lower > baseline or lift > 1.5) and replay_successes >= len(successes)
            replicated = (empirical and len(successes) >= 5 and (len(submissions) >= 2 or len(cycles) >= 2)
                          and len(failures) <= len(successes) and meaningful_margin)
            trust = TRUST_REPLICATED_LAW if replicated else TRUST_EMPIRICAL_LAW if empirical else TRUST_DEMOTED
            vectors = [trial.coefficient_delta for trial in successes or context]
            representative = tuple(int(round(value)) for value in _vector_stat(vectors, statistics.median))
            identity = stable_json([source_basin, mutation_type, pair_before, target_pair, representative])
            evidence = Evidence(len(successes), len(failures), lower, upper,
                                tuple(trial.trial_id for trial in context[:100]), replay_successes, replay_failures)
            targets = Counter(trial.destination_basin for trial in successes)
            laws.append(BasinLaw(
                law_id="LAW" + hashlib.sha256(identity.encode()).hexdigest()[:16], source_basin=source_basin,
                target_basin=targets.most_common(1)[0][0] if targets else source_basin, target_pair=target_pair,
                mutation_type=mutation_type, coefficient_deltas=representative,
                preconditions={"source_basin": source_basin, "pair_before": pair_before}, evidence=evidence,
                mutation_parameters_summary=_parameter_summary(successes or context),
                coefficient_delta_mean=_vector_stat(vectors, statistics.mean),
                coefficient_delta_median=_vector_stat(vectors, statistics.median),
                coefficient_delta_mode=tuple(_mode([vector[i] for vector in vectors]) for i in range(25)),
                fingerprint_delta_mean=_fingerprint_stats(successes or context, statistics.mean),
                fingerprint_delta_median=_fingerprint_stats(successes or context, statistics.median),
                success_rate=success_rate, lift_over_baseline=lift, matched_failure_count=len(failures),
                novelty_score=len({trial.child_hash for trial in successes}) / max(1, len(successes)),
                submission_ids=submissions, cycle_ids=cycles, trust_label=trust,
            ))
    order = {TRUST_REPLICATED_LAW: 2, TRUST_EMPIRICAL_LAW: 1, TRUST_DEMOTED: 0}
    return sorted(laws, key=lambda law: (order[law.trust_label], law.confidence_lower, law.success_rate), reverse=True)

