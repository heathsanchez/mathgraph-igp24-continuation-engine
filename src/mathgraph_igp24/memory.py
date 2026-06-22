from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .models import MutationTrial, Pair, Polynomial, TRUST_HELD_OUT_REPLICATION, TRUST_OBSERVED_TRIAL


@dataclass
class EmpiricalMemory:
    pair_counts: Counter = field(default_factory=Counter)
    verified_examples: dict[Pair, list[Polynomial]] = field(default_factory=lambda: defaultdict(list))
    known_child_hashes: set[str] = field(default_factory=set)
    trials: list[MutationTrial] = field(default_factory=list)


def build_memory(trials: Sequence[MutationTrial]) -> EmpiricalMemory:
    memory = EmpiricalMemory(trials=list(trials))
    for trial in trials:
        memory.known_child_hashes.add(trial.child_hash)
        if trial.trust_label not in {TRUST_OBSERVED_TRIAL, TRUST_HELD_OUT_REPLICATION} or trial.pair_after is None:
            continue
        memory.pair_counts[trial.pair_after] += 1
        memory.verified_examples[trial.pair_after].append(trial.child_poly)
    return memory

