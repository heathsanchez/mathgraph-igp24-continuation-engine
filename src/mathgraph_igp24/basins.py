from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from .models import MutationTrial, Pair, TRUST_HELD_OUT_REPLICATION, TRUST_OBSERVED_TRIAL


@dataclass
class Basin:
    basin_id: str
    pair_counts: Counter = field(default_factory=Counter)
    outgoing: Counter = field(default_factory=Counter)


def build_basin_atlas(trials: Sequence[MutationTrial]) -> dict[str, Basin]:
    """Construct transitions only from explicit parent/mutation/child provenance."""
    basins: dict[str, Basin] = {}
    for trial in trials:
        if trial.trust_label not in {TRUST_OBSERVED_TRIAL, TRUST_HELD_OUT_REPLICATION}:
            continue
        source = basins.setdefault(trial.source_basin, Basin(trial.source_basin))
        basins.setdefault(trial.destination_basin, Basin(trial.destination_basin))
        source.outgoing[trial.destination_basin] += 1
        if trial.pair_after is not None:
            basins[trial.destination_basin].pair_counts[trial.pair_after] += 1
    return basins

