from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

Polynomial = tuple[int, ...]
Pair = tuple[int, int]

TRUST_GENERATED = "GENERATED"
TRUST_SUBMITTED = "SUBMITTED"
TRUST_OBSERVED_TRIAL = "OBSERVED_TRIAL"
TRUST_HELD_OUT_REPLICATION = "HELD_OUT_REPLICATION"
TRUST_EMPIRICAL_LAW = "EMPIRICAL_LAW"
TRUST_REPLICATED_LAW = "HELD_OUT_REPLICATED_LAW"
TRUST_NAMED_OBSTRUCTION = "NAMED_OBSTRUCTION"
TRUST_ROUTING_POLICY = "ROUTING_POLICY"


@dataclass(frozen=True)
class Evidence:
    successes: int
    failures: int
    confidence_lower: float
    confidence_upper: float
    trial_ids: tuple[str, ...] = ()
    replay_successes: int = 0
    replay_failures: int = 0

    @property
    def trials(self) -> int:
        return self.successes + self.failures

    @property
    def posterior_mean(self) -> float:
        return (self.successes + 1) / (self.trials + 2)


@dataclass(frozen=True)
class BasinLaw:
    law_id: str
    source_basin: str
    target_basin: str
    target_pair: Pair
    coefficient_deltas: tuple[int, ...]
    preconditions: dict[str, Any]
    evidence: Evidence
    trust_label: str = TRUST_EMPIRICAL_LAW
    version: str = "1"

    def __post_init__(self) -> None:
        if len(self.coefficient_deltas) != 25:
            raise ValueError("a basin law requires exactly 25 coefficient deltas")
        if self.coefficient_deltas[24] != 0:
            raise ValueError("a basin law cannot change the monic leading coefficient")


@dataclass(frozen=True)
class Obstruction:
    obstruction_id: str
    source_basin: str
    mutation_signature: str
    harmful_pair: Pair
    support: int
    failure_rate: float
    confidence: float
    example_trial_ids: tuple[str, ...] = ()
    trust_label: str = TRUST_NAMED_OBSTRUCTION


@dataclass(frozen=True)
class RouteRecommendation:
    recommendation_id: str
    parent_hash: str
    target_pair: Pair
    law: BasinLaw
    expected_child_hash: str
    expected_probability: float
    evidence: Evidence
    obstructions: tuple[Obstruction, ...]
    rationale: tuple[str, ...]
    trust_label: str = TRUST_ROUTING_POLICY


@dataclass
class MutationTrial:
    trial_id: str
    parent_hash: str
    child_hash: str
    parent_poly: Polynomial
    child_poly: Polynomial
    source_basin: str
    destination_basin: str
    mutation_type: str
    mutation_parameters: dict[str, Any]
    target_pair: Optional[Pair]
    pair_before: Optional[Pair]
    pair_after: Optional[Pair]
    submission_id: str
    polynomial_index: int
    status: str
    generator_version: str
    random_seed: int
    trust_label: str
    timestamp: str
    outcome: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Outcome:
    status: str
    pair: Optional[Pair]
    payload: dict[str, Any] = field(default_factory=dict)

