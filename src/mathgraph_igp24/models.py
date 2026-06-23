from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

Polynomial = tuple[int, ...]
Pair = tuple[int, int]

TRUST_GENERATED = "GENERATED"
TRUST_SUBMITTED = "SUBMITTED"
TRUST_OBSERVED_TRIAL = "OBSERVED_TRIAL"
TRUST_EMPIRICAL_LAW = "EMPIRICAL_LAW"
TRUST_REPLICATED_LAW = "REPLICATED_LAW"
TRUST_DEMOTED = "DEMOTED"
TRUST_OBSTRUCTION = "OBSTRUCTION"
TRUST_OBSTRUCTED = TRUST_OBSTRUCTION
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
    mutation_type: str
    coefficient_deltas: tuple[int, ...]
    preconditions: dict[str, Any]
    evidence: Evidence
    mutation_parameters_summary: dict[str, Any] = field(default_factory=dict)
    coefficient_delta_mean: tuple[float, ...] = ()
    coefficient_delta_median: tuple[float, ...] = ()
    coefficient_delta_mode: tuple[int, ...] = ()
    fingerprint_delta_mean: dict[str, float] = field(default_factory=dict)
    fingerprint_delta_median: dict[str, float] = field(default_factory=dict)
    success_rate: float = 0.0
    lift_over_baseline: float = 0.0
    matched_failure_count: int = 0
    novelty_score: float = 1.0
    submission_ids: tuple[str, ...] = ()
    cycle_ids: tuple[str, ...] = ()
    trust_label: str = TRUST_GENERATED
    version: str = "2"

    def __post_init__(self) -> None:
        if len(self.coefficient_deltas) != 25:
            raise ValueError("a basin law requires exactly 25 coefficient deltas")
        if self.coefficient_deltas[24] != 0:
            raise ValueError("a basin law cannot change the monic leading coefficient")

    @property
    def success_count(self) -> int:
        return self.evidence.successes

    @property
    def failure_count(self) -> int:
        return self.evidence.failures

    @property
    def trial_count(self) -> int:
        return self.evidence.trials

    @property
    def confidence_lower(self) -> float:
        return self.evidence.confidence_lower

    @property
    def confidence_upper(self) -> float:
        return self.evidence.confidence_upper

    @property
    def replay_successes(self) -> int:
        return self.evidence.replay_successes

    @property
    def replay_failures(self) -> int:
        return self.evidence.replay_failures


@dataclass(frozen=True)
class Obstruction:
    obstruction_id: str
    source_basin: str
    mutation_type: str
    harmful_pair: Pair
    feature_signature: str
    failure_count: int
    matched_success_count: int
    confidence: float
    avoided_by: tuple[str, ...] = ()
    example_trial_ids: tuple[str, ...] = ()
    trust_label: str = TRUST_OBSTRUCTED

    @property
    def support(self) -> int:
        return self.failure_count

    @property
    def failure_rate(self) -> float:
        total = self.failure_count + self.matched_success_count
        return self.failure_count / total if total else 0.0


@dataclass(frozen=True)
class RouteRecommendation:
    recommendation_id: str
    parent_hash: str
    target_pair: Pair
    law: BasinLaw
    coefficient_edits: tuple[dict[str, int], ...]
    mutation_type: str
    mutation_parameters: dict[str, Any]
    expected_child_hash: str
    expected_child_polynomial: Polynomial
    predicted_destination_basin: str
    target_pair_posterior: float
    confidence_interval: tuple[float, float]
    support_count: int
    failure_count: int
    replay_evidence: dict[str, int]
    relevant_obstructions: tuple[Obstruction, ...]
    rationale: tuple[str, ...]
    trust_label: str = TRUST_ROUTING_POLICY

    @property
    def expected_probability(self) -> float:
        return self.target_pair_posterior

    @property
    def evidence(self) -> Evidence:
        return self.law.evidence

    @property
    def obstructions(self) -> tuple[Obstruction, ...]:
        return self.relevant_obstructions


@dataclass
class MutationTrial:
    trial_id: str
    parent_hash: str
    child_hash: str
    parent_polynomial: Polynomial
    child_polynomial: Polynomial
    source_basin: str
    predicted_target_pair: Optional[Pair]
    mutation_type: str
    mutation_parameters: dict[str, Any]
    coefficient_delta: tuple[int, ...]
    fingerprint_before: tuple[Any, ...]
    fingerprint_after: tuple[Any, ...]
    fingerprint_delta: dict[str, Any]
    generator_version: str
    random_seed: int
    cycle_id: str
    submission_id: str
    polynomial_index: int
    pair_before: Optional[Pair]
    pair_after: Optional[Pair]
    outcome_status: str
    api_label: str
    api_t: Optional[int]
    api_r: Optional[int]
    api_reason: str
    trust_label: str
    created_at: str
    updated_at: str

    @property
    def parent_poly(self) -> Polynomial:
        return self.parent_polynomial

    @property
    def child_poly(self) -> Polynomial:
        return self.child_polynomial

    @property
    def destination_basin(self) -> str:
        from .fingerprints import basin_id
        return basin_id(self.child_polynomial)

    @property
    def target_pair(self) -> Optional[Pair]:
        return self.predicted_target_pair

    @property
    def status(self) -> str:
        return self.outcome_status

    @property
    def timestamp(self) -> str:
        return self.updated_at

    @property
    def outcome(self) -> dict[str, Any]:
        return {
            "status": self.outcome_status, "label": self.api_label,
            "t": self.api_t, "r": self.api_r, "reason": self.api_reason,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Outcome:
    status: str
    pair: Optional[Pair]
    label: str = ""
    reason: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
