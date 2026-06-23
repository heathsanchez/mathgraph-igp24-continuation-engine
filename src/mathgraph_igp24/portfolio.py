from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from .fingerprints import basin_id, fingerprint
from .models import BasinLaw, MutationTrial, Obstruction, Pair, Polynomial
from .mutations import OPERATORS, apply_mutation
from .polynomial import hash_poly, poly_to_line, valid_poly
from .router import law_score, recommend
from .submission import MAX_BYTES, MAX_POLYNOMIALS
from .trials import TrialLedger, create_trial


@dataclass(frozen=True)
class PortfolioCandidate:
    polynomial: Polynomial
    trial: MutationTrial
    expected_value: float
    obstruction_penalty: float
    fingerprint_key: str


def _guided(parent: Polynomial, law: BasinLaw, index: int) -> tuple[Polynomial, dict]:
    deltas = list(law.coefficient_deltas)
    if index:
        position = 1 + (index * 7) % 23
        deltas[position] += -1 if index % 2 else 1
    params = {"law_id": law.law_id, "coefficient_deltas": deltas}
    result = apply_mutation(parent, "law_replay", params, seed=index)
    return result.child, result.parameters


def generate_candidates(
    parents: Sequence[Polynomial], target_pair: Pair, laws: Sequence[BasinLaw], obstructions: Sequence[Obstruction],
    known_hashes: set[str] | None = None, count: int = 600, cycle_id: str = "cycle", seed: int = 0,
) -> list[PortfolioCandidate]:
    if not parents: raise ValueError("at least one parent polynomial is required")
    known = known_hashes or set(); candidates: list[PortfolioCandidate] = []; seen: set[str] = set()
    viable_laws = [law for law in laws if law.target_pair == target_pair and law_score(law, obstructions) > 0]
    for index in range(count):
        parent = parents[index % len(parents)]; fraction = index / max(1, count)
        matching_laws = [law for law in viable_laws if law.source_basin == basin_id(parent)]
        if matching_laws and fraction < 0.60:
            law = matching_laws[index % len(matching_laws)]; child, params = _guided(parent, law, index // len(parents))
            mutation_type = "law_replay"; expected = law_score(law, obstructions)
        elif matching_laws and fraction < 0.85:
            law = matching_laws[index % len(matching_laws)]; base, _ = _guided(parent, law, 0)
            result = apply_mutation(base, "basin_guided", {"target_edits": [{"index": 1 + index % 23, "delta": 1 if index % 2 else -1}]}, seed + index)
            child = result.child; params = {**result.parameters, "law_id": law.law_id}; mutation_type = "basin_guided"; expected = law_score(law, obstructions) * 0.8
        elif fraction < 0.95:
            name = "support_add" if index % 2 else "support_drop"
            result = apply_mutation(parent, name, {}, seed + index); child, params, mutation_type, expected = result.child, result.parameters, name, 0.05
        else:
            result = apply_mutation(parent, "random_exploration", {"count": 2 + index % 3}, seed + index)
            child, params, mutation_type, expected = result.child, result.parameters, result.name, 0.01
        child_hash = hash_poly(child)
        if child_hash in seen: continue
        seen.add(child_hash)
        trial = create_trial(parent, child, mutation_type, params, target_pair, generator_version="v102", random_seed=seed + index, cycle_id=cycle_id)
        penalty = 0.2 if any(item.source_basin == trial.source_basin and item.mutation_type == mutation_type for item in obstructions) else 1.0
        novelty = 0.1 if child_hash in known else 1.0
        candidates.append(PortfolioCandidate(child, trial, expected * penalty * novelty, penalty, repr(fingerprint(child))))
    return candidates


def fallback_atlas_candidates(parents: Sequence[Polynomial], target_pair: Pair, count: int, cycle_id: str, seed: int) -> list[PortfolioCandidate]:
    result = []
    names = tuple(name for name in OPERATORS if name != "law_replay")
    for index in range(count):
        parent = parents[index % len(parents)]; name = names[index % len(names)]
        mutation = apply_mutation(parent, name, {}, seed + index)
        trial = create_trial(parent, mutation.child, name, mutation.parameters, target_pair, generator_version="v102", random_seed=seed + index, cycle_id=cycle_id)
        result.append(PortfolioCandidate(mutation.child, trial, 0.01, 1.0, repr(fingerprint(mutation.child))))
    unique = {candidate.trial.child_hash: candidate for candidate in result}
    return list(unique.values())


def select_portfolio(candidates: Sequence[PortfolioCandidate], limit: int = MAX_POLYNOMIALS) -> list[PortfolioCandidate]:
    selected = []; hashes = set(); lines = set(); basin_counts = Counter(); mutation_counts = Counter(); support_counts = Counter(); fingerprint_counts = Counter(); bytes_used = 0
    for candidate in sorted(candidates, key=lambda item: (item.expected_value, item.trial.trust_label), reverse=True):
        line = poly_to_line(candidate.polynomial); encoded = len((line + "\n").encode())
        support = tuple(i for i, value in enumerate(candidate.polynomial) if value); basin = basin_id(candidate.polynomial)
        if (candidate.trial.child_hash in hashes or line in lines or not valid_poly(candidate.polynomial)
                or bytes_used + encoded > MAX_BYTES or basin_counts[basin] >= 20
                or mutation_counts[candidate.trial.mutation_type] >= 60 or support_counts[support] >= 4
                or fingerprint_counts[candidate.fingerprint_key] >= 4):
            continue
        selected.append(candidate); hashes.add(candidate.trial.child_hash); lines.add(line); bytes_used += encoded
        basin_counts[basin] += 1; mutation_counts[candidate.trial.mutation_type] += 1; support_counts[support] += 1; fingerprint_counts[candidate.fingerprint_key] += 1
        if len(selected) >= limit: break
    return selected


def build_portfolio(
    parents: Sequence[Polynomial], target_pair: Pair, laws: Sequence[BasinLaw] = (), obstructions: Sequence[Obstruction] = (),
    known_hashes: set[str] | None = None, ledger: TrialLedger | None = None, cycle_id: str = "cycle", seed: int = 0,
    candidate_count: int = 2000,
) -> list[PortfolioCandidate]:
    candidates = (generate_candidates(parents, target_pair, laws, obstructions, known_hashes, candidate_count, cycle_id, seed)
                  if laws else fallback_atlas_candidates(parents, target_pair, candidate_count, cycle_id, seed))
    selected = select_portfolio(candidates)
    if ledger is not None: ledger.append_many([candidate.trial for candidate in selected])
    return selected
