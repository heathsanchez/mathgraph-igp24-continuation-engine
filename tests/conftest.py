from dataclasses import replace

import pytest

from mathgraph_igp24 import BasinLaw, Evidence, apply_mutation, basin_id, create_trial
from mathgraph_igp24.models import TRUST_EMPIRICAL_LAW, TRUST_OBSERVED_TRIAL


@pytest.fixture
def parent():
    return tuple([1, 1] + [0] * 22 + [1])


def observed(parent, mutation_type, parameters, pair_after, index=0, cycle="c1", submission="s1", target=(14010, 8)):
    result = apply_mutation(parent, mutation_type, parameters, seed=index)
    trial = create_trial(parent, result.child, mutation_type, result.parameters, target,
                         random_seed=index, cycle_id=cycle)
    return replace(trial, submission_id=submission, polynomial_index=index, pair_after=pair_after,
                   outcome_status="ok", api_t=pair_after[0], api_r=pair_after[1],
                   trust_label=TRUST_OBSERVED_TRIAL)


def empirical_law(parent, mutation_type="center_mass_push", delta=18, target=(14010, 8), law_id="LAW-test"):
    deltas = [0] * 25; deltas[12] = delta
    return BasinLaw(
        law_id=law_id, source_basin=basin_id(parent), target_basin="B-target", target_pair=target,
        mutation_type=mutation_type, coefficient_deltas=tuple(deltas),
        preconditions={"source_basin": basin_id(parent)}, evidence=Evidence(7, 3, 0.4, 0.86, (), 7, 0),
        mutation_parameters_summary={"index": 12, "delta": delta}, success_rate=0.7,
        lift_over_baseline=2.0, novelty_score=1.0, trust_label=TRUST_EMPIRICAL_LAW,
    )

