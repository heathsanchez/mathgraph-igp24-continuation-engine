from mathgraph_igp24 import build_basin_atlas, hash_poly, learn_laws, replay_law
from mathgraph_igp24.models import TRUST_REPLICATED_LAW

from conftest import observed


def training_trials(parent):
    trials = []
    for index in range(8):
        pair = (14010, 8) if index < 6 else (24979, 8)
        trials.append(observed(parent, "center_mass_push", {"index": 12, "delta": 18}, pair,
                               index, f"good-cycle-{index % 2}", f"good-submission-{index % 2}"))
    for index in range(20):
        pair = (14010, 8) if index == 0 else (24979, 8)
        trials.append(observed(parent, "a0_sweep", {"delta": 2}, pair,
                               100 + index, f"base-{index}", f"base-submission-{index % 2}"))
    return trials


def test_explicit_trials_create_law_evidence_but_adjacency_does_not(parent):
    assert build_basin_atlas([]) == {}
    trials = training_trials(parent)
    atlas = build_basin_atlas(trials)
    assert sum(sum(basin.outgoing.values()) for basin in atlas.values()) == len(trials)
    laws = learn_laws(trials)
    law = next(item for item in laws if item.mutation_type == "center_mass_push" and item.target_pair == (14010, 8))
    assert law.success_count == 6
    assert law.failure_count == 2
    assert law.replay_successes == law.trial_count
    assert law.trust_label == TRUST_REPLICATED_LAW


def test_replay_law_reproduces_expected_child_hash(parent):
    law = next(item for item in learn_laws(training_trials(parent))
               if item.mutation_type == "center_mass_push" and item.target_pair == (14010, 8))
    first = replay_law(parent, law); second = replay_law(parent, law)
    assert hash_poly(first) == hash_poly(second)

