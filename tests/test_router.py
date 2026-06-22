from mathgraph_igp24 import Obstruction, hash_poly, recommend, replay_law

from conftest import empirical_law


def test_recommend_with_learned_law_beats_fallback(parent):
    learned = recommend(parent, (14010, 8), laws=[empirical_law(parent)])
    fallback = recommend(parent, (14010, 8))
    assert learned.target_pair_posterior > fallback.target_pair_posterior == 0
    assert learned.support_count == 7 and learned.failure_count == 3
    assert hash_poly(replay_law(parent, learned.law)) == learned.expected_child_hash
    assert "no empirical law exists" in fallback.rationale[0]


def test_obstruction_penalizes_harmful_law(parent):
    harmful = empirical_law(parent, mutation_type="center_mass_push", law_id="LAW-harmful")
    safe = empirical_law(parent, mutation_type="a0_sweep", delta=2, law_id="LAW-safe")
    obstruction = Obstruction("OBS-x", harmful.source_basin, "center_mass_push", (24979, 8),
                              "support explosion", 9, 1, 0.9)
    selected = recommend(parent, (14010, 8), laws=[harmful, safe], obstructions=[obstruction])
    assert selected.law.law_id == "LAW-safe"

