from mathgraph_igp24 import learn_obstructions

from conftest import observed


def test_obstruction_learning_names_repeated_harmful_routes(parent):
    trials = [
        observed(parent, "center_mass_push", {"index": 12, "delta": 18}, (24979, 8), index)
        for index in range(4)
    ]
    trials.append(observed(parent, "center_mass_push", {"index": 12, "delta": 18}, (14010, 8), 100))
    obstructions = learn_obstructions(trials, minimum_failures=3)
    assert obstructions
    obs = obstructions[0]
    assert obs.mutation_type == "center_mass_push"
    assert obs.harmful_pair == (24979, 8)
    assert obs.support >= 3
    assert obs.confidence > 0
    assert obs.trust_label == "OBSTRUCTION"
