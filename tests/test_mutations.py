from mathgraph_igp24 import OPERATORS, apply_mutation, hash_poly, replay_mutation, valid_poly


def test_every_operator_is_seeded_and_replayable(parent):
    for index, name in enumerate(OPERATORS):
        parameters = {"coefficient_deltas": (0,) * 25, "law_id": "LAW-zero"} if name == "law_replay" else {}
        result = apply_mutation(parent, name, parameters, seed=100 + index)
        replayed = replay_mutation(parent, name, result.parameters)
        assert valid_poly(result.child), name
        assert hash_poly(replayed) == hash_poly(result.child), name
        assert result.parameters["seed"] == 100 + index

