from mathgraph_igp24 import apply_mutation, build_submission


def test_submission_builder_emits_at_most_100_valid_unique(parent):
    polynomials = []
    for index in range(100):
        result = apply_mutation(parent, "a0_sweep", {"delta": index + 1}, seed=index)
        polynomials.append(result.child)
    text = build_submission(polynomials)
    assert len(text.splitlines()) == 100
    assert len(text.encode()) < 100_000

