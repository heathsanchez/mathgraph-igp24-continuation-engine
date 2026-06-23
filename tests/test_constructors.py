from mathgraph_igp24.constructors import CONSTRUCTORS, reciprocal_score
from mathgraph_igp24 import valid_poly


def test_all_constructors_emit_valid_provenance(parent):
    import random

    for index, constructor in enumerate(CONSTRUCTORS.values()):
        candidate = constructor.generate(parent, random.Random(index), seed=index)
        assert valid_poly(candidate.child), constructor.name
        assert candidate.parent_hash and candidate.child_hash
        assert candidate.constructor_name == constructor.name
        assert candidate.source_basin_id


def test_reciprocal_breaker_reduces_reciprocal_score(parent):
    import random

    candidate = CONSTRUCTORS["reciprocal_breaker"].generate(parent, random.Random(0), seed=0)
    assert reciprocal_score(candidate.child) <= reciprocal_score(parent)


def test_support_transport_changes_support(parent):
    import random

    candidate = CONSTRUCTORS["support_transport"].generate(parent, random.Random(1), seed=1)
    assert candidate.constructor_params["src"] != candidate.constructor_params["dst"]
