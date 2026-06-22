import pytest

from mathgraph_igp24 import TrialLedger, apply_mutation, create_trial, join_result


def test_api_join_uses_submission_id_and_index(tmp_path, parent):
    ledger = TrialLedger(tmp_path / "trials.jsonl")
    trials = []
    for index in range(2):
        result = apply_mutation(parent, "a0_sweep", {}, seed=index)
        trials.append(create_trial(parent, result.child, result.name, result.parameters, (14010, 8), cycle_id="c"))
    ledger.register_submission(trials, "submission-x")
    observed = join_result("submission-x", 1, {"status": "ok", "computed_label": "24T14010", "computed_r": 8}, ledger)
    assert observed.polynomial_index == 1
    assert observed.pair_after == (14010, 8)
    with pytest.raises(KeyError):
        join_result("wrong-submission", 1, {"status": "ok"}, ledger)

