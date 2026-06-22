from dataclasses import replace

from mathgraph_igp24 import TrialLedger, apply_mutation, create_trial
from mathgraph_igp24.models import TRUST_SUBMITTED


def test_ledger_roundtrips_without_data_loss(tmp_path, parent):
    result = apply_mutation(parent, "center_mass_push", {"delta": 18}, seed=4)
    trial = create_trial(parent, result.child, result.name, result.parameters, (14010, 8), cycle_id="cycle-4")
    ledger = TrialLedger(tmp_path / "trials.jsonl"); ledger.append(trial)
    assert ledger.load()[0].to_dict() == trial.to_dict()


def test_submission_registration_is_indexed_deterministically(tmp_path, parent):
    ledger = TrialLedger(tmp_path / "trials.jsonl"); trials = []
    for index in range(3):
        result = apply_mutation(parent, "a0_sweep", {}, seed=index)
        trials.append(create_trial(parent, result.child, result.name, result.parameters, (14010, 8), cycle_id="c"))
    registered = ledger.register_submission(trials, "submission-x")
    assert [trial.polynomial_index for trial in registered] == [0, 1, 2]
    assert all(trial.trust_label == TRUST_SUBMITTED for trial in registered)

