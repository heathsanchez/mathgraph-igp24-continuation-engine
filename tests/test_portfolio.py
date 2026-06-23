import json

from mathgraph_igp24 import TrialLedger, build_portfolio, valid_poly


def test_portfolio_selects_100_valid_polynomials_with_trials(parent, tmp_path):
    ledger = TrialLedger(tmp_path / "trials.jsonl")
    selected = build_portfolio([parent], (14010, 8), ledger=ledger, candidate_count=2000)
    assert len(selected) == 100
    assert len({candidate.trial.child_hash for candidate in selected}) == 100
    assert all(valid_poly(candidate.polynomial) for candidate in selected)
    rows = [json.loads(line) for line in (tmp_path / "trials.jsonl").read_text().splitlines()]
    assert len(rows) == 100
    assert {row["trial_id"] for row in rows} == {candidate.trial.trial_id for candidate in selected}
