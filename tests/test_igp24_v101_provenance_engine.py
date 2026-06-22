import mathgraph_igp24_v101_provenance_bearing_continuation_engine as v101


def example(poly, pair, sequence):
    return v101.VerifiedExample(tuple(poly), pair, "same_submission.csv", sequence, sequence)


def memory_with(*examples):
    memory = v101.EmpiricalMemory()
    for item in examples:
        memory.all_verified.append(item)
        memory.verified_examples[item.pair].append(item)
        memory.pair_counts[item.pair] += 1
    return memory


def basin_id(poly):
    key = v101.coarse_basin_key(poly)
    return "B" + v101.hashlib.sha1(repr(key).encode()).hexdigest()[:10]


def test_official_coefficient_format_is_comma_separated():
    poly = (1,) + (0,) * 23 + (1,)
    line = v101.poly_to_line(poly)
    assert line.count(",") == 24
    assert " " not in line
    assert v101.parse_poly(line) == poly
    assert v101.valid_poly(poly)


def test_csv_adjacency_does_not_create_a_transition():
    parent = v101.make_high_sparse(False)
    child = v101.make_center_peak(False)
    memory = memory_with(example(parent, (24970, 12), 0), example(child, (14010, 8), 1))
    basins = v101.build_basins(memory)
    assert sum(sum(basin.transitions.values()) for basin in basins.values()) == 0


def test_observed_parent_child_provenance_creates_transition():
    parent = v101.make_high_sparse(False)
    child = v101.make_center_peak(False)
    memory = memory_with(example(parent, (24970, 12), 0), example(child, (14010, 8), 1))
    trial = v101.make_trial(parent, child, "center_peak", basin_id(parent), (14010, 8), "cycle", 0)
    trial.pair_before = (24970, 12)
    trial.pair_after = (14010, 8)
    trial.status = "ok"
    trial.trust_label = v101.TRUST_OBSERVED_TRIAL
    basins = v101.build_basins(memory, [trial])
    assert basins[basin_id(parent)].transitions[basin_id(child)] == 1


def test_deterministic_api_join_promotes_replicated_law():
    parent = v101.make_center_peak(False)
    child = list(parent)
    child[12] += 18
    child = v101.normalize_poly(child)
    assert child is not None
    trials = []
    for index in range(6):
        trial = v101.make_trial(parent, child, "center_mass_push", "B17", (14010, 8), f"cycle-{index}", index)
        trial.polynomial_index = index
        trials.append(trial)
    final = {
        "verifiedPolynomials": [
            {"polynomialIndex": index, "computed_label": "24T14010", "computed_r": 8, "status": "ok"}
            for index in range(6)
        ],
        "failedPolynomials": [],
    }
    observed = v101.join_api_results(trials, final, "submission-current")
    failures = []
    for index in range(4):
        trial = v101.make_trial(parent, child, "center_mass_push", "B17", (14010, 8), f"failure-{index}", index)
        trial.polynomial_index = index
        failures.append(trial)
    failure_json = {
        "verifiedPolynomials": [
            {"polynomialIndex": index, "computed_label": "24T24979", "computed_r": 8, "status": "ok"}
            for index in range(4)
        ],
        "failedPolynomials": [],
    }
    observed_failures = v101.join_api_results(failures, failure_json, "submission-failures")
    empirical = v101.learn_basin_laws(observed + observed_failures)
    target_law = next(law for law in empirical if law.destination_pair == (14010, 8))
    assert target_law.success_count == 6
    assert target_law.failure_count == 4
    assert target_law.trust_label == v101.TRUST_EMPIRICAL_LAW

    replay_child = v101.replay_law(parent, target_law)
    replays = []
    for index in range(6):
        trial = v101.make_trial(parent, replay_child, "law_apply", "B17", (14010, 8), f"replay-{index}", index)
        trial.mutation_spec["law_id"] = target_law.law_id
        trial.polynomial_index = index
        replays.append(trial)
    replay_json = {
        "verifiedPolynomials": [
            {"polynomialIndex": index, "computed_label": "24T14010", "computed_r": 8, "status": "ok"}
            for index in range(6)
        ],
        "failedPolynomials": [],
    }
    observed_replays = v101.join_api_results(replays, replay_json, "submission-replay")
    for index, trial in enumerate(observed_replays):
        trial.submission_id = "replay-a" if index < 3 else "replay-b"
    laws = v101.learn_basin_laws(observed + observed_failures + observed_replays)
    promoted = next(law for law in laws if law.law_id == target_law.law_id)
    assert len(observed) == 6
    assert all(trial.trust_label == v101.TRUST_HELD_OUT_REPLICATION for trial in observed_replays)
    assert promoted.replay_successes == 6
    assert promoted.trust_label == v101.TRUST_REPLICATED_LAW


def test_named_obstruction_records_support_and_failure_rate():
    parent = v101.make_even_lacunary(False)
    child = list(parent)
    child[8] += 5
    child = v101.normalize_poly(child)
    trials = []
    for index in range(6):
        trial = v101.make_trial(parent, child, "support_push", "B-source", None, f"obs-{index}", index)
        trial.pair_after = (24979, 8) if index < 4 else (14010, 8)
        trial.status = "ok"
        trial.trust_label = v101.TRUST_OBSERVED_TRIAL
        trials.append(trial)
    obstructions = v101.learn_obstructions(trials)
    assert obstructions
    obstruction = obstructions[0]
    assert obstruction.support == 4
    assert obstruction.trial_count == 6
    assert obstruction.failure_rate == 4 / 6
    assert obstruction.trust_label == v101.TRUST_NAMED_OBSTRUCTION
