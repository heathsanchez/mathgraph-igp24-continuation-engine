import json

from mathgraph_igp24.survivor_atlas import (
    basin_id_from_signature,
    basin_signature,
    build_survivor_atlas,
    parse_poly,
    reciprocal_score,
)


def test_survivor_atlas_loads_fixture_runs_and_stable_features(tmp_path, parent):
    run = tmp_path / "run_a"
    run.mkdir()
    (run / "submission.txt").write_text(",".join(map(str, parent)) + "\n", encoding="utf-8")
    (run / "verified_joined.json").write_text(json.dumps([{
        "index": 0, "polynomial": ",".join(map(str, parent)), "computed_label": "24T24648", "computed_r": 4
    }]), encoding="utf-8")
    records, summary = build_survivor_atlas(tmp_path, tmp_path / "out")
    assert records and summary
    poly = parse_poly(",".join(map(str, parent)))
    assert basin_id_from_signature(basin_signature(poly, "test")) == basin_id_from_signature(basin_signature(poly, "test"))
    assert 0 <= reciprocal_score(poly) <= 1
    assert (tmp_path / "out" / "basin_atlas.csv").exists()


def test_survivor_atlas_handles_missing_root(tmp_path):
    records, summary = build_survivor_atlas(tmp_path / "missing", tmp_path / "out")
    assert records == []
    assert summary == []
