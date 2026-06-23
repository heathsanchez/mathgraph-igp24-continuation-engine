# MathGraph IGP24 Continuation Engine

MathGraph IGP24 v102 — Provenance Lawbook Scoring Engine is a replayable, provenance-bearing continuation engine for the
degree-24 inverse Galois problem competition.

Its core record is:

```text
Parent → Mutation → Child → Submission → Observed outcome
```

From replayable mutation trials it learns empirical basin laws, identifies
named obstructions, and builds inverse routes toward desired `(24Tt, r)` pairs.
Empirical routing objects are explicitly labelled and are not represented as
verified mathematical theorems.

## Public contracts

```python
from mathgraph_igp24 import hash_poly, join_result, recommend, replay_law

rec = recommend(parent, (14010, 8), laws=lawbook.laws)
child = replay_law(parent, rec.law)

assert hash_poly(child) == rec.expected_child_hash
```

The package is organized by responsibility:

```text
src/mathgraph_igp24/
  memory.py         empirical outcomes
  fingerprints.py  polynomial geometry
  basins.py         provenance-only transitions
  trials.py         mutation ledger and deterministic result joins
  laws.py           success/failure law learning
  obstructions.py   harmful continuation detection
  router.py         evidence-bearing recommendations
  replay.py         deterministic law application
  submission.py     official artifact validation
  api.py            eligibility-aware SAIR client
```

## Repository boundary

Git contains source, schemas, compact fixtures, tests, the build script, and
the generated Colab runner. Runtime artifacts stay in Drive or external
storage:

- `trials.jsonl`, candidate pools, lawbooks
- submissions and API results
- trial certificates and run reports

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

## Generate the Colab runner

```bash
python scripts/build_colab_artifact.py
```

This deterministically produces:

```text
dist/mathgraph_igp24_v102_colab.py
dist/artifact_manifest.json
```

The manifest freezes the artifact hash and every embedded source hash. The
generated runner can be uploaded to Colab as a single `.py` file.

In Colab:

```python
%run dist/mathgraph_igp24_v102_colab.py --root /content/drive/MyDrive/MathGraph_IGP24
```

The default is a dry cycle. Submission and polling are explicit:

```python
%env SAIR_SUBMIT=1
%env SAIR_POLL=1
%env SAIR_API_KEY=your_key_here
%run dist/mathgraph_igp24_v102_colab.py
```

The key is read only from `SAIR_API_KEY` or an interactive `getpass` prompt.

## Feedback cycle

```text
parent → frozen mutation parameters → child → durable pending trial
       → submission_id + polynomial_index → API outcome
       → observed trial → law/obstruction update → next recommendation
```

Every selected polynomial is written to the ledger before submission. API
results join only on `submission_id + polynomial_index`; CSV adjacency is never
treated as causal evidence. Mutation operators are seeded and replay from their
stored parameters.

The cycle portfolio targets 60% law replay/guidance, 25% basin-near variants,
10% obstruction falsification, and 5% exploration when empirical laws exist.
With an empty lawbook it falls back to deterministic atlas exploration.

## Join an observed result

```python
observed = join_result(
    submission_id,
    polynomial_index,
    {"status": "ok", "computed_label": "24T14010", "computed_r": 8},
    ledger,
)
```

This engine does not guarantee leaderboard dominance. It supplies the durable,
falsifiable feedback loop needed to compound empirical routing evidence across
repeated submissions.

## v109 Survivor Geometry Engine

v109 adds a survivor-geometry layer above the v102 continuation ledger. The
objective is not accepted-count maximization. IGP24 rewards low-k scoreable
signatures: roughly, a non-baseline pair with `k=1` can be worth `1.0`, `k=2`
at most `0.5`, and `k=3` at most `0.25`; crowded high-k pairs are mostly dead
for leaderboard purposes.

The v109 loop is:

```text
prior runs → survivor atlas → leaderboard/k context → obstruction atlas
          → H-tilt route scores → constructor factory → strict portfolio
          → optional API submit/poll → joined survivor evidence
```

Empirical route objects keep MathGraph trust discipline:

- `GENERATED`
- `EMPIRICAL_ROUTE`
- `OBSERVED_SURVIVOR`
- `OBSTRUCTED`
- `VERIFIED_BY_API`

They are not mathematical theorems.

### Survivor atlas

`src/mathgraph_igp24/survivor_atlas.py` ingests old run artifacts wherever it
can find them, including `submission.txt`, `selected_meta.json`,
`verified_joined.json`, `trials.jsonl`, `api_submit_cleaned/submission_final.json`,
`selected_top100.csv`, and `candidate_pool.csv`. Missing files are tolerated.

It writes:

```text
basin_atlas.csv
survivor_records.csv
basin_summary.json
```

The atlas records support geometry, parity, reciprocal/palindromic scores,
central mass, constructor family, observed `(T, r)` pair when known, and stable
survivor basin IDs.

### Leaderboard / k-aware context

Place leaderboard files in the run root, `root/leaderboard/`, or pass
`--leaderboard-path`. Supported names include:

```text
igp24_leaderboard.csv
igp24_leaderboard.json
igp24_leaderboard_by_member.csv
```

The loader estimates pair value conservatively. If exact `k` is unavailable but
a visible public score exists, it marks the k estimate as approximate. Known
dead/crowded pairs such as `(25000, 2)`, `(24979, 4)`, `(9993, 0)`, and similar
attractors are hard-blacklisted pairwise rather than banning a whole `T` group.

Inspect:

```text
leaderboard_context.csv
route_scores.csv
```

### Obstruction atlas

`src/mathgraph_igp24/obstruction_atlas.py` names repeated harmful geometry:

- `reciprocal_collapse`
- `low_r_gravel`
- `crowded_attractor`
- `irreducibility_failure`
- `duplicate_signature`
- `baseline_dead_zone`
- `overfit_known_lane`
- `false_virginity`
- `high_r_failure`
- `support_singularity`

These obstructions reduce route scores and select escape constructors such as
`reciprocal_breaker`, `quotient_escape`, `high_r_lift`, and
`virgin_support_probe`.

### Run v109 dry

```bash
python scripts/run_v109_survivor_cycle.py \
  --root /content/drive/MyDrive/MathGraph_IGP24 \
  --mode mixed \
  --candidate-count 250000 \
  --dry-run
```

Smoke:

```bash
python scripts/run_v109_survivor_cycle.py --root /tmp/igp24 --candidate-count 2000 --dry-run
```

Dry output:

```text
v109_survivor_geometry_cycle/
  submission.txt
  selected_meta.json
  candidate_pool.csv
  portfolio_report.json
  basin_atlas.csv
  survivor_records.csv
  basin_summary.json
  obstruction_atlas.csv
  obstruction_atlas.json
  route_scores.csv
  phase_boundaries.csv
  selected_routes.json
  leaderboard_context.csv
  api_submit_cleaned/submitted_valid_polys.txt
```

### Run v109 live

```bash
export SAIR_API_KEY="..."
python scripts/run_v109_survivor_cycle.py \
  --root /content/drive/MyDrive/MathGraph_IGP24 \
  --mode mixed \
  --candidate-count 250000 \
  --submit \
  --poll
```

The API key is sanitized and never printed. The runner retries transient
`429/500/502/503/504`-style failures and saves partial API artifacts under
`api_submit_cleaned/`.

### Analyze survivor geometry

```bash
python scripts/analyze_survivor_geometry.py \
  --root /content/drive/MyDrive/MathGraph_IGP24 \
  --mode mixed
```

This writes:

```text
v109_survivor_geometry_analysis/
  basin_atlas.csv
  obstruction_atlas.csv
  route_scores.csv
  phase_boundaries.csv
  leaderboard_context.csv
  run_recommendations.txt
```

### Colab

Build artifacts:

```bash
python scripts/build_colab_artifact.py
```

Run smoke in Colab:

```python
%run dist/mathgraph_igp24_v109_survivor_colab.py --root /content/drive/MyDrive/MathGraph_IGP24 --mode mixed --candidate-count 2000 --dry-run
```

Run live in Colab:

```python
%env SAIR_API_KEY=your_key_here
%run dist/mathgraph_igp24_v109_survivor_colab.py --root /content/drive/MyDrive/MathGraph_IGP24 --mode mixed --candidate-count 250000 --submit --poll
```

After API results, rerun the same command. The new `submission_final.json` /
`selected_meta.json` evidence is ingested into the next survivor atlas, so the
cycle compounds batch after batch.
