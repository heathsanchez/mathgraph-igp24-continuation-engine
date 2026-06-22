# MathGraph IGP24 Continuation Engine

MathGraph IGP24 v102 is a replayable, provenance-bearing continuation engine for the
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
python scripts/build_colab.py
```

This deterministically produces:

```text
dist/mathgraph_igp24_v102_colab.py
dist/artifact_manifest.json
```

The manifest freezes the artifact hash and every embedded source hash. The
generated runner can be uploaded to Colab as a single `.py` file.

## Join an observed result

```python
observed = join_result(
    submission_id,
    polynomial_index,
    {"status": "ok", "computed_label": "24T14010", "computed_r": 8},
    ledger,
)
```
