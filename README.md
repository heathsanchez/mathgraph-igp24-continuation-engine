# MathGraph IGP24 Continuation Engine

MathGraph IGP24 v101 is a provenance-bearing continuation engine for the
degree-24 inverse Galois problem competition.

Its core record is:

```text
Parent → Mutation → Child → Submission → Observed outcome
```

From replayable mutation trials it learns empirical basin laws, identifies
named obstructions, and builds inverse routes toward desired `(24Tt, r)` pairs.
Empirical routing objects are explicitly labelled and are not represented as
verified mathematical theorems.

## Outputs

- `trials.jsonl`
- `submission.txt`
- `api_results.json`
- `trial_certificates.csv`
- `basin_laws.csv`
- `obstructions.csv`
- `inverse_routes.csv`
- `lawbook.json`

## Dry run

The default is one safe cycle without an API submission:

```bash
python mathgraph_igp24_v101_provenance_bearing_continuation_engine.py
```

Use a smaller candidate count for a quick smoke run:

```bash
MATHGRAPH_CANDIDATES=5000 python mathgraph_igp24_v101_provenance_bearing_continuation_engine.py
```

## Continuous SAIR operation

```bash
export SAIR_API_KEY="..."
export SAIR_AUTORUN=1
export MATHGRAPH_MAX_CYCLES=0
python mathgraph_igp24_v101_provenance_bearing_continuation_engine.py
```

Before every submission, the engine checks the live competition specification
and team eligibility. It stops or backs off when the API reports that
submission is blocked.

## Tests

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

