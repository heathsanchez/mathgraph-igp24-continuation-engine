from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mathgraph_igp24.api import SairClient, sanitize_api_key
from mathgraph_igp24.leaderboard import context_rows, load_leaderboard_context
from mathgraph_igp24.obstruction_atlas import learn_obstruction_atlas, write_obstruction_atlas
from mathgraph_igp24.portfolio_v2 import (
    generate_survivor_candidates,
    parents_from_records,
    select_survivor_portfolio,
    write_portfolio,
)
from mathgraph_igp24.submission import read_submission
from mathgraph_igp24.survivor_atlas import build_survivor_atlas, write_survivor_atlas
from mathgraph_igp24.survivor_router import recommend_routes, write_routes


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resilient_call(fn, *args, retries: int = 5, sleep_seconds: int = 10, **kwargs):
    last = None
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last = exc
            text = str(exc)
            if not any(code in text for code in ("429", "500", "502", "503", "504", "timed out", "timeout")):
                raise
            time.sleep(sleep_seconds * (attempt + 1))
    raise RuntimeError(f"API retries exhausted: {last}") from last


def join_selected_meta(out: Path, final: dict[str, Any]) -> None:
    selected_path = out / "selected_meta.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8")) if selected_path.exists() else []
    by_index = {int(row["index"]): row for row in selected if row.get("index") is not None}
    for key, status in (("verifiedPolynomials", "VERIFIED_BY_API"), ("failedPolynomials", "OBSTRUCTED")):
        for row in final.get(key, []) or []:
            index = row.get("polynomialIndex", row.get("polynomial_index", row.get("index")))
            if index is None or int(index) not in by_index:
                continue
            by_index[int(index)].update({"api_result": row, "trust_label": status})
    selected_path.write_text(json.dumps(selected, indent=2, sort_keys=True, default=str), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    out = root / args.out_name
    out.mkdir(parents=True, exist_ok=True)
    leaderboard = load_leaderboard_context(root, args.leaderboard_path)
    records, basin_summary = build_survivor_atlas(root, out, leaderboard)
    write_csv(out / "leaderboard_context.csv", context_rows(leaderboard))
    obstructions = learn_obstruction_atlas(records, basin_summary, leaderboard)
    write_obstruction_atlas(out, obstructions)
    routes = recommend_routes(basin_summary, leaderboard, obstructions, target_count=200, mode=args.mode)
    write_routes(out, routes)
    parents = parents_from_records(records)
    candidates = generate_survivor_candidates(parents, routes, leaderboard, args.candidate_count, args.seed, args.mode)
    selected = select_survivor_portfolio(candidates, lane_exploit=args.mode == "lane_24648")
    report = write_portfolio(out, selected, candidates)
    summary = {
        "out": str(out),
        "records_loaded": len(records),
        "basins": len(basin_summary),
        "obstructions": len(obstructions),
        "routes": len(routes),
        "candidates": len(candidates),
        "selected": len(selected),
        "submitted": False,
        "mode": args.mode,
        "obstructions_avoided": report.get("obstruction_avoidance_count", {}),
        "constructor_rationale": report.get("constructor_rationale", {}),
    }
    (out / "run_report.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    if args.dry_run or not args.submit:
        return summary

    key = sanitize_api_key(os.environ.get(args.api_key_env, ""))
    if not key:
        raise RuntimeError(f"{args.api_key_env} is required for --submit")
    client = SairClient(key, base_url=os.environ.get("SAIR_API_BASE", "https://api.sair.foundation/api/public/v1"))
    polynomials = read_submission(out / "submission.txt")
    response = resilient_call(client.submit, polynomials, f"MathGraph v109 survivor geometry {args.mode}")
    submission_id = str(response.get("submissionId") or response.get("id") or "")
    summary.update({"submitted": True, "submission_id": submission_id, "submit_response": response})
    (out / "api_submit_cleaned" / "submit_response.json").write_text(json.dumps(response, indent=2, sort_keys=True, default=str), encoding="utf-8")
    if args.poll and submission_id:
        final = resilient_call(client.poll, submission_id, args.max_polls, args.poll_seconds)
        (out / "api_submit_cleaned" / "submission_final.json").write_text(json.dumps(final, indent=2, sort_keys=True, default=str), encoding="utf-8")
        join_selected_meta(out, final)
    (out / "run_report.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.environ.get("MATHGRAPH_ROOT", "/content/drive/MyDrive/MathGraph_IGP24"))
    parser.add_argument("--out-name", default="v109_survivor_geometry_cycle")
    parser.add_argument("--candidate-count", type=int, default=int(os.environ.get("MATHGRAPH_CANDIDATES", "250000")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("MATHGRAPH_SEED", "109")))
    parser.add_argument("--submit", action="store_true", default=os.environ.get("SAIR_SUBMIT", "0") == "1")
    parser.add_argument("--poll", action="store_true", default=os.environ.get("SAIR_POLL", "0") == "1")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--api-key-env", default="SAIR_API_KEY")
    parser.add_argument("--leaderboard-path", default=None)
    parser.add_argument("--max-polls", type=int, default=160)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--mode", choices=["survivor", "high_r", "virgin", "lane_24648", "discriminant_minimize", "mixed"], default="mixed")
    args, _unknown = parser.parse_known_args(argv)
    run(args)


if __name__ == "__main__":
    main()
