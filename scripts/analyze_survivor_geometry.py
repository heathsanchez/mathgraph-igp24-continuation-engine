from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mathgraph_igp24.leaderboard import context_rows, load_leaderboard_context
from mathgraph_igp24.obstruction_atlas import learn_obstruction_atlas, write_obstruction_atlas
from mathgraph_igp24.survivor_atlas import build_survivor_atlas
from mathgraph_igp24.survivor_router import recommend_routes, write_routes

from run_v109_survivor_cycle import write_csv


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/content/drive/MyDrive/MathGraph_IGP24")
    parser.add_argument("--out-name", default="v109_survivor_geometry_analysis")
    parser.add_argument("--leaderboard-path", default=None)
    parser.add_argument("--mode", default="mixed")
    args, _unknown = parser.parse_known_args(argv)
    out = Path(args.root) / args.out_name
    leaderboard = load_leaderboard_context(args.root, args.leaderboard_path)
    records, basin_summary = build_survivor_atlas(args.root, out, leaderboard)
    write_csv(out / "leaderboard_context.csv", context_rows(leaderboard))
    obstructions = learn_obstruction_atlas(records, basin_summary, leaderboard)
    write_obstruction_atlas(out, obstructions)
    routes = recommend_routes(basin_summary, leaderboard, obstructions, 100, args.mode)
    write_routes(out, routes)
    lines = [
        "MathGraph IGP24 v109 survivor geometry recommendations",
        "",
        f"records: {len(records)}",
        f"basins: {len(basin_summary)}",
        f"obstructions: {len(obstructions)}",
        "",
        "Top survivor basins:",
        *[f"- {row['basin_id']} survival={row['survival_rate']:.3f} novelty={row['novelty_score']:.3f}" for row in basin_summary[:5]],
        "",
        "Top dead/crowded obstructions:",
        *[f"- {obs.label}: evidence={obs.evidence_count} escape={obs.suggested_escape_operator}" for obs in obstructions[:8]],
        "",
        "Top phase boundaries/routes:",
        *[f"- {route.route_id} score={route.route_score:.5f} escape={route.intended_escape}" for route in routes[:8]],
        "",
        "What to run next:",
        "python scripts/run_v109_survivor_cycle.py --mode mixed --candidate-count 250000 --dry-run",
    ]
    (out / "run_recommendations.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
