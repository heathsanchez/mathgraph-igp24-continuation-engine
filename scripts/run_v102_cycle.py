from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mathgraph_igp24.cycle import CycleConfig, run_cycle


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--target-t", type=int, default=14010)
    parser.add_argument("--target-r", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--candidate-count", type=int, default=None)
    parser.add_argument("--version", default=None)
    args, _unknown = parser.parse_known_args(argv)
    config = CycleConfig.from_env(
        root=args.root,
        target_pair=(args.target_t, args.target_r),
        seed=args.seed,
        candidate_count=args.candidate_count,
        version=args.version,
    )
    run_cycle(config)


if __name__ == "__main__":
    main()
