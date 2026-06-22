"""Public contracts for the MathGraph IGP24 continuation engine."""

from .basins import Basin, build_basin_atlas
from .fingerprints import basin_id, fingerprint
from .lawbook import LawBook
from .laws import learn_laws
from .memory import EmpiricalMemory, build_memory
from .models import (
    BasinLaw,
    Evidence,
    MutationTrial,
    Obstruction,
    Outcome,
    Pair,
    Polynomial,
    RouteRecommendation,
)
from .obstructions import learn_obstructions
from .polynomial import hash_poly, normalize_poly, parse_poly, poly_to_line, valid_poly
from .replay import replay_law
from .router import recommend
from .submission import build_submission, read_submission, write_submission
from .trials import TrialLedger, create_trial, join_result

__all__ = [
    "Basin", "BasinLaw", "EmpiricalMemory", "Evidence", "LawBook", "MutationTrial",
    "Obstruction", "Outcome", "Pair", "Polynomial", "RouteRecommendation", "TrialLedger",
    "basin_id", "build_basin_atlas", "build_memory", "build_submission", "create_trial",
    "fingerprint", "hash_poly", "join_result", "learn_laws", "learn_obstructions",
    "normalize_poly", "parse_poly", "poly_to_line", "read_submission", "recommend",
    "replay_law", "valid_poly", "write_submission",
]

