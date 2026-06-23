"""Provenance-facing public helpers.

This module is deliberately thin: the durable implementation lives in
``trials.py``.  Keeping this facade gives notebooks and scripts a stable place
to import the continuation spine without turning empirical objects into proof
claims.
"""

from __future__ import annotations

from .models import MutationTrial, Outcome
from .trials import TrialLedger, create_trial, join_result

__all__ = ["MutationTrial", "Outcome", "TrialLedger", "create_trial", "join_result"]
