from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence

from .models import Polynomial
from .polynomial import normalize_poly


@dataclass(frozen=True)
class MutationResult:
    name: str
    parameters: dict[str, Any]
    description: str
    child: Polynomial
    coefficient_delta: tuple[int, ...]


def _apply_edits(parent: Polynomial, edits: Sequence[dict[str, int]]) -> Polynomial:
    child = list(parent)
    for edit in edits:
        index = int(edit["index"])
        if not 0 <= index < 24:
            raise ValueError("mutations may edit only a0..a23")
        if "value" in edit:
            child[index] = int(edit["value"])
        else:
            child[index] += int(edit["delta"])
    if child[0] == 0:
        child[0] = 1
    return normalize_poly(child)


def _materialize(name: str, parent: Polynomial, parameters: dict[str, Any], seed: int) -> tuple[list[dict[str, int]], str]:
    rng = random.Random(seed)
    if "edits" in parameters:
        return [dict(edit) for edit in parameters["edits"]], str(parameters.get("description", name))
    aliases = {
        "support_toggle": "support_add" if seed % 2 else "support_drop",
        "coefficient_scale": "high_sparse_scale",
        "even_lacunary": "even_lacunary_shift",
        "center_peak": "center_mass_push",
        "high_sparse": "high_sparse_perturb",
        "transition_mutation": "basin_guided",
        "local_noise": "random_exploration",
    }
    name = aliases.get(name, name)
    support = [index for index in range(24) if parent[index] != 0]
    zeros = [index for index in range(1, 24) if parent[index] == 0]
    if name == "center_mass_push":
        index = int(parameters.get("index", 12)); delta = int(parameters.get("delta", 18))
        return [{"index": index, "delta": delta}], f"increase center mass by adding {delta} to a{index}"
    if name == "a0_sweep":
        delta = int(parameters.get("delta", rng.choice([-8, -4, -2, 2, 4, 8])))
        if parent[0] + delta == 0: delta += 1
        return [{"index": 0, "delta": delta}], f"sweep constant coefficient by {delta}"
    if name == "sign_flip":
        choices = [index for index in support if index not in (0, 24)] or [0]
        index = int(parameters.get("index", rng.choice(choices)))
        return [{"index": index, "value": -parent[index]}], f"flip sign of a{index}"
    if name == "sparse_keep_shape":
        choices = support or [0]; index = int(parameters.get("index", rng.choice(choices)))
        delta = int(parameters.get("delta", rng.choice([-3, -1, 1, 3])))
        return [{"index": index, "delta": delta}], f"perturb supported coefficient a{index} by {delta}"
    if name == "support_add":
        index = int(parameters.get("index", rng.choice(zeros or [1]))); value = int(parameters.get("value", rng.choice([-3, -1, 1, 3])))
        return [{"index": index, "value": value}], f"add support at a{index} with value {value}"
    if name == "support_drop":
        choices = [index for index in support if index != 0] or [min(23, max(1, support[0] if support else 1))]
        index = int(parameters.get("index", rng.choice(choices)))
        return [{"index": index, "value": 0}], f"drop support at a{index}"
    if name == "even_lacunary_shift":
        index = int(parameters.get("index", rng.choice(list(range(2, 24, 2)))))
        delta = int(parameters.get("delta", rng.choice([-8, -4, 4, 8])))
        return [{"index": index, "delta": delta}], f"shift even lacunary coefficient a{index} by {delta}"
    if name == "high_sparse_scale":
        factor = int(parameters.get("factor", 2)); choices = [index for index in support if index != 24] or [0]
        edits = [{"index": index, "value": parent[index] * factor} for index in choices]
        return edits, f"scale supported non-leading coefficients by {factor}"
    if name == "high_sparse_perturb":
        choices = [index for index in support if index != 24] or [0]; index = int(parameters.get("index", rng.choice(choices)))
        delta = int(parameters.get("delta", rng.choice([-17, -7, 7, 17])))
        return [{"index": index, "delta": delta}], f"perturb high sparse coefficient a{index} by {delta}"
    if name == "basin_clone":
        indices = parameters.get("indices") or rng.sample(range(1, 24), k=3)
        deltas = parameters.get("deltas") or [rng.choice([-1, 1]) for _ in indices]
        return [{"index": int(index), "delta": int(delta)} for index, delta in zip(indices, deltas)], "clone basin shape with micro-edits"
    if name == "basin_guided":
        edits = parameters.get("target_edits") or [{"index": 12, "delta": 4}, {"index": 6, "delta": -1}]
        return [dict(edit) for edit in edits], "apply basin-guided coefficient edits"
    if name == "law_replay":
        deltas = tuple(int(value) for value in parameters.get("coefficient_deltas", ()))
        if len(deltas) != 25: raise ValueError("law_replay requires 25 coefficient_deltas")
        return [{"index": index, "delta": delta} for index, delta in enumerate(deltas[:24]) if delta], f"replay law {parameters.get('law_id', '')}"
    if name == "random_exploration":
        count = int(parameters.get("count", 3)); indices = parameters.get("indices") or rng.sample(range(1, 24), k=count)
        deltas = parameters.get("deltas") or [rng.randint(-10, 10) or 1 for _ in indices]
        return [{"index": int(index), "delta": int(delta)} for index, delta in zip(indices, deltas)], "seeded exploration edits"
    raise KeyError(f"unknown mutation operator {name!r}")


OPERATORS = (
    "center_mass_push", "a0_sweep", "sign_flip", "support_toggle", "sparse_keep_shape",
    "coefficient_scale", "basin_clone", "local_noise", "even_lacunary", "center_peak",
    "high_sparse", "transition_mutation", "support_add", "support_drop", "even_lacunary_shift",
    "high_sparse_scale", "high_sparse_perturb", "basin_guided", "law_replay", "random_exploration",
)


def apply_mutation(parent: Sequence[int], name: str, parameters: dict[str, Any] | None = None, seed: int = 0) -> MutationResult:
    normalized = normalize_poly(parent)
    edits, description = _materialize(name, normalized, dict(parameters or {}), seed)
    child = _apply_edits(normalized, edits)
    frozen_parameters = {**dict(parameters or {}), "edits": edits, "description": description, "seed": seed}
    return MutationResult(name, frozen_parameters, description, child, tuple(b - a for a, b in zip(normalized, child)))


def replay_mutation(parent: Sequence[int], name: str, parameters: dict[str, Any]) -> Polynomial:
    return apply_mutation(parent, name, parameters, int(parameters.get("seed", 0))).child
