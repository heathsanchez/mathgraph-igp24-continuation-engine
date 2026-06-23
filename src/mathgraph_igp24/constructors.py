from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

from .polynomial import hash_poly, normalize_poly, valid_poly
from .survivor_atlas import basin_id_from_signature, basin_signature, reciprocal_score, support_tuple

Polynomial = tuple[int, ...]


@dataclass(frozen=True)
class ConstructedCandidate:
    parent: Polynomial
    child: Polynomial
    parent_hash: str
    child_hash: str
    constructor_name: str
    constructor_params: dict[str, Any]
    source_basin_id: str
    predicted_profile: str
    obstruction_avoided: str
    generation_seed: int
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["parent"] = ",".join(map(str, self.parent))
        row["child"] = ",".join(map(str, self.child))
        return row


@dataclass(frozen=True)
class Constructor:
    name: str
    expected_effect: str
    obstruction_addressed: str
    generate_fn: Callable[[Polynomial, random.Random], tuple[Polynomial, dict[str, Any]]]

    def generate(self, parent: Sequence[int], rng: random.Random, seed: int = 0, predicted_profile: str = "unknown") -> ConstructedCandidate:
        normalized = normalize_poly(parent)
        child, params = self.generate_fn(normalized, rng)
        child = _repair(child)
        source_basin = basin_id_from_signature(basin_signature(normalized, self.name))
        return ConstructedCandidate(
            parent=normalized,
            child=child,
            parent_hash=hash_poly(normalized),
            child_hash=hash_poly(child),
            constructor_name=self.name,
            constructor_params=params,
            source_basin_id=source_basin,
            predicted_profile=predicted_profile,
            obstruction_avoided=self.obstruction_addressed,
            generation_seed=seed,
            explanation=self.explain(normalized, child),
        )

    def explain(self, parent: Polynomial, child: Polynomial) -> str:
        edits = [f"a{index}:{left}->{right}" for index, (left, right) in enumerate(zip(parent, child)) if left != right]
        return f"{self.name}: {self.expected_effect}; edits={';'.join(edits[:8])}"


def _repair(values: Sequence[int]) -> Polynomial:
    child = list(values[:25])
    child += [0] * (25 - len(child))
    child[24] = 1
    if child[0] == 0:
        child[0] = 1
    if not valid_poly(tuple(child)):
        child[0] = child[0] + 1 if child[0] > 0 else child[0] - 1
    return normalize_poly(child)


def _edit(parent: Polynomial, edits: list[tuple[int, int | None, int | None]]) -> Polynomial:
    child = list(parent)
    for index, value, delta in edits:
        if not 0 <= index < 24:
            continue
        child[index] = int(value) if value is not None else child[index] + int(delta or 0)
    return _repair(child)


def reciprocal_breaker(parent: Polynomial, rng: random.Random):
    left = rng.choice([3, 5, 7, 9, 11])
    right = 24 - left
    delta = rng.choice([-7, -5, -3, 3, 5, 7])
    return _edit(parent, [(left, None, delta), (right, None, -delta // 2)]), {"left": left, "right": right, "delta": delta}


def asymmetry_injection(parent: Polynomial, rng: random.Random):
    index = rng.choice([1, 3, 5, 7, 17, 19, 21, 23])
    value = rng.choice([-5, -3, -1, 1, 3, 5])
    return _edit(parent, [(index, value, None)]), {"index": index, "value": value}


def support_transport(parent: Polynomial, rng: random.Random):
    support = [i for i in support_tuple(parent) if 1 <= i < 24] or [12]
    src = rng.choice(support)
    choices = [min(23, max(1, src + shift)) for shift in (-2, -1, 1, 2)]
    choices = [choice for choice in choices if choice != src] or [2 if src != 2 else 3]
    dst = rng.choice(choices)
    return _edit(parent, [(dst, parent[src], None), (src, 0, None)]), {"src": src, "dst": dst}


def high_r_lift(parent: Polynomial, rng: random.Random):
    index = rng.choice([10, 11, 12, 13, 14])
    delta = rng.choice([-24, -16, 16, 24])
    side = rng.choice([4, 6, 18, 20])
    return _edit(parent, [(index, None, delta), (side, None, -delta // 4)]), {"center": index, "side": side, "delta": delta}


def parity_bridge(parent: Polynomial, rng: random.Random):
    odd = rng.choice([1, 3, 5, 7, 9, 15, 17, 19, 21, 23])
    even = rng.choice([2, 4, 6, 8, 10, 14, 16, 18, 20, 22])
    return _edit(parent, [(odd, rng.choice([-2, 2]), None), (even, None, rng.choice([-3, 3]))]), {"odd": odd, "even": even}


def separatrix_perturb(parent: Polynomial, rng: random.Random):
    edits = []
    for _ in range(3):
        edits.append((rng.randrange(1, 24), None, rng.choice([-2, -1, 1, 2])))
    return _edit(parent, edits), {"edits": edits}


def quotient_escape(parent: Polynomial, rng: random.Random):
    edits = [(2, None, rng.choice([-11, 11])), (12, None, rng.choice([-13, 13])), (22, None, rng.choice([-5, 5]))]
    return _edit(parent, edits), {"edits": edits}


def lane_exploit(parent: Polynomial, rng: random.Random):
    lane = rng.choice([24648, 16055])
    edits = [(6, None, rng.choice([-9, 9])), (12, None, rng.choice([-18, 18])), (18, None, rng.choice([-9, 9]))]
    return _edit(parent, edits), {"lane": lane, "edits": edits}


def virgin_support_probe(parent: Polynomial, rng: random.Random):
    indices = sorted(rng.sample(range(1, 24), k=rng.choice([5, 6, 7, 8])))
    edits = [(index, rng.choice([-4, -2, -1, 1, 2, 4]), None) for index in indices]
    return _edit(parent, edits), {"indices": indices}


def discriminant_minimizer(parent: Polynomial, rng: random.Random):
    support = [i for i in support_tuple(parent) if 1 <= i < 24] or [12]
    edits = []
    for index in rng.sample(support, k=min(3, len(support))):
        if parent[index] > 1:
            edits.append((index, None, -1))
        elif parent[index] < -1:
            edits.append((index, None, 1))
    return _edit(parent, edits or [(12, None, 1)]), {"edits": edits}


CONSTRUCTORS: dict[str, Constructor] = {
    "reciprocal_breaker": Constructor("reciprocal_breaker", "break reciprocal/palindromic collapse", "reciprocal_collapse", reciprocal_breaker),
    "asymmetry_injection": Constructor("asymmetry_injection", "inject mixed odd/even asymmetric support", "support_singularity", asymmetry_injection),
    "support_transport": Constructor("support_transport", "move mass to nearby support skeletons", "support_singularity", support_transport),
    "high_r_lift": Constructor("high_r_lift", "tilt central/sign geometry toward high-r attempts", "high_r_failure", high_r_lift),
    "parity_bridge": Constructor("parity_bridge", "bridge even-only and mixed parity spaces", "low_r_gravel", parity_bridge),
    "separatrix_perturb": Constructor("separatrix_perturb", "small perturbation near phase boundaries", "crowded_attractor", separatrix_perturb),
    "quotient_escape": Constructor("quotient_escape", "disrupt quotient/reciprocal crowded signatures", "crowded_attractor", quotient_escape),
    "lane_exploit": Constructor("lane_exploit", "controlled descendants for 24648/16055 lanes", "false_virginity", lane_exploit),
    "virgin_support_probe": Constructor("virgin_support_probe", "new support skeleton probe", "duplicate_signature", virgin_support_probe),
    "discriminant_minimizer": Constructor("discriminant_minimizer", "reduce height near known scoreable routes", "overfit_known_lane", discriminant_minimizer),
    "boundary_microshift": Constructor("boundary_microshift", "micro-shift around survivor phase boundaries", "crowded_attractor", separatrix_perturb),
    "odd_lane_probe": Constructor("odd_lane_probe", "probe odd support lanes away from even-only collapse", "low_r_gravel", asymmetry_injection),
    "height_trim_probe": Constructor("height_trim_probe", "trim coefficient height while preserving support shape", "overfit_known_lane", discriminant_minimizer),
    "wide_support_probe": Constructor("wide_support_probe", "wide new support skeleton probe", "duplicate_signature", virgin_support_probe),
    "center_shear_probe": Constructor("center_shear_probe", "shear central mass without reciprocal symmetry", "reciprocal_collapse", high_r_lift),
    "mixed_parity_probe": Constructor("mixed_parity_probe", "force mixed parity support with low reuse", "low_r_gravel", parity_bridge),
    "quotient_jitter_probe": Constructor("quotient_jitter_probe", "jitter quotient-like supports away from 25000/24979", "crowded_attractor", quotient_escape),
    "separatrix_wide_probe": Constructor("separatrix_wide_probe", "wide perturbation near survivor boundaries", "support_singularity", separatrix_perturb),
    "asymmetric_height_probe": Constructor("asymmetric_height_probe", "combine asymmetry and coefficient height variation", "false_virginity", asymmetry_injection),
}


def choose_constructor(name: str) -> Constructor:
    return CONSTRUCTORS.get(name, CONSTRUCTORS["virgin_support_probe"])


__all__ = ["ConstructedCandidate", "Constructor", "CONSTRUCTORS", "choose_constructor", "reciprocal_score"]
