from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Iterable, Sequence

from .models import Polynomial

DEGREE = 24
COEFFICIENT_COUNT = 25
INTEGER_RE = re.compile(r"[-+]?\d+")


def gcd_many(values: Iterable[int]) -> int:
    result = 0
    for value in values:
        result = math.gcd(result, abs(int(value)))
    return result


def normalize_poly(poly: Sequence[int]) -> Polynomial:
    if len(poly) != COEFFICIENT_COUNT:
        raise ValueError("degree-24 polynomials require exactly 25 coefficients")
    normalized = tuple(int(value) for value in poly)
    if not valid_poly(normalized):
        raise ValueError("polynomial must be integral, monic, primitive, and have nonzero constant term")
    return normalized


def valid_poly(poly: Sequence[Any]) -> bool:
    return (
        len(poly) == COEFFICIENT_COUNT
        and all(isinstance(value, int) and not isinstance(value, bool) for value in poly)
        and poly[0] != 0
        and poly[-1] == 1
        and gcd_many(poly) == 1
    )


def poly_to_line(poly: Sequence[int]) -> str:
    normalized = normalize_poly(poly)
    return ",".join(str(value) for value in normalized)


def parse_poly(value: str | Sequence[int]) -> Polynomial:
    if isinstance(value, str):
        without_comment = value.split("#", 1)[0]
        coefficients = tuple(int(token) for token in INTEGER_RE.findall(without_comment))
    else:
        coefficients = tuple(int(token) for token in value)
    return normalize_poly(coefficients)


def hash_poly(poly: Sequence[int]) -> str:
    return hashlib.sha256(poly_to_line(poly).encode("utf-8")).hexdigest()

