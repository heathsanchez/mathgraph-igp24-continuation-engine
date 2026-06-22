from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .models import Polynomial
from .polynomial import parse_poly, poly_to_line

MAX_POLYNOMIALS = 100
MAX_BYTES = 100_000


def build_submission(polynomials: Sequence[Polynomial]) -> str:
    if len(polynomials) > MAX_POLYNOMIALS:
        raise ValueError("IGP24 submissions contain at most 100 polynomials")
    lines = [poly_to_line(poly) for poly in polynomials]
    if len(lines) != len(set(lines)):
        raise ValueError("submission contains duplicate polynomials")
    text = "\n".join(lines) + ("\n" if lines else "")
    if len(text.encode("utf-8")) > MAX_BYTES:
        raise ValueError("submission exceeds 100000 bytes")
    return text


def read_submission(path: str | Path) -> list[Polynomial]:
    result = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            result.append(parse_poly(stripped))
    return result


def write_submission(path: str | Path, polynomials: Sequence[Polynomial]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build_submission(polynomials), encoding="utf-8")

