import pytest

from mathgraph_igp24 import parse_poly, poly_to_line, valid_poly


def test_poly_line_roundtrip_uses_official_commas(parent):
    line = poly_to_line(parent)
    assert line.count(",") == 24
    assert parse_poly(line) == parent


def test_valid_poly_rejects_zero_constant(parent):
    invalid = (0,) + parent[1:]
    assert not valid_poly(invalid)
    with pytest.raises(ValueError):
        parse_poly(invalid)

