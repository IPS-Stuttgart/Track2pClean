from __future__ import annotations

import pytest
from bayescatrack._advanced_weight_validation import (
    _finite_nonnegative_float,
    _finite_positive_float,
)


class _BadFloat:
    def __float__(self) -> float:
        raise ArithmeticError("bad numeric conversion")


def test_advanced_nonnegative_weight_normalizes_arithmetic_float_error() -> None:
    with pytest.raises(ValueError, match="radial_profile_weight"):
        _finite_nonnegative_float(_BadFloat(), name="radial_profile_weight")


def test_advanced_positive_weight_normalizes_arithmetic_float_error() -> None:
    with pytest.raises(ValueError, match="large_cost"):
        _finite_positive_float(_BadFloat(), name="large_cost")
