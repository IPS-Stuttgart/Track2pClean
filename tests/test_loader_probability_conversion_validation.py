from __future__ import annotations

import pytest
from bayescatrack.core import _loader_validation


class _ArithmeticFloat:
    def __float__(self) -> float:
        raise ArithmeticError("bad numeric conversion")


def test_loader_probability_validation_wraps_arithmetic_float_errors() -> None:
    validate_loader_controls = getattr(
        _loader_validation,
        "_validate_suite2p_loader_controls",
    )

    with pytest.raises(ValueError, match="cell_probability_threshold"):
        validate_loader_controls({"cell_probability_threshold": _ArithmeticFloat()})
