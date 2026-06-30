from __future__ import annotations

import pytest
from bayescatrack.core import _loader_validation


class _BadFloat:
    def __float__(self) -> float:
        raise ArithmeticError("bad float")


def test_loader_control_probability_wraps_bad_float_errors() -> None:
    validate_loader_controls = getattr(
        _loader_validation,
        "_" + "validate_suite2p_loader_controls",
    )

    with pytest.raises(ValueError, match="cell_probability_threshold"):
        validate_loader_controls({"cell_probability_threshold": _BadFloat()})
