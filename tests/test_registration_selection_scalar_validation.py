from __future__ import annotations

import numpy as np
import pytest
from bayescatrack._registration_selection_validation import (
    _finite_nonnegative_scalar,
    _finite_unit_interval_scalar,
)


@pytest.mark.parametrize(
    "malformed",
    (
        np.asarray(True),
        np.asarray(False, dtype=object),
        np.asarray("0.5"),
        np.asarray(b"0.5"),
        np.asarray("0.5", dtype=object),
        np.asarray(b"0.5", dtype=object),
    ),
)
def test_registration_selection_rejects_zero_dimensional_non_numeric_controls(
    malformed: object,
) -> None:
    with pytest.raises(ValueError, match="min_fov_correlation_gain"):
        _finite_nonnegative_scalar(malformed, "min_fov_correlation_gain")


@pytest.mark.parametrize(
    "valid",
    (
        0,
        0.5,
        np.float64(0.25),
        np.asarray(0.25),
        np.asarray(0.25, dtype=object),
    ),
)
def test_registration_selection_keeps_numeric_zero_dimensional_controls(
    valid: object,
) -> None:
    expected = float(np.asarray(valid, dtype=object).item())

    assert _finite_unit_interval_scalar(
        valid,
        "max_empty_roi_fraction",
    ) == pytest.approx(expected)
