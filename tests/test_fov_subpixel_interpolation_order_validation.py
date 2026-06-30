from __future__ import annotations

import bayescatrack  # noqa: F401
import numpy as np
import pytest
from bayescatrack.fov_registration import estimate_subpixel_fov_shift


class _IndexRaisesOverflow:
    def __index__(self) -> int:
        raise OverflowError("boom")


class _IndexRaisesValue:
    def __index__(self) -> int:
        raise ValueError("boom")


@pytest.mark.parametrize(
    "bad_order",
    [
        _IndexRaisesOverflow(),
        _IndexRaisesValue(),
    ],
)
def test_subpixel_interpolation_order_protocol_failures_are_value_errors(
    bad_order: object,
) -> None:
    reference_fov = np.zeros((8, 9), dtype=float)
    reference_fov[2:5, 3:7] = 1.0
    measurement_fov = reference_fov.copy()

    with pytest.raises(ValueError, match="subpixel interpolation order"):
        estimate_subpixel_fov_shift(
            reference_fov,
            measurement_fov,
            interpolation_order=bad_order,  # type: ignore[arg-type]
        )
