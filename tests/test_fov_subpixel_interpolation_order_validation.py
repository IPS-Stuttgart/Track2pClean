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


def _example_fovs() -> tuple[np.ndarray, np.ndarray]:
    reference_fov = np.zeros((8, 9), dtype=float)
    reference_fov[2:5, 3:7] = 1.0
    return reference_fov, reference_fov.copy()


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
    reference_fov, measurement_fov = _example_fovs()

    with pytest.raises(ValueError, match="subpixel interpolation order"):
        estimate_subpixel_fov_shift(
            reference_fov,
            measurement_fov,
            interpolation_order=bad_order,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "bad_order",
    [
        "1",
        b"1",
        bytearray(b"1"),
        memoryview(b"1"),
        np.str_("1"),
        np.bytes_(b"1"),
        np.asarray("1"),
    ],
)
def test_subpixel_interpolation_order_rejects_text_like_controls(
    bad_order: object,
) -> None:
    reference_fov, measurement_fov = _example_fovs()

    with pytest.raises(ValueError, match="subpixel interpolation order"):
        estimate_subpixel_fov_shift(
            reference_fov,
            measurement_fov,
            interpolation_order=bad_order,  # type: ignore[arg-type]
        )
