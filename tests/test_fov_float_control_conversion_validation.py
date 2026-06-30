from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.fov_registration import estimate_subpixel_fov_shift


class _ArithmeticFloat:
    def __float__(self) -> float:
        raise ArithmeticError("bad float conversion")


def test_estimate_subpixel_fov_shift_normalizes_arithmetic_refinement_radius_errors():
    reference_fov = np.zeros((8, 8), dtype=float)
    reference_fov[2:5, 2:5] = 1.0
    measurement_fov = reference_fov.copy()

    with pytest.raises(ValueError, match="refinement_radius"):
        estimate_subpixel_fov_shift(
            reference_fov,
            measurement_fov,
            refinement_radius=_ArithmeticFloat(),
        )
