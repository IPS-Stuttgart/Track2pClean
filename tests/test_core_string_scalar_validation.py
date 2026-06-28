from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from bayescatrack import CalciumPlaneData


def _plane() -> CalciumPlaneData:
    masks = np.zeros((1, 2, 2), dtype=float)
    masks[0, 0, 0] = 1.0
    return CalciumPlaneData(masks)


@pytest.mark.parametrize(
    ("control_name", "call"),
    [
        (
            "regularization",
            lambda plane: plane.position_covariances(regularization="1e-6"),
        ),
        (
            "velocity_variance",
            lambda plane: plane.to_constant_velocity_state_moments(
                velocity_variance=np.array("25.0"),
            ),
        ),
        (
            "centroid_weight",
            lambda plane: plane.build_pairwise_cost_matrix(
                plane,
                centroid_weight="1.0",
            ),
        ),
        (
            "large_cost",
            lambda plane: plane.build_pairwise_cost_matrix(
                plane,
                large_cost=b"100.0",
            ),
        ),
        (
            "max_centroid_distance",
            lambda plane: plane.build_pairwise_cost_matrix(
                plane,
                max_centroid_distance=np.str_("5.0"),
            ),
        ),
    ],
)
def test_core_numeric_scalar_controls_reject_string_like_values(
    control_name: str,
    call: Callable[[CalciumPlaneData], object],
) -> None:
    with pytest.raises(ValueError, match=rf"{control_name} must be"):
        call(_plane())
