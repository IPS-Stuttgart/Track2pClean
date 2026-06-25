from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from bayescatrack import CalciumPlaneData


def _single_roi_plane() -> CalciumPlaneData:
    masks = np.zeros((1, 4, 4), dtype=bool)
    masks[0, 1:3, 1:3] = True
    return CalciumPlaneData(masks)


@pytest.mark.parametrize("regularization", [True, np.bool_(False), np.nan, np.inf, -1.0])
def test_position_covariances_reject_invalid_regularization(regularization: Any) -> None:
    with pytest.raises(ValueError, match="regularization must be a finite non-negative value"):
        _single_roi_plane().position_covariances(regularization=regularization)


@pytest.mark.parametrize("velocity_variance", [True, np.bool_(False), np.nan, np.inf, -1.0])
def test_constant_velocity_state_moments_reject_invalid_velocity_variance(
    velocity_variance: Any,
) -> None:
    with pytest.raises(ValueError, match="velocity_variance must be a finite non-negative value"):
        _single_roi_plane().to_constant_velocity_state_moments(
            velocity_variance=velocity_variance,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"centroid_weight": True}, "centroid_weight must be a finite non-negative value"),
        ({"iou_weight": np.nan}, "iou_weight must be a finite non-negative value"),
        ({"mask_cosine_weight": np.inf}, "mask_cosine_weight must be a finite non-negative value"),
        ({"large_cost": np.inf}, "large_cost must be a finite strictly positive value"),
        ({"similarity_epsilon": np.nan}, "similarity_epsilon must be a finite strictly positive value"),
        ({"centroid_scale": 0.0}, "centroid_scale must be a finite strictly positive value"),
        ({"max_centroid_distance": np.bool_(True)}, "max_centroid_distance must be a finite strictly positive value"),
    ],
)
def test_pairwise_cost_rejects_invalid_core_scalar_controls(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    plane = _single_roi_plane()

    with pytest.raises(ValueError, match=message):
        plane.build_pairwise_cost_matrix(plane, **kwargs)
