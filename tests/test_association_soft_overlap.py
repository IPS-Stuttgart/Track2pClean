"""Tests for the legacy association soft-overlap wrapper."""

from __future__ import annotations

import numpy as np
import pytest
from bayescatrack import CalciumPlaneData
from bayescatrack.association.soft_overlap import (
    dilate_mask_stack,
    soft_iou_pairwise_cost_matrix,
)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"soft_iou_radius": True}, "soft_iou_radius"),
        ({"soft_iou_radius": 1.5}, "soft_iou_radius"),
        ({"soft_iou_radius": -1}, "soft_iou_radius"),
        ({"use_soft_iou_for_iou_cost": "false"}, "use_soft_iou_for_iou_cost"),
        ({"return_components": 1}, "return_components"),
        ({"similarity_epsilon": np.nan}, "similarity_epsilon"),
        ({"similarity_epsilon": 0.0}, "similarity_epsilon"),
        ({"large_cost": np.inf}, "large_cost"),
        ({"iou_weight": True}, "iou_weight"),
        ({"iou_weight": -0.1}, "iou_weight"),
    ],
)
def test_soft_iou_wrapper_rejects_invalid_runtime_controls(
    kwargs: dict[str, object], message: str
) -> None:
    reference = np.zeros((1, 5, 5), dtype=bool)
    measurement = np.zeros((1, 5, 5), dtype=bool)
    reference[0, 1:3, 1:3] = True
    measurement[0, 1:3, 2:4] = True
    reference_plane = CalciumPlaneData(reference)
    measurement_plane = CalciumPlaneData(measurement)

    def original_method(self, other, **method_kwargs):
        del self, other
        assert method_kwargs["return_components"] is True
        return np.zeros((1, 1), dtype=float), {
            "iou": np.zeros((1, 1), dtype=float),
            "gated": np.zeros((1, 1), dtype=bool),
        }

    common_kwargs: dict[str, object] = {
        "soft_iou_radius": 1,
        "use_soft_iou_for_iou_cost": True,
    }
    common_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        soft_iou_pairwise_cost_matrix(
            original_method,
            reference_plane,
            measurement_plane,
            **common_kwargs,
        )


@pytest.mark.parametrize("radius", [True, 1.5, np.nan])
def test_soft_overlap_dilation_rejects_non_integer_radius(radius: object) -> None:
    masks = np.zeros((1, 5, 5), dtype=bool)
    masks[0, 2, 2] = True

    with pytest.raises(ValueError, match="radius"):
        dilate_mask_stack(masks, radius=radius)
