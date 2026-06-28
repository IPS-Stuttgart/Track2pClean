from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association.pyrecest_global_assignment import (
    registered_soft_iou_cost_kwargs,
)
from bayescatrack.core.bridge import CalciumPlaneData


class _OverflowingFloat:
    def __float__(self) -> float:
        raise OverflowError("too large")


def _single_roi_plane(mask: np.ndarray) -> CalciumPlaneData:
    roi_masks = np.asarray(mask, dtype=float).reshape(1, 1, -1)
    return CalciumPlaneData(roi_masks=roi_masks)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"distance_transform_overlap_weight": "not-a-number"},
            "distance_transform_overlap_weight must be a finite non-negative value",
        ),
        (
            {"distance_transform_overlap_weight": _OverflowingFloat()},
            "distance_transform_overlap_weight must be a finite non-negative value",
        ),
        (
            {"similarity_epsilon": object()},
            "similarity_epsilon must be a finite positive value",
        ),
    ],
)
def test_registered_soft_iou_preset_rejects_invalid_numeric_controls(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        registered_soft_iou_cost_kwargs(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"soft_iou_weight": "not-a-number"},
            "soft_iou_weight must be a finite non-negative value",
        ),
        (
            {"large_cost": object()},
            "large_cost must be a finite positive value",
        ),
        (
            {"large_cost": _OverflowingFloat()},
            "large_cost must be a finite positive value",
        ),
    ],
)
def test_soft_overlap_runtime_rejects_invalid_numeric_controls(
    kwargs: dict[str, object], message: str
) -> None:
    reference = _single_roi_plane(np.array([1.0, 0.0, 0.0]))
    measurement = _single_roi_plane(np.array([1.0, 0.0, 0.0]))
    common_kwargs = {
        "centroid_weight": 0.0,
        "iou_weight": 0.0,
        "mask_cosine_weight": 0.0,
        "area_weight": 0.0,
        "roi_feature_weight": 0.0,
        "cell_probability_weight": 0.0,
        "soft_iou_weight": 1.0,
        "soft_iou_radius": 1,
        "distance_transform_overlap_weight": 0.5,
        "distance_transform_overlap_radius": 1,
    }
    common_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        reference.build_pairwise_cost_matrix(measurement, **common_kwargs)
