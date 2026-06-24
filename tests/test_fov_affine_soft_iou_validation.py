"""Regression tests for FOV-affine soft-IoU radius validation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from bayescatrack.core.bridge import CalciumPlaneData
from bayescatrack.experiments.track2p_fov_affine_benchmark import (
    _soft_iou_pairwise_cost_matrix,
)


def _single_roi_plane(mask: np.ndarray) -> CalciumPlaneData:
    roi_masks = np.asarray(mask, dtype=float).reshape(1, 1, -1)
    return CalciumPlaneData(roi_masks=roi_masks)


def _unexpected_original_method(
    self: CalciumPlaneData,
    other: CalciumPlaneData,
    **kwargs: Any,
) -> np.ndarray:
    raise AssertionError(
        "invalid soft_iou_radius should be rejected before fallback cost computation"
    )


@pytest.mark.parametrize(
    ("soft_iou_radius", "message"),
    [
        (-1, "soft_iou_radius must be non-negative"),
        (True, "soft_iou_radius must be an integer"),
        (1.5, "soft_iou_radius must be an integer"),
        (np.inf, "soft_iou_radius must be an integer"),
    ],
)
def test_fov_affine_soft_iou_radius_rejects_invalid_values_before_iou_only_fast_path(
    soft_iou_radius: object,
    message: str,
) -> None:
    reference = _single_roi_plane(np.array([1.0, 0.0, 0.0]))
    measurement = _single_roi_plane(np.array([1.0, 0.0, 0.0]))

    with pytest.raises(ValueError, match=message):
        _soft_iou_pairwise_cost_matrix(
            _unexpected_original_method,
            reference,
            measurement,
            centroid_weight=0.0,
            iou_weight=1.0,
            mask_cosine_weight=0.0,
            area_weight=0.0,
            roi_feature_weight=0.0,
            cell_probability_weight=0.0,
            soft_iou_radius=soft_iou_radius,
        )
