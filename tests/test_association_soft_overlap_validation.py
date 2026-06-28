"""Regression tests for association soft-overlap validation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from bayescatrack.association.soft_overlap import (
    dilate_mask_stack,
    soft_iou_pairwise_cost_matrix,
)
from bayescatrack.core.bridge import CalciumPlaneData


def _single_roi_plane(mask: np.ndarray) -> CalciumPlaneData:
    roi_masks = np.asarray(mask, dtype=float).reshape(1, 1, -1)
    return CalciumPlaneData(roi_masks=roi_masks)


def _unexpected_original_method(*_args: Any, **_kwargs: Any) -> np.ndarray:
    raise AssertionError(
        "invalid association soft-overlap controls should be rejected before "
        "fallback cost computation"
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"soft_iou_radius": True}, "soft_iou_radius must be an integer"),
        ({"soft_iou_radius": np.bool_(True)}, "soft_iou_radius must be an integer"),
        ({"soft_iou_radius": b"1"}, "soft_iou_radius must be an integer"),
        (
            {"soft_iou_radius": bytearray(b"1")},
            "soft_iou_radius must be an integer",
        ),
        ({"soft_iou_radius": 1.5}, "soft_iou_radius must be an integer"),
        ({"soft_iou_radius": "1.5"}, "soft_iou_radius must be an integer"),
        ({"soft_iou_radius": np.inf}, "soft_iou_radius must be an integer"),
        ({"soft_iou_radius": np.array([1])}, "soft_iou_radius must be an integer"),
        ({"soft_iou_radius": -1}, "soft_iou_radius must be non-negative"),
        (
            {"use_soft_iou_for_iou_cost": 1},
            "use_soft_iou_for_iou_cost must be a boolean",
        ),
        (
            {"use_soft_iou_for_iou_cost": "false"},
            "use_soft_iou_for_iou_cost must be a boolean",
        ),
        (
            {"similarity_epsilon": np.nan},
            "similarity_epsilon must be a finite positive value",
        ),
        (
            {"similarity_epsilon": np.array([1.0e-6])},
            "similarity_epsilon must be a finite positive value",
        ),
        ({"large_cost": np.inf}, "large_cost must be a finite positive value"),
        (
            {"large_cost": np.array([1.0e6])},
            "large_cost must be a finite positive value",
        ),
        ({"iou_weight": True}, "iou_weight must be a finite non-negative value"),
        ({"iou_weight": -1.0}, "iou_weight must be a finite non-negative value"),
        (
            {"iou_weight": np.array([1.0])},
            "iou_weight must be a finite non-negative value",
        ),
    ],
)
def test_association_soft_overlap_rejects_invalid_scalar_controls(
    kwargs: dict[str, object], message: str
) -> None:
    reference = _single_roi_plane(np.array([1.0, 0.0, 0.0]))
    measurement = _single_roi_plane(np.array([1.0, 0.0, 0.0]))
    common_kwargs: dict[str, object] = {
        "soft_iou_radius": 1,
        "use_soft_iou_for_iou_cost": True,
    }
    common_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        soft_iou_pairwise_cost_matrix(
            _unexpected_original_method,
            reference,
            measurement,
            **common_kwargs,
        )


def test_association_soft_overlap_keeps_integer_like_radius_values() -> None:
    mask = np.zeros((1, 3, 3), dtype=bool)
    mask[0, 1, 1] = True

    from_numpy_integer = dilate_mask_stack(mask, radius=np.int64(1))
    from_zero_dim_array = dilate_mask_stack(mask, radius=np.array(1))
    from_string_integer = dilate_mask_stack(mask, radius="1")

    assert np.array_equal(from_numpy_integer, from_zero_dim_array)
    assert np.array_equal(from_numpy_integer, from_string_integer)
    assert int(np.count_nonzero(from_numpy_integer)) == 5
