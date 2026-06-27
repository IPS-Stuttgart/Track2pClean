from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from bayescatrack.association.pyrecest_global_assignment import (
    registered_iou_cost_kwargs,
    registered_shifted_iou_cost_kwargs,
    roi_aware_local_cost_kwargs,
    roi_aware_shifted_cost_kwargs,
)

PresetFactory = Callable[..., dict[str, Any]]


@pytest.mark.parametrize(
    ("factory", "kwargs", "message"),
    [
        (
            registered_iou_cost_kwargs,
            {"similarity_epsilon": np.nan},
            "similarity_epsilon must be a finite positive value",
        ),
        (
            registered_shifted_iou_cost_kwargs,
            {"shifted_iou_radius": True},
            "shifted_iou_radius must be a non-negative integer",
        ),
        (
            registered_shifted_iou_cost_kwargs,
            {"shifted_iou_radius": 1.5},
            "shifted_iou_radius must be a non-negative integer",
        ),
        (
            registered_shifted_iou_cost_kwargs,
            {"shifted_iou_shift_penalty_weight": np.inf},
            "shifted_iou_shift_penalty_weight must be a finite non-negative value",
        ),
        (
            registered_shifted_iou_cost_kwargs,
            {"shifted_iou_shift_penalty_scale": 0.0},
            "shifted_iou_shift_penalty_scale must be a finite positive value",
        ),
        (
            roi_aware_local_cost_kwargs,
            {"weighted_dice_weight": np.nan},
            "weighted_dice_weight must be a finite non-negative value",
        ),
        (
            roi_aware_local_cost_kwargs,
            {"patch_radius": False},
            "patch_radius must be a non-negative integer",
        ),
        (
            roi_aware_local_cost_kwargs,
            {"neighbor_k": 0},
            "neighbor_k must be a positive integer",
        ),
        (
            roi_aware_shifted_cost_kwargs,
            {"shifted_iou_shift_penalty_weight": True},
            "shifted_iou_shift_penalty_weight must be a finite non-negative value",
        ),
    ],
)
def test_global_cost_presets_reject_ambiguous_or_nonfinite_controls(
    factory: PresetFactory,
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        factory(**kwargs)


def test_global_cost_presets_normalize_valid_numeric_controls() -> None:
    shifted = registered_shifted_iou_cost_kwargs(
        similarity_epsilon="1e-5",
        shifted_iou_radius="3",
        shifted_iou_shift_penalty_weight="0.25",
        shifted_iou_shift_penalty_scale="2.0",
    )

    assert shifted["similarity_epsilon"] == pytest.approx(1.0e-5)
    assert shifted["shifted_iou_radius"] == 3
    assert shifted["shifted_iou_shift_penalty_weight"] == pytest.approx(0.25)
    assert shifted["shifted_iou_shift_penalty_scale"] == pytest.approx(2.0)

    local = roi_aware_local_cost_kwargs(
        patch_radius=np.int64(4),
        neighbor_k=np.float64(5.0),
        weighted_dice_weight=np.float64(1.25),
    )

    assert local["patch_radius"] == 4
    assert local["neighbor_k"] == 5
    assert local["weighted_dice_weight"] == pytest.approx(1.25)
