from __future__ import annotations

import numpy as np
import pytest
from bayescatrack import CalciumPlaneData
from bayescatrack.association.shifted_overlap import shifted_iou_pairwise_cost_matrix


class _OverflowingFloat:
    def __float__(self) -> float:
        raise OverflowError("too large")


def _base_cost_should_not_run(*args: object, **kwargs: object) -> np.ndarray:
    raise AssertionError("validation must run before base cost computation")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"similarity_epsilon": _OverflowingFloat()},
            "similarity_epsilon must be a finite positive value",
        ),
        (
            {"large_cost": _OverflowingFloat()},
            "large_cost must be a finite positive value",
        ),
        (
            {"iou_weight": _OverflowingFloat()},
            "iou_weight must be a finite non-negative value",
        ),
        (
            {"mask_cosine_weight": _OverflowingFloat()},
            "mask_cosine_weight must be a finite non-negative value",
        ),
        (
            {"shifted_iou_shift_penalty_weight": _OverflowingFloat()},
            "shifted_iou_shift_penalty_weight must be a finite non-negative value",
        ),
        (
            {"shifted_iou_shift_penalty_scale": _OverflowingFloat()},
            "shifted_iou_shift_penalty_scale must be a finite positive value",
        ),
    ],
)
def test_shifted_overlap_normalizes_overflowing_float_controls(
    kwargs: dict[str, object], message: str
) -> None:
    reference_plane = CalciumPlaneData(np.array([[[1.0, 0.0, 0.0]]]))
    measurement_plane = CalciumPlaneData(np.array([[[1.0, 0.0, 0.0]]]))
    common_kwargs: dict[str, object] = {
        "shifted_iou_radius": 1,
        "use_shifted_iou_for_iou_cost": True,
    }
    common_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        shifted_iou_pairwise_cost_matrix(
            _base_cost_should_not_run,
            reference_plane,
            measurement_plane,
            **common_kwargs,
        )
