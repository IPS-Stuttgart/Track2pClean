from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association import pyrecest_global_assignment as global_assignment


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"shifted_iou_radius": True}, "shifted_iou_radius"),
        ({"shifted_iou_radius": 1.5}, "shifted_iou_radius"),
        ({"shifted_iou_radius": np.asarray(2)}, "shifted_iou_radius"),
        ({"shifted_iou_radius": np.asarray([2])}, "shifted_iou_radius"),
        ({"similarity_epsilon": True}, "similarity_epsilon"),
        ({"similarity_epsilon": np.asarray(1.0e-6)}, "similarity_epsilon"),
        ({"similarity_epsilon": np.asarray([1.0e-6])}, "similarity_epsilon"),
        (
            {"shifted_iou_shift_penalty_weight": True},
            "shifted_iou_shift_penalty_weight",
        ),
        (
            {"shifted_iou_shift_penalty_weight": np.asarray(0.25)},
            "shifted_iou_shift_penalty_weight",
        ),
        (
            {"shifted_iou_shift_penalty_weight": np.asarray([0.25])},
            "shifted_iou_shift_penalty_weight",
        ),
        (
            {"shifted_iou_shift_penalty_scale": True},
            "shifted_iou_shift_penalty_scale",
        ),
        (
            {"shifted_iou_shift_penalty_scale": np.asarray(2.0)},
            "shifted_iou_shift_penalty_scale",
        ),
        (
            {"shifted_iou_shift_penalty_scale": np.asarray([2.0])},
            "shifted_iou_shift_penalty_scale",
        ),
    ],
)
def test_registered_shifted_iou_cost_kwargs_reject_invalid_controls(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        global_assignment.registered_shifted_iou_cost_kwargs(**kwargs)


def test_roi_aware_shifted_cost_kwargs_rejects_fractional_radius() -> None:
    with pytest.raises(ValueError, match="shifted_iou_radius"):
        global_assignment.roi_aware_shifted_cost_kwargs(shifted_iou_radius=1.5)


def test_roi_aware_shifted_cost_kwargs_rejects_array_radius() -> None:
    with pytest.raises(ValueError, match="shifted_iou_radius"):
        global_assignment.roi_aware_shifted_cost_kwargs(
            shifted_iou_radius=np.asarray(2)
        )
