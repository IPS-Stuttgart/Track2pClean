from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association import pyrecest_global_assignment as global_assignment


def test_registered_shifted_iou_cost_kwargs_preserve_numeric_inputs():
    kwargs = global_assignment.registered_shifted_iou_cost_kwargs(
        similarity_epsilon="1e-5",
        shifted_iou_radius="3.0",
        shifted_iou_shift_penalty_weight="0.25",
        shifted_iou_shift_penalty_scale=np.float64(2.5),
    )

    assert kwargs["similarity_epsilon"] == pytest.approx(1e-5)
    assert kwargs["shifted_iou_radius"] == 3
    assert kwargs["use_shifted_iou_for_iou_cost"] is True
    assert kwargs["shifted_iou_shift_penalty_weight"] == pytest.approx(0.25)
    assert kwargs["shifted_iou_shift_penalty_scale"] == pytest.approx(2.5)


@pytest.mark.parametrize(
    "bad_radius",
    [
        True,
        np.bool_(False),
        np.asarray(3),
        np.asarray([3]),
        1.5,
        "1.5",
        np.inf,
        "",
        object(),
    ],
)
def test_registered_shifted_iou_cost_kwargs_reject_invalid_radius(bad_radius):
    with pytest.raises(ValueError, match="shifted_iou_radius"):
        global_assignment.registered_shifted_iou_cost_kwargs(
            shifted_iou_radius=bad_radius
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"similarity_epsilon": True}, "similarity_epsilon"),
        ({"similarity_epsilon": 0.0}, "similarity_epsilon"),
        ({"similarity_epsilon": np.nan}, "similarity_epsilon"),
        ({"similarity_epsilon": np.asarray(1.0)}, "similarity_epsilon"),
        ({"similarity_epsilon": np.asarray([1.0])}, "similarity_epsilon"),
        (
            {"shifted_iou_shift_penalty_weight": True},
            "shifted_iou_shift_penalty_weight",
        ),
        (
            {"shifted_iou_shift_penalty_weight": np.nan},
            "shifted_iou_shift_penalty_weight",
        ),
        (
            {"shifted_iou_shift_penalty_weight": np.inf},
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
        ({"shifted_iou_shift_penalty_scale": True}, "shifted_iou_shift_penalty_scale"),
        ({"shifted_iou_shift_penalty_scale": 0.0}, "shifted_iou_shift_penalty_scale"),
        (
            {"shifted_iou_shift_penalty_scale": np.nan},
            "shifted_iou_shift_penalty_scale",
        ),
        (
            {"shifted_iou_shift_penalty_scale": np.inf},
            "shifted_iou_shift_penalty_scale",
        ),
        (
            {"shifted_iou_shift_penalty_scale": np.asarray(2.5)},
            "shifted_iou_shift_penalty_scale",
        ),
        (
            {"shifted_iou_shift_penalty_scale": np.asarray([2.5])},
            "shifted_iou_shift_penalty_scale",
        ),
    ],
)
def test_registered_shifted_iou_cost_kwargs_reject_invalid_float_controls(
    kwargs,
    match,
):
    with pytest.raises(ValueError, match=match):
        global_assignment.registered_shifted_iou_cost_kwargs(**kwargs)
