from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from bayescatrack.association.pyrecest_global_assignment import (
    registered_iou_cost_kwargs,
    registered_shifted_iou_cost_kwargs,
    roi_aware_local_cost_kwargs,
    roi_aware_shifted_cost_kwargs,
)

PresetFactory = Callable[..., dict[str, Any]]


class _NonFiniteFloatAdapter:
    def __float__(self) -> float:
        raise OverflowError("cannot convert")


class _ArithmeticFloatAdapter:
    def __float__(self) -> float:
        raise ArithmeticError("cannot convert")


@pytest.mark.parametrize(
    ("factory", "kwargs", "message"),
    [
        (
            registered_iou_cost_kwargs,
            {"similarity_epsilon": _NonFiniteFloatAdapter()},
            "similarity_epsilon must be a finite positive value",
        ),
        (
            registered_shifted_iou_cost_kwargs,
            {"shifted_iou_shift_penalty_weight": _ArithmeticFloatAdapter()},
            "shifted_iou_shift_penalty_weight must be a finite non-negative value",
        ),
        (
            registered_shifted_iou_cost_kwargs,
            {"shifted_iou_radius": _NonFiniteFloatAdapter()},
            "shifted_iou_radius must be a non-negative integer",
        ),
        (
            roi_aware_local_cost_kwargs,
            {"patch_radius": _ArithmeticFloatAdapter()},
            "patch_radius must be a non-negative integer",
        ),
        (
            roi_aware_local_cost_kwargs,
            {"neighbor_k": _NonFiniteFloatAdapter()},
            "neighbor_k must be a positive integer",
        ),
        (
            roi_aware_shifted_cost_kwargs,
            {"shifted_iou_shift_penalty_scale": _NonFiniteFloatAdapter()},
            "shifted_iou_shift_penalty_scale must be a finite positive value",
        ),
    ],
)
def test_global_cost_presets_normalize_adapter_conversion_failures(
    factory: PresetFactory,
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        factory(**kwargs)
