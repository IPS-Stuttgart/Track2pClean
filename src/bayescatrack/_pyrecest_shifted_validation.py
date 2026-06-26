"""Strict validation for PyRecEst global-assignment shifted-IoU presets."""

from __future__ import annotations

import operator
from typing import Any

import numpy as np
from bayescatrack.association import pyrecest_global_assignment as _global_assignment


def install_pyrecest_shifted_validation() -> None:
    """Reject malformed numeric controls before constructing shifted-IoU presets."""

    original_iou_cost_kwargs = _global_assignment.registered_iou_cost_kwargs
    original_shifted_cost_kwargs = _global_assignment.registered_shifted_iou_cost_kwargs
    if getattr(
        original_shifted_cost_kwargs,
        "_bayescatrack_shifted_preset_validation",
        False,
    ):
        return

    def registered_iou_cost_kwargs(
        *,
        similarity_epsilon: float = 1.0e-6,
    ) -> dict[str, Any]:
        return original_iou_cost_kwargs(
            similarity_epsilon=_finite_positive_float(
                similarity_epsilon,
                name="similarity_epsilon",
            )
        )

    def registered_shifted_iou_cost_kwargs(
        *,
        similarity_epsilon: float = 1.0e-6,
        shifted_iou_radius: int = 2,
        shifted_iou_shift_penalty_weight: float = 0.0,
        shifted_iou_shift_penalty_scale: float | None = None,
    ) -> dict[str, float | int | bool]:
        radius = _nonnegative_int(shifted_iou_radius, name="shifted_iou_radius")
        shift_penalty_weight = _finite_nonnegative_float(
            shifted_iou_shift_penalty_weight,
            name="shifted_iou_shift_penalty_weight",
        )
        shift_penalty_scale = None
        if shifted_iou_shift_penalty_scale is not None:
            shift_penalty_scale = _finite_positive_float(
                shifted_iou_shift_penalty_scale,
                name="shifted_iou_shift_penalty_scale",
            )
        return original_shifted_cost_kwargs(
            similarity_epsilon=_finite_positive_float(
                similarity_epsilon,
                name="similarity_epsilon",
            ),
            shifted_iou_radius=radius,
            shifted_iou_shift_penalty_weight=shift_penalty_weight,
            shifted_iou_shift_penalty_scale=shift_penalty_scale,
        )

    setattr(
        registered_iou_cost_kwargs,
        "_bayescatrack_shifted_preset_validation",
        True,
    )
    setattr(
        registered_iou_cost_kwargs,
        "_bayescatrack_original",
        original_iou_cost_kwargs,
    )
    setattr(
        registered_shifted_iou_cost_kwargs,
        "_bayescatrack_shifted_preset_validation",
        True,
    )
    setattr(
        registered_shifted_iou_cost_kwargs,
        "_bayescatrack_original",
        original_shifted_cost_kwargs,
    )

    _global_assignment.registered_iou_cost_kwargs = registered_iou_cost_kwargs
    _global_assignment.registered_shifted_iou_cost_kwargs = (
        registered_shifted_iou_cost_kwargs
    )


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or not float(value).is_integer():
            raise ValueError(f"{name} must be an integer")
        integer_value = int(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{name} must be an integer")
        try:
            numeric_value = float(stripped)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(f"{name} must be an integer")
        integer_value = int(numeric_value)
    else:
        try:
            integer_value = operator.index(value)
        except TypeError as exc:
            raise ValueError(f"{name} must be an integer") from exc
    if integer_value < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(integer_value)


def _finite_nonnegative_float(value: Any, *, name: str) -> float:
    return _finite_float(value, name=name, lower_bound=0.0, positive=False)


def _finite_positive_float(value: Any, *, name: str) -> float:
    return _finite_float(value, name=name, lower_bound=0.0, positive=True)


def _finite_float(
    value: Any,
    *,
    name: str,
    lower_bound: float,
    positive: bool,
) -> float:
    qualifier = "positive" if positive else "non-negative"
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite {qualifier} value")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite {qualifier} value") from exc
    violates_bound = (
        numeric_value <= lower_bound if positive else numeric_value < lower_bound
    )
    if not np.isfinite(numeric_value) or violates_bound:
        raise ValueError(f"{name} must be a finite {qualifier} value")
    return numeric_value


__all__ = ["install_pyrecest_shifted_validation"]
