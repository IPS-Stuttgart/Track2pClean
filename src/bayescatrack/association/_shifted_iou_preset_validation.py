"""Strict validation for shifted-IoU global-assignment preset controls."""

from __future__ import annotations

import operator
from functools import wraps
from typing import Any

import numpy as np

from . import pyrecest_global_assignment as _global_assignment

_PATCH_ATTR = "_bayescatrack_shifted_iou_preset_validation_patch"


def install_shifted_iou_preset_validation() -> None:
    """Install an idempotent validator around shifted-IoU preset kwargs."""

    original = _global_assignment.registered_shifted_iou_cost_kwargs
    if getattr(original, _PATCH_ATTR, False):
        return

    @wraps(original)
    def registered_shifted_iou_cost_kwargs(
        *,
        similarity_epsilon: float = 1.0e-6,
        shifted_iou_radius: int = 2,
        shifted_iou_shift_penalty_weight: float = 0.0,
        shifted_iou_shift_penalty_scale: float | None = None,
    ) -> dict[str, float | int | bool]:
        normalized_scale = (
            None
            if shifted_iou_shift_penalty_scale is None
            else _finite_positive_float(
                shifted_iou_shift_penalty_scale,
                name="shifted_iou_shift_penalty_scale",
            )
        )
        return original(
            similarity_epsilon=_finite_positive_float(
                similarity_epsilon,
                name="similarity_epsilon",
            ),
            shifted_iou_radius=_integer_like(
                shifted_iou_radius,
                name="shifted_iou_radius",
                minimum=0,
            ),
            shifted_iou_shift_penalty_weight=_finite_nonnegative_float(
                shifted_iou_shift_penalty_weight,
                name="shifted_iou_shift_penalty_weight",
            ),
            shifted_iou_shift_penalty_scale=normalized_scale,
        )

    setattr(registered_shifted_iou_cost_kwargs, _PATCH_ATTR, True)
    setattr(registered_shifted_iou_cost_kwargs, "_bayescatrack_original", original)
    _global_assignment.registered_shifted_iou_cost_kwargs = (
        registered_shifted_iou_cost_kwargs
    )


def _finite_nonnegative_float(value: Any, *, name: str) -> float:
    numeric_value = _finite_float(value, name=name)
    if numeric_value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return numeric_value


def _finite_positive_float(value: Any, *, name: str) -> float:
    numeric_value = _finite_float(value, name=name)
    if numeric_value <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    return numeric_value


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_, np.ndarray)):
        raise ValueError(f"{name} must be a finite numeric value")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric value") from exc
    if not np.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite")
    return numeric_value


def _integer_like(value: Any, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, (int, np.integer)):
        integer_value = int(value)
    elif isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(f"{name} must be an integer")
        integer_value = int(numeric_value)
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
            integer_value = int(operator.index(value))
        except TypeError as exc:
            raise ValueError(f"{name} must be an integer") from exc

    if integer_value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(integer_value)


__all__ = ["install_shifted_iou_preset_validation"]
