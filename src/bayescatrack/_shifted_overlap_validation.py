"""Strict scalar validation for shifted-overlap pairwise-cost knobs.

The shifted-overlap wrapper is often exercised through experiment manifests and
YAML-derived dictionaries.  Avoid Python truthiness/NaN corner cases there so a
string such as ``"false"`` cannot enable a replacement cost term and non-finite
weights cannot silently become hard-gated costs or non-finite diagnostic
components.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

_MARKER = "_bayescatrack_shifted_overlap_scalar_validation"
_BOOLEAN_KWARGS = (
    "use_shifted_iou_for_iou_cost",
    "use_shifted_mask_cosine_for_mask_cosine_cost",
)
_NONNEGATIVE_FLOAT_KWARGS = (
    "shifted_iou_weight",
    "shifted_mask_cosine_weight",
    "shifted_iou_shift_penalty_weight",
    "iou_weight",
    "mask_cosine_weight",
)
_POSITIVE_FLOAT_KWARGS = (
    "similarity_epsilon",
    "large_cost",
)


def install_shifted_overlap_scalar_validation() -> None:
    """Install idempotent validation around shifted-overlap scalar kwargs."""

    from .association import shifted_overlap  # pylint: disable=import-outside-toplevel

    current = shifted_overlap.shifted_iou_pairwise_cost_matrix
    if getattr(current, _MARKER, False):
        return

    def _validated_shifted_iou_pairwise_cost_matrix(
        original_method: Callable[..., Any],
        self: Any,
        other: Any,
        **kwargs: Any,
    ) -> Any:
        validated_kwargs = _validate_shifted_overlap_scalar_kwargs(kwargs)
        return current(original_method, self, other, **validated_kwargs)

    setattr(_validated_shifted_iou_pairwise_cost_matrix, _MARKER, True)
    setattr(_validated_shifted_iou_pairwise_cost_matrix, "_bayescatrack_original", current)
    shifted_overlap.shifted_iou_pairwise_cost_matrix = _validated_shifted_iou_pairwise_cost_matrix


def _validate_shifted_overlap_scalar_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    validated = dict(kwargs)
    for name in _BOOLEAN_KWARGS:
        if name in validated:
            validated[name] = _strict_bool_or_none(validated[name], name=name)
    for name in _NONNEGATIVE_FLOAT_KWARGS:
        if name in validated:
            validated[name] = _finite_nonnegative_float(validated[name], name=name)
    for name in _POSITIVE_FLOAT_KWARGS:
        if name in validated:
            validated[name] = _finite_positive_float(validated[name], name=name)
    if (
        "shifted_iou_shift_penalty_scale" in validated
        and validated["shifted_iou_shift_penalty_scale"] is not None
    ):
        validated["shifted_iou_shift_penalty_scale"] = _finite_positive_float(
            validated["shifted_iou_shift_penalty_scale"],
            name="shifted_iou_shift_penalty_scale",
        )
    return validated


def _strict_bool_or_none(value: Any, *, name: str) -> bool:
    if value is None:
        return False
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def _finite_nonnegative_float(value: Any, *, name: str) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite non-negative value")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative value") from exc
    if not np.isfinite(numeric_value) or numeric_value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative value")
    return numeric_value


def _finite_positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite positive value")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive value") from exc
    if not np.isfinite(numeric_value) or numeric_value <= 0.0:
        raise ValueError(f"{name} must be a finite positive value")
    return numeric_value


__all__ = ["install_shifted_overlap_scalar_validation"]
