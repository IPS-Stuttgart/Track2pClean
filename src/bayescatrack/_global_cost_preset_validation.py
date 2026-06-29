"""Strict validation for global-assignment cost-preset helpers.

Benchmark manifests call the public preset helpers before pairwise-cost wrappers
run.  Validate scalar controls at this boundary so booleans, NaNs, infinities,
and fractional integer knobs cannot be converted into plausible-looking
benchmark settings by bare ``int(...)`` or ``float(...)`` coercion.
"""

from __future__ import annotations

import operator
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_global_cost_preset_validation_patch"


def install_global_cost_preset_validation() -> None:
    """Install idempotent validation around global-assignment preset helpers."""

    from .association import (
        pyrecest_global_assignment as global_assignment,  # pylint: disable=import-outside-toplevel
    )

    original_registered_iou = global_assignment.registered_iou_cost_kwargs
    if not getattr(original_registered_iou, _PATCH_MARKER, False):

        @wraps(original_registered_iou)
        def registered_iou_cost_kwargs(
            *, similarity_epsilon: Any = 1.0e-6
        ) -> dict[str, Any]:
            return original_registered_iou(
                similarity_epsilon=_finite_positive_float(
                    similarity_epsilon,
                    name="similarity_epsilon",
                )
            )

        _mark_wrapper(registered_iou_cost_kwargs, original_registered_iou)
        global_assignment.registered_iou_cost_kwargs = registered_iou_cost_kwargs

    original_registered_shifted = global_assignment.registered_shifted_iou_cost_kwargs
    if not getattr(original_registered_shifted, _PATCH_MARKER, False):

        @wraps(original_registered_shifted)
        def registered_shifted_iou_cost_kwargs(
            *,
            similarity_epsilon: Any = 1.0e-6,
            shifted_iou_radius: Any = 2,
            shifted_iou_shift_penalty_weight: Any = 0.0,
            shifted_iou_shift_penalty_scale: Any | None = None,
        ) -> dict[str, Any]:
            return original_registered_shifted(
                similarity_epsilon=_finite_positive_float(
                    similarity_epsilon,
                    name="similarity_epsilon",
                ),
                shifted_iou_radius=_nonnegative_int(
                    shifted_iou_radius,
                    name="shifted_iou_radius",
                ),
                shifted_iou_shift_penalty_weight=_finite_nonnegative_float(
                    shifted_iou_shift_penalty_weight,
                    name="shifted_iou_shift_penalty_weight",
                ),
                shifted_iou_shift_penalty_scale=(
                    None
                    if shifted_iou_shift_penalty_scale is None
                    else _finite_positive_float(
                        shifted_iou_shift_penalty_scale,
                        name="shifted_iou_shift_penalty_scale",
                    )
                ),
            )

        _mark_wrapper(registered_shifted_iou_cost_kwargs, original_registered_shifted)
        global_assignment.registered_shifted_iou_cost_kwargs = (
            registered_shifted_iou_cost_kwargs
        )

    original_roi_aware_local = global_assignment.roi_aware_local_cost_kwargs
    if not getattr(original_roi_aware_local, _PATCH_MARKER, False):

        @wraps(original_roi_aware_local)
        def roi_aware_local_cost_kwargs(
            *,
            weighted_dice_weight: Any = 1.0,
            overlap_fraction_weight: Any = 0.5,
            distance_transform_weight: Any = 0.5,
            image_patch_weight: Any = 0.15,
            neighbor_constellation_weight: Any = 0.25,
            centroid_rank_weight: Any = 0.25,
            patch_radius: Any = 8,
            neighbor_k: Any = 8,
        ) -> dict[str, Any]:
            return original_roi_aware_local(
                weighted_dice_weight=_finite_nonnegative_float(
                    weighted_dice_weight,
                    name="weighted_dice_weight",
                ),
                overlap_fraction_weight=_finite_nonnegative_float(
                    overlap_fraction_weight,
                    name="overlap_fraction_weight",
                ),
                distance_transform_weight=_finite_nonnegative_float(
                    distance_transform_weight,
                    name="distance_transform_weight",
                ),
                image_patch_weight=_finite_nonnegative_float(
                    image_patch_weight,
                    name="image_patch_weight",
                ),
                neighbor_constellation_weight=_finite_nonnegative_float(
                    neighbor_constellation_weight,
                    name="neighbor_constellation_weight",
                ),
                centroid_rank_weight=_finite_nonnegative_float(
                    centroid_rank_weight,
                    name="centroid_rank_weight",
                ),
                patch_radius=_nonnegative_int(patch_radius, name="patch_radius"),
                neighbor_k=_positive_int(neighbor_k, name="neighbor_k"),
            )

        _mark_wrapper(roi_aware_local_cost_kwargs, original_roi_aware_local)
        global_assignment.roi_aware_local_cost_kwargs = roi_aware_local_cost_kwargs

    original_roi_aware_shifted = global_assignment.roi_aware_shifted_cost_kwargs
    if not getattr(original_roi_aware_shifted, _PATCH_MARKER, False):

        @wraps(original_roi_aware_shifted)
        def roi_aware_shifted_cost_kwargs(
            *,
            shifted_iou_radius: Any = 2,
            shifted_iou_shift_penalty_weight: Any = 0.25,
            shifted_iou_shift_penalty_scale: Any | None = None,
        ) -> dict[str, Any]:
            return original_roi_aware_shifted(
                shifted_iou_radius=_nonnegative_int(
                    shifted_iou_radius,
                    name="shifted_iou_radius",
                ),
                shifted_iou_shift_penalty_weight=_finite_nonnegative_float(
                    shifted_iou_shift_penalty_weight,
                    name="shifted_iou_shift_penalty_weight",
                ),
                shifted_iou_shift_penalty_scale=(
                    None
                    if shifted_iou_shift_penalty_scale is None
                    else _finite_positive_float(
                        shifted_iou_shift_penalty_scale,
                        name="shifted_iou_shift_penalty_scale",
                    )
                ),
            )

        _mark_wrapper(roi_aware_shifted_cost_kwargs, original_roi_aware_shifted)
        global_assignment.roi_aware_shifted_cost_kwargs = roi_aware_shifted_cost_kwargs


def _mark_wrapper(wrapper: Any, original: Any) -> None:
    setattr(wrapper, _PATCH_MARKER, True)
    setattr(wrapper, "_bayescatrack_original", original)


def _nonnegative_int(value: Any, *, name: str) -> int:
    normalized = _integer_value(value, name=name, qualifier="a non-negative integer")
    if normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return normalized


def _positive_int(value: Any, *, name: str) -> int:
    normalized = _integer_value(value, name=name, qualifier="a positive integer")
    if normalized <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return normalized


def _integer_value(value: Any, *, name: str, qualifier: str) -> int:
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ValueError(f"{name} must be {qualifier}")
        value = value.item()
    if isinstance(value, (bool, np.bool_, bytes, bytearray)):
        raise ValueError(f"{name} must be {qualifier}")
    try:
        return int(operator.index(value))
    except (TypeError, ValueError, OverflowError):
        pass
    candidate: Any = value
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            raise ValueError(f"{name} must be {qualifier}")
    try:
        numeric_value = float(candidate)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be {qualifier}") from exc
    if not np.isfinite(numeric_value) or not numeric_value.is_integer():
        raise ValueError(f"{name} must be {qualifier}")
    return int(numeric_value)


def _finite_nonnegative_float(value: Any, *, name: str) -> float:
    return _finite_float(value, name=name, lower_bound=0.0, positive=False)


def _finite_positive_float(value: Any, *, name: str) -> float:
    return _finite_float(value, name=name, lower_bound=0.0, positive=True)


def _finite_float(
    value: Any, *, name: str, lower_bound: float, positive: bool
) -> float:
    qualifier = "positive" if positive else "non-negative"
    if isinstance(value, (bool, np.bool_, bytes, bytearray, np.ndarray)):
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


__all__ = ["install_global_cost_preset_validation"]
