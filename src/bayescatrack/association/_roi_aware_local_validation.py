"""Strict validation for ROI-aware local-evidence preset controls."""

from __future__ import annotations

import operator
from typing import Any

import numpy as np

from . import pyrecest_global_assignment as _global_assignment

_PATCH_ATTR = "_bayescatrack_roi_aware_local_validation_patch"


def install_roi_aware_local_validation() -> None:
    """Install an idempotent validator around the ROI-aware local preset."""

    original = _global_assignment.roi_aware_local_cost_kwargs
    if getattr(original, _PATCH_ATTR, False):
        return

    def roi_aware_local_cost_kwargs(
        *,
        weighted_dice_weight: float = 1.0,
        overlap_fraction_weight: float = 0.5,
        distance_transform_weight: float = 0.5,
        image_patch_weight: float = 0.15,
        neighbor_constellation_weight: float = 0.25,
        centroid_rank_weight: float = 0.25,
        patch_radius: int = 8,
        neighbor_k: int = 8,
    ) -> dict[str, float | int | bool]:
        weights = {
            weight_name: _finite_nonnegative_float(weight_value, name=weight_name)
            for weight_name, weight_value in {
                "weighted_dice_weight": weighted_dice_weight,
                "overlap_fraction_weight": overlap_fraction_weight,
                "distance_transform_weight": distance_transform_weight,
                "image_patch_weight": image_patch_weight,
                "neighbor_constellation_weight": neighbor_constellation_weight,
                "centroid_rank_weight": centroid_rank_weight,
            }.items()
        }
        return original(
            weighted_dice_weight=weights["weighted_dice_weight"],
            overlap_fraction_weight=weights["overlap_fraction_weight"],
            distance_transform_weight=weights["distance_transform_weight"],
            image_patch_weight=weights["image_patch_weight"],
            neighbor_constellation_weight=weights["neighbor_constellation_weight"],
            centroid_rank_weight=weights["centroid_rank_weight"],
            patch_radius=_integer_like(patch_radius, name="patch_radius", minimum=0),
            neighbor_k=_integer_like(neighbor_k, name="neighbor_k", minimum=1),
        )

    setattr(roi_aware_local_cost_kwargs, _PATCH_ATTR, True)
    setattr(roi_aware_local_cost_kwargs, "_bayescatrack_original", original)
    _global_assignment.roi_aware_local_cost_kwargs = roi_aware_local_cost_kwargs


def _finite_nonnegative_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite non-negative value")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative value") from exc
    if not np.isfinite(numeric_value) or numeric_value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative value")
    return numeric_value


def _integer_like(value: Any, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, (int, np.integer)):
        integer_value = int(value)
    elif isinstance(value, (float, np.floating)):
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

    if integer_value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(integer_value)


__all__ = ["install_roi_aware_local_validation"]
