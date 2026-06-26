"""Strict validation for automatic registration selector controls.

The automatic registration selector turns several caller-provided thresholds and
penalties into candidate scores.  Plain Python comparisons reject negative values
but let ``NaN`` through for some controls, and ``float()`` turns booleans into
numeric values.  That can produce accepted candidates with non-finite selection
scores.  These hooks fail fast before malformed controls reach the scorer.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_registration_selection_validation_patch"


def install_registration_selection_validation() -> None:
    """Install idempotent validation around auto-registration selection."""

    from . import (
        registration_selection as _registration_selection,  # pylint: disable=import-outside-toplevel
    )

    original = _registration_selection.select_registration_transform
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def select_registration_transform_with_validation(
        reference_plane: Any,
        moving_plane: Any,
        *,
        candidate_transforms: Any = _registration_selection.DEFAULT_AUTO_REGISTRATION_CANDIDATES,
        fov_affine_mask_warp_mode: str = "nearest",
        min_fov_correlation_gain: Any = 0.02,
        max_empty_roi_fraction: Any = 0.1,
        min_retained_mask_area_fraction: Any = 0.5,
        min_nonrigid_inverse_warp_valid_fraction: Any = 0.90,
        empty_roi_penalty: Any = 0.75,
        retained_area_penalty: Any = 0.5,
        nonrigid_valid_fraction_penalty: Any = 0.75,
        complexity_penalty: Any = None,
    ) -> Any:
        min_fov_correlation_gain = _finite_nonnegative_scalar(
            min_fov_correlation_gain,
            "min_fov_correlation_gain",
        )
        max_empty_roi_fraction = _finite_unit_interval_scalar(
            max_empty_roi_fraction,
            "max_empty_roi_fraction",
        )
        min_retained_mask_area_fraction = _finite_nonnegative_scalar(
            min_retained_mask_area_fraction,
            "min_retained_mask_area_fraction",
        )
        min_nonrigid_inverse_warp_valid_fraction = _finite_unit_interval_scalar(
            min_nonrigid_inverse_warp_valid_fraction,
            "min_nonrigid_inverse_warp_valid_fraction",
        )
        empty_roi_penalty = _finite_nonnegative_scalar(
            empty_roi_penalty,
            "empty_roi_penalty",
        )
        retained_area_penalty = _finite_nonnegative_scalar(
            retained_area_penalty,
            "retained_area_penalty",
        )
        nonrigid_valid_fraction_penalty = _finite_nonnegative_scalar(
            nonrigid_valid_fraction_penalty,
            "nonrigid_valid_fraction_penalty",
        )
        complexity_penalty = _validated_complexity_penalty(complexity_penalty)

        return original(
            reference_plane,
            moving_plane,
            candidate_transforms=candidate_transforms,
            fov_affine_mask_warp_mode=fov_affine_mask_warp_mode,
            min_fov_correlation_gain=min_fov_correlation_gain,
            max_empty_roi_fraction=max_empty_roi_fraction,
            min_retained_mask_area_fraction=min_retained_mask_area_fraction,
            min_nonrigid_inverse_warp_valid_fraction=min_nonrigid_inverse_warp_valid_fraction,
            empty_roi_penalty=empty_roi_penalty,
            retained_area_penalty=retained_area_penalty,
            nonrigid_valid_fraction_penalty=nonrigid_valid_fraction_penalty,
            complexity_penalty=complexity_penalty,
        )

    setattr(select_registration_transform_with_validation, _PATCH_MARKER, True)
    setattr(
        select_registration_transform_with_validation,
        "_bayescatrack_original",
        original,
    )
    _registration_selection.select_registration_transform = (
        select_registration_transform_with_validation
    )


def _validated_complexity_penalty(value: Any) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(
            "complexity_penalty must be a mapping of transform names to finite non-negative values"
        )
    return {
        str(transform_type): _finite_nonnegative_scalar(
            penalty,
            f"complexity_penalty[{transform_type!r}]",
        )
        for transform_type, penalty in value.items()
    }


def _finite_unit_interval_scalar(value: Any, field_name: str) -> float:
    converted = _finite_scalar(value, field_name, "must be a finite value in [0, 1]")
    if not 0.0 <= converted <= 1.0:
        raise ValueError(f"{field_name} must be a finite value in [0, 1]")
    return converted


def _finite_nonnegative_scalar(value: Any, field_name: str) -> float:
    converted = _finite_scalar(value, field_name, "must be a finite non-negative value")
    if converted < 0.0:
        raise ValueError(f"{field_name} must be a finite non-negative value")
    return converted


def _finite_scalar(value: Any, field_name: str, detail: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} {detail}")
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} {detail}") from exc
    if not np.isfinite(converted):
        raise ValueError(f"{field_name} {detail}")
    return converted


__all__ = ["install_registration_selection_validation"]
