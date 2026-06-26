"""Strict validation for registration ROI-mask warp controls.

Registration mask warping accepts a ``binarize`` flag and threshold that are often
threaded through experiment configs.  Python truthiness would otherwise turn
malformed values such as ``"false"`` or ``1`` into an enabled binarization gate,
and boolean thresholds would be reinterpreted as numeric ``0.0``/``1.0``.  The
hook below preserves valid numeric thresholds while failing fast for ambiguous
runtime controls before registered masks are changed.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_registration_warp_validation_patch"


def install_registration_warp_validation() -> None:
    """Install idempotent validation around registration ROI-mask warping."""

    from . import registration as _registration  # pylint: disable=import-outside-toplevel

    original_warp = _registration.warp_roi_masks_into_reference_frame
    if getattr(original_warp, _PATCH_MARKER, False):
        return

    @wraps(original_warp)
    def warp_roi_masks_into_reference_frame_with_validation(
        roi_masks: Any,
        reference_to_measurement_matrix: Any,
        reference_to_measurement_offset: Any,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        normalized_kwargs = _normalize_registration_warp_kwargs(kwargs)
        return original_warp(
            roi_masks,
            reference_to_measurement_matrix,
            reference_to_measurement_offset,
            *args,
            **normalized_kwargs,
        )

    setattr(warp_roi_masks_into_reference_frame_with_validation, _PATCH_MARKER, True)
    setattr(
        warp_roi_masks_into_reference_frame_with_validation,
        "_bayescatrack_original",
        original_warp,
    )
    _registration.warp_roi_masks_into_reference_frame = (  # type: ignore[assignment]
        warp_roi_masks_into_reference_frame_with_validation
    )


def _normalize_registration_warp_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    normalized_kwargs = dict(kwargs)
    if "binarize" in normalized_kwargs:
        normalized_kwargs["binarize"] = _strict_bool(
            normalized_kwargs["binarize"],
            name="binarize",
        )
    if "threshold" in normalized_kwargs:
        normalized_kwargs["threshold"] = _finite_unit_interval_float(
            normalized_kwargs["threshold"],
            name="threshold",
        )
    return normalized_kwargs


def _strict_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def _finite_unit_interval_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite value in [0, 1]")
    numeric_value = float(value)
    if not np.isfinite(numeric_value) or numeric_value < 0.0 or numeric_value > 1.0:
        raise ValueError(f"{name} must be a finite value in [0, 1]")
    return numeric_value


__all__ = ["install_registration_warp_validation"]
