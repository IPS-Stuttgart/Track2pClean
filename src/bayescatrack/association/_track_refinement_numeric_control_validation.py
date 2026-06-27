"""Strict validation for track-refinement numeric control scalars.

``TrackSmoothingConfig`` thresholds are scalar controls, but the underlying
implementation normalizes them with ``float(...)``.  NumPy currently allows
``float(np.array([3.5]))`` and ``float(np.array([True]))`` with only a warning,
which can silently turn array-like or boolean controls into valid-looking
thresholds.  Validate the controls before that coercion happens.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_track_refinement_numeric_control_validation_patch"


def install_track_refinement_numeric_control_validation() -> None:
    """Install idempotent validation for track-refinement numeric controls."""

    from . import track_refinement as module  # pylint: disable=import-outside-toplevel

    original_post_init = module.TrackSmoothingConfig.__post_init__
    if getattr(original_post_init, _PATCH_MARKER, False):
        return

    @wraps(original_post_init)
    def checked_post_init(self: Any) -> None:
        residual_z_threshold = _normalize_float_control(
            self.residual_z_threshold,
            name="residual_z_threshold",
            allow_zero=False,
        )
        min_edge_residual = _normalize_float_control(
            self.min_edge_residual,
            name="min_edge_residual",
            allow_zero=True,
        )
        original_post_init(self)
        object.__setattr__(self, "residual_z_threshold", residual_z_threshold)
        object.__setattr__(self, "min_edge_residual", min_edge_residual)

    setattr(checked_post_init, _PATCH_MARKER, True)
    setattr(checked_post_init, "_bayescatrack_original", original_post_init)
    module.TrackSmoothingConfig.__post_init__ = checked_post_init


def _normalize_float_control(value: Any, *, name: str, allow_zero: bool) -> float:
    bound_description = "finite non-negative" if allow_zero else "finite positive"
    message = f"{name} must be a {bound_description} value"

    if isinstance(value, (bool, np.bool_, bytes, bytearray, np.bytes_)):
        raise ValueError(message)

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(message)
        try:
            normalized = float(stripped)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(message) from exc
    else:
        try:
            value_array = np.asarray(value, dtype=object)
        except (TypeError, ValueError) as exc:
            raise ValueError(message) from exc
        if value_array.shape != ():
            raise ValueError(message)
        scalar = value_array.item()
        if isinstance(scalar, (bool, np.bool_, str, bytes, bytearray, np.bytes_)):
            raise ValueError(message)
        try:
            normalized = float(scalar)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(message) from exc

    if not np.isfinite(normalized):
        raise ValueError(message)
    if allow_zero:
        if normalized < 0.0:
            raise ValueError(message)
    elif normalized <= 0.0:
        raise ValueError(message)
    return normalized


__all__ = ["install_track_refinement_numeric_control_validation"]
