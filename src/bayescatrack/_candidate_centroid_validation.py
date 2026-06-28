"""Strict validation for centroid-based candidate prefilters.

Centroid candidate masks derive distances and top-k neighborhoods from ROI
centroid coordinates. NaN or infinite coordinates should be rejected explicitly so
malformed centroid metadata cannot silently produce dense or unstable candidate
sets. Candidate prefilter scalar controls are also validated before Python/NumPy
scalar coercion can reinterpret malformed array- or text-valued configuration
fields as valid benchmark knobs.
"""

from __future__ import annotations

import operator
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_candidate_centroid_validation_patch"


def install_candidate_centroid_validation(candidate_prefilter_module: Any) -> None:
    """Install idempotent finite-coordinate and scalar-control validation."""

    _install_scalar_control_validation(candidate_prefilter_module)

    original = candidate_prefilter_module.centroid_candidate_mask
    if _method_chain_has_patch(original):
        return

    @wraps(original)
    def centroid_candidate_mask_with_finite_centroid_validation(
        reference_centroids: Any,
        measurement_centroids: Any,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        _validate_finite_centroid_coordinates(
            reference_centroids,
            name="reference_centroids",
        )
        _validate_finite_centroid_coordinates(
            measurement_centroids,
            name="measurement_centroids",
        )
        return original(reference_centroids, measurement_centroids, *args, **kwargs)

    setattr(
        centroid_candidate_mask_with_finite_centroid_validation,
        _PATCH_MARKER,
        True,
    )
    setattr(
        centroid_candidate_mask_with_finite_centroid_validation,
        "_bayescatrack_original",
        original,
    )
    candidate_prefilter_module.centroid_candidate_mask = (  # type: ignore[assignment]
        centroid_candidate_mask_with_finite_centroid_validation
    )


def _install_scalar_control_validation(candidate_prefilter_module: Any) -> None:
    candidate_prefilter_module._positive_int = _positive_int
    candidate_prefilter_module._finite_nonnegative_float = _finite_nonnegative_float
    candidate_prefilter_module._finite_positive_float = _finite_positive_float


def _method_chain_has_patch(method: Any) -> bool:
    seen: set[int] = set()
    current: Any = method
    while current is not None:
        current_id = id(current)
        if current_id in seen:
            return False
        if getattr(current, _PATCH_MARKER, False):
            return True
        seen.add(current_id)
        current = getattr(current, "_bayescatrack_original", None)
    return False


def _validate_finite_centroid_coordinates(values: Any, *, name: str) -> None:
    try:
        points = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric centroid coordinates") from exc
    if points.ndim != 2:
        return
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must contain only finite centroid coordinates")


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, np.ndarray):
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
    if integer_value < 1:
        raise ValueError(f"{name} must be at least 1")
    return int(integer_value)


def _strict_float_scalar(value: Any, *, message: str) -> float:
    if isinstance(value, (bool, np.bool_, str, bytes, bytearray, np.ndarray)):
        raise ValueError(message)
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc


def _finite_nonnegative_float(value: Any, *, name: str) -> float:
    message = f"{name} must be a finite non-negative value"
    numeric_value = _strict_float_scalar(value, message=message)
    if not np.isfinite(numeric_value) or numeric_value < 0.0:
        raise ValueError(message)
    return numeric_value


def _finite_positive_float(value: Any, *, name: str) -> float:
    message = f"{name} must be a finite positive value"
    numeric_value = _strict_float_scalar(value, message=message)
    if not np.isfinite(numeric_value) or numeric_value <= 0.0:
        raise ValueError(message)
    return numeric_value


__all__ = ["install_candidate_centroid_validation"]
