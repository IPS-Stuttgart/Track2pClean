"""Validation patch for activity-similarity runtime controls."""

from __future__ import annotations

import operator
from typing import Any

import numpy as np

from . import activity_similarity as _activity_similarity

_PATCH_MARKER = "_bayescatrack_activity_similarity_control_validation_patch"
_ORIGINAL_ATTR = "_bayescatrack_activity_similarity_control_validation_original"
_TEXT_TYPES = (str, bytes, bytearray, np.str_, np.bytes_)


def install_activity_similarity_control_validation() -> None:
    """Reject malformed scalar controls before activity components are computed."""

    if getattr(_activity_similarity, _PATCH_MARKER, False):
        return

    original = _activity_similarity.activity_similarity_components

    def validated_activity_similarity_components(
        reference_plane: Any,
        measurement_plane: Any,
        *,
        trace_source: str = "auto",
        similarity_epsilon: Any = 1.0e-12,
        event_threshold: Any = 0.0,
    ) -> dict[str, np.ndarray]:
        _validated_plane_roi_count(reference_plane, "reference_plane")
        _validated_plane_roi_count(measurement_plane, "measurement_plane")
        return original(
            reference_plane,
            measurement_plane,
            trace_source=trace_source,
            similarity_epsilon=_finite_positive_float(
                similarity_epsilon,
                name="similarity_epsilon",
            ),
            event_threshold=_finite_float(event_threshold, name="event_threshold"),
        )

    validated_activity_similarity_components.__name__ = original.__name__
    validated_activity_similarity_components.__qualname__ = original.__qualname__
    setattr(validated_activity_similarity_components, _ORIGINAL_ATTR, original)
    _activity_similarity.activity_similarity_components = (
        validated_activity_similarity_components
    )
    setattr(_activity_similarity, _PATCH_MARKER, True)


def _validated_plane_roi_count(plane: Any, plane_name: str) -> int:
    message = f"{plane_name}.n_rois must be a finite non-negative integer"
    raw_count = getattr(plane, "n_rois", None)
    if isinstance(raw_count, np.ndarray):
        if raw_count.ndim != 0 or np.issubdtype(raw_count.dtype, np.bool_):
            raise ValueError(message)
        raw_count = raw_count.item()

    if isinstance(raw_count, (bool, np.bool_)):
        raise ValueError(message)

    try:
        count = int(operator.index(raw_count))
    except TypeError:
        try:
            numeric_count = float(raw_count)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(message) from exc
        if not np.isfinite(numeric_count) or not numeric_count.is_integer():
            raise ValueError(message)
        count = int(numeric_count)
    except (ValueError, OverflowError) as exc:
        raise ValueError(message) from exc

    if count < 0:
        raise ValueError(message)
    return count


def _coerce_scalar_float(value: Any, *, error_message: str) -> float:
    if isinstance(value, (bool, np.bool_, *_TEXT_TYPES)):
        raise ValueError(error_message)

    if isinstance(value, np.ndarray):
        if value.ndim != 0 or np.issubdtype(value.dtype, np.bool_):
            raise ValueError(error_message)
        value = value.item()

    if isinstance(value, (bool, np.bool_, *_TEXT_TYPES)):
        raise ValueError(error_message)

    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(error_message) from exc


def _finite_positive_float(value: Any, *, name: str) -> float:
    error_message = f"{name} must be a finite positive value"
    numeric_value = _coerce_scalar_float(value, error_message=error_message)
    if not np.isfinite(numeric_value) or numeric_value <= 0.0:
        raise ValueError(error_message)
    return numeric_value


def _finite_float(value: Any, *, name: str) -> float:
    error_message = f"{name} must be finite"
    numeric_value = _coerce_scalar_float(value, error_message=error_message)
    if not np.isfinite(numeric_value):
        raise ValueError(error_message)
    return numeric_value


__all__ = ["install_activity_similarity_control_validation"]
