"""Validation patch for activity-similarity runtime controls."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import activity_similarity as _activity_similarity

_PATCH_MARKER = "_bayescatrack_activity_similarity_control_validation_patch"
_ORIGINAL_ATTR = "_bayescatrack_activity_similarity_control_validation_original"


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
    _activity_similarity.activity_similarity_components = validated_activity_similarity_components
    setattr(_activity_similarity, _PATCH_MARKER, True)


def _finite_positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite positive value")
    numeric_value = float(value)
    if not np.isfinite(numeric_value) or numeric_value <= 0.0:
        raise ValueError(f"{name} must be a finite positive value")
    return numeric_value


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite")
    numeric_value = float(value)
    if not np.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite")
    return numeric_value


__all__ = ["install_activity_similarity_control_validation"]
