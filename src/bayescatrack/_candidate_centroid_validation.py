"""Strict validation for centroid-based candidate prefilters.

Centroid candidate masks derive distances and top-k neighborhoods from ROI
centroid coordinates. NaN or infinite coordinates should be rejected explicitly so
malformed centroid metadata cannot silently produce dense or unstable candidate
sets.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_candidate_centroid_validation_patch"


def install_candidate_centroid_validation(candidate_prefilter_module: Any) -> None:
    """Install idempotent finite-coordinate validation for centroid masks."""

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


__all__ = ["install_candidate_centroid_validation"]
