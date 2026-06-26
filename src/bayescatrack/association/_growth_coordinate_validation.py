"""Finite-coordinate validation for growth-prior helpers.

Growth-prior transforms and penalty matrices operate on Suite2p ROI centroids.
The core helpers already validate scalar controls, but malformed coordinate
arrays with NaN or infinite entries could still reach affine fitting and radial
penalty construction. That can silently produce non-finite transforms or costs
that are only masked much later, obscuring the bad input.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

import numpy as np

_PATCH_MARKER = "_bayescatrack_growth_coordinate_validation_patch"


def install_growth_coordinate_validation() -> None:
    """Install idempotent finite-coordinate checks on growth-prior helpers."""

    from . import growth_priors as _growth_priors  # pylint: disable=import-outside-toplevel

    original_xy_matrix: Callable[..., Any] = _growth_priors._as_xy_point_matrix  # pylint: disable=protected-access
    if not getattr(original_xy_matrix, _PATCH_MARKER, False):

        @wraps(original_xy_matrix)
        def as_xy_point_matrix_with_coordinate_validation(
            values: Any,
            *,
            name: str,
            peer_values: Any | None = None,
        ) -> np.ndarray:
            points = original_xy_matrix(values, name=name, peer_values=peer_values)
            _require_finite_array(points, name=name, value_description="finite xy coordinates")
            return np.ascontiguousarray(points, dtype=float)

        _mark_patched(as_xy_point_matrix_with_coordinate_validation, original_xy_matrix)
        _growth_priors._as_xy_point_matrix = as_xy_point_matrix_with_coordinate_validation  # pylint: disable=protected-access

    original_residuals: Callable[..., Any] = _growth_priors.affine_growth_residuals
    if not getattr(original_residuals, _PATCH_MARKER, False):

        @wraps(original_residuals)
        def affine_growth_residuals_with_coordinate_validation(
            source_points_xy: Any,
            target_points_xy: Any,
            *,
            affine: Any,
        ) -> np.ndarray:
            return original_residuals(
                source_points_xy,
                target_points_xy,
                affine=_normalize_affine_matrix(affine, name="affine"),
            )

        _mark_patched(affine_growth_residuals_with_coordinate_validation, original_residuals)
        _growth_priors.affine_growth_residuals = (
            affine_growth_residuals_with_coordinate_validation
        )

    original_affine_penalty: Callable[..., Any] = _growth_priors.affine_growth_penalty_matrix
    if not getattr(original_affine_penalty, _PATCH_MARKER, False):

        @wraps(original_affine_penalty)
        def affine_growth_penalty_matrix_with_coordinate_validation(
            reference_centroids_xy: Any,
            measurement_centroids_xy: Any,
            affine_xy: Any,
            *,
            scale: float,
        ) -> np.ndarray:
            return original_affine_penalty(
                reference_centroids_xy,
                measurement_centroids_xy,
                _normalize_affine_matrix(affine_xy, name="affine_xy"),
                scale=scale,
            )

        _mark_patched(
            affine_growth_penalty_matrix_with_coordinate_validation,
            original_affine_penalty,
        )
        _growth_priors.affine_growth_penalty_matrix = (
            affine_growth_penalty_matrix_with_coordinate_validation
        )

    original_radial_penalty: Callable[..., Any] = _growth_priors.radial_growth_penalty_matrix
    if not getattr(original_radial_penalty, _PATCH_MARKER, False):

        @wraps(original_radial_penalty)
        def radial_growth_penalty_matrix_with_coordinate_validation(
            reference_centroids_xy: Any,
            measurement_centroids_xy: Any,
            *,
            center_xy: Any | None = None,
            scale: float = 10.0,
        ) -> np.ndarray:
            normalized_center = (
                None
                if center_xy is None
                else _normalize_xy_vector(center_xy, name="center_xy")
            )
            return original_radial_penalty(
                reference_centroids_xy,
                measurement_centroids_xy,
                center_xy=normalized_center,
                scale=scale,
            )

        _mark_patched(
            radial_growth_penalty_matrix_with_coordinate_validation,
            original_radial_penalty,
        )
        _growth_priors.radial_growth_penalty_matrix = (
            radial_growth_penalty_matrix_with_coordinate_validation
        )


def _mark_patched(wrapper: Callable[..., Any], original: Callable[..., Any]) -> None:
    setattr(wrapper, _PATCH_MARKER, True)
    setattr(wrapper, "_bayescatrack_original", original)


def _normalize_affine_matrix(values: Any, *, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have shape (2, 3)") from exc
    if matrix.shape != (2, 3):
        raise ValueError(f"{name} must have shape (2, 3)")
    _require_finite_array(matrix, name=name, value_description="finite values")
    return np.ascontiguousarray(matrix, dtype=float)


def _normalize_xy_vector(values: Any, *, name: str) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite xy coordinates") from exc
    if vector.size != 2:
        raise ValueError(f"{name} must have shape (2,)")
    vector = vector.reshape(2)
    _require_finite_array(vector, name=name, value_description="finite xy coordinates")
    return np.ascontiguousarray(vector, dtype=float)


def _require_finite_array(array: np.ndarray, *, name: str, value_description: str) -> None:
    if not np.all(np.isfinite(np.asarray(array, dtype=float))):
        raise ValueError(f"{name} must contain {value_description}")


__all__ = ["install_growth_coordinate_validation"]
