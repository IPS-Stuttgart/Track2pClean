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
_XY_SHAPE_ERROR = "must have shape (n, 2) or (2, n)"


def install_growth_coordinate_validation() -> None:
    """Install idempotent finite-coordinate checks on growth-prior helpers."""

    from . import growth_priors as _growth_priors  # pylint: disable=import-outside-toplevel

    original_fit: Callable[..., Any] = _growth_priors.fit_affine_growth_transform
    if not getattr(original_fit, _PATCH_MARKER, False):

        @wraps(original_fit)
        def fit_affine_growth_transform_with_coordinate_validation(
            source_points_xy: Any,
            target_points_xy: Any,
            *,
            regularization: float = 1.0e-6,
        ) -> np.ndarray:
            return original_fit(
                _normalize_xy_point_matrix(
                    source_points_xy,
                    name="source_points_xy",
                    peer_values=target_points_xy,
                ),
                _normalize_xy_point_matrix(
                    target_points_xy,
                    name="target_points_xy",
                    peer_values=source_points_xy,
                ),
                regularization=regularization,
            )

        _mark_patched(fit_affine_growth_transform_with_coordinate_validation, original_fit)
        _growth_priors.fit_affine_growth_transform = (
            fit_affine_growth_transform_with_coordinate_validation
        )

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
                _normalize_xy_point_matrix(
                    source_points_xy,
                    name="source_points_xy",
                    peer_values=target_points_xy,
                ),
                _normalize_xy_point_matrix(
                    target_points_xy,
                    name="target_points_xy",
                    peer_values=source_points_xy,
                ),
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
                _normalize_xy_point_matrix(
                    reference_centroids_xy,
                    name="reference_centroids_xy",
                    peer_values=measurement_centroids_xy,
                ),
                _normalize_xy_point_matrix(
                    measurement_centroids_xy,
                    name="measurement_centroids_xy",
                    peer_values=reference_centroids_xy,
                ),
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
                _normalize_xy_point_matrix(
                    reference_centroids_xy,
                    name="reference_centroids_xy",
                    peer_values=measurement_centroids_xy,
                ),
                _normalize_xy_point_matrix(
                    measurement_centroids_xy,
                    name="measurement_centroids_xy",
                    peer_values=reference_centroids_xy,
                ),
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


def _normalize_xy_point_matrix(
    values: Any,
    *,
    name: str,
    peer_values: Any | None = None,
) -> np.ndarray:
    try:
        points = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite xy coordinates") from exc

    if points.ndim != 2:
        raise ValueError(f"{name} {_XY_SHAPE_ERROR}")

    if points.shape == (2, 2):
        peer_layout = _unambiguous_xy_layout(peer_values)
        normalized = points if peer_layout == "point_rows" else points.T
    elif points.shape[1] == 2:
        normalized = points
    elif points.shape[0] == 2:
        normalized = points.T
    else:
        raise ValueError(f"{name} {_XY_SHAPE_ERROR}")

    if not np.all(np.isfinite(normalized)):
        raise ValueError(f"{name} must contain finite xy coordinates")
    return np.ascontiguousarray(normalized, dtype=float)


def _unambiguous_xy_layout(values: Any | None) -> str | None:
    if values is None:
        return None
    try:
        points = np.asarray(values)
    except (TypeError, ValueError):
        return None
    if points.ndim != 2:
        return None
    if points.shape[1] == 2 and points.shape[0] != 2:
        return "point_rows"
    if points.shape[0] == 2 and points.shape[1] != 2:
        return "coordinate_rows"
    return None


def _normalize_affine_matrix(values: Any, *, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have shape (2, 3)") from exc
    if matrix.shape != (2, 3):
        raise ValueError(f"{name} must have shape (2, 3)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite values")
    return np.ascontiguousarray(matrix, dtype=float)


def _normalize_xy_vector(values: Any, *, name: str) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite xy coordinates") from exc
    if vector.size != 2:
        raise ValueError(f"{name} must have shape (2,)")
    vector = vector.reshape(2)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain finite xy coordinates")
    return np.ascontiguousarray(vector, dtype=float)


__all__ = ["install_growth_coordinate_validation"]
