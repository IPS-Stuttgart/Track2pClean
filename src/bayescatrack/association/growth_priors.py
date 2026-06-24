"""Growth and deformation priors for longitudinal ROI association."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class GrowthPriorConfig:
    """Weights for adding expected growth/deformation penalties to costs."""

    affine_weight: float = 0.25
    radial_weight: float = 0.25
    displacement_scale: float = 10.0
    regularization: float = 1.0e-6

    def __post_init__(self) -> None:
        for name in ("affine_weight", "radial_weight", "regularization"):
            object.__setattr__(
                self,
                name,
                _nonnegative_float(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "displacement_scale",
            _positive_float(self.displacement_scale, name="displacement_scale"),
        )


def fit_affine_growth_transform(
    source_points_xy: Any, target_points_xy: Any, *, regularization: float = 1.0e-6
) -> np.ndarray:
    """Fit an affine transform mapping source points to target points."""

    source = np.asarray(source_points_xy, dtype=float)
    target = np.asarray(target_points_xy, dtype=float)
    regularization = _nonnegative_float(regularization, name="regularization")
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError(
            "source_points_xy and target_points_xy must both have shape (n, 2)"
        )
    if source.shape[0] < 3:
        raise ValueError("At least three point pairs are required for affine growth")
    design = np.column_stack((source, np.ones(source.shape[0], dtype=float)))
    normal = design.T @ design + regularization * np.eye(3)
    rhs = design.T @ target
    coef = np.linalg.solve(normal, rhs)
    return np.asarray(coef.T, dtype=float)


def estimate_affine_growth_field(
    source_points_xy: Any, target_points_xy: Any, *, regularization: float = 0.0
) -> np.ndarray:
    """Estimate an affine source-to-target growth field."""

    return fit_affine_growth_transform(
        source_points_xy, target_points_xy, regularization=regularization
    )


def affine_growth_residuals(
    source_points_xy: Any, target_points_xy: Any, *, affine: Any
) -> np.ndarray:
    """Return per-pair residual distances under an affine growth field."""

    source = np.asarray(source_points_xy, dtype=float)
    target = np.asarray(target_points_xy, dtype=float)
    matrix = np.asarray(affine, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError(
            "source_points_xy and target_points_xy must both have shape (n, 2)"
        )
    if matrix.shape != (2, 3):
        raise ValueError("affine must have shape (2, 3)")
    predicted = source @ matrix[:, :2].T + matrix[:, 2][None, :]
    return np.linalg.norm(predicted - target, axis=1)


def affine_growth_penalty_matrix(
    reference_centroids_xy: Any,
    measurement_centroids_xy: Any,
    affine_xy: Any,
    *,
    scale: float,
) -> np.ndarray:
    """Return normalized displacement residuals under an affine growth field."""

    ref = np.asarray(reference_centroids_xy, dtype=float)
    meas = np.asarray(measurement_centroids_xy, dtype=float)
    matrix = np.asarray(affine_xy, dtype=float)
    scale = _positive_float(scale, name="scale")
    if matrix.shape != (2, 3):
        raise ValueError("affine_xy must have shape (2, 3)")
    predicted = ref @ matrix[:, :2].T + matrix[:, 2][None, :]
    diffs = predicted[:, None, :] - meas[None, :, :]
    return np.linalg.norm(diffs, axis=2) / scale


def growth_penalty_matrix(
    reference_centroids_xy: Any,
    measurement_centroids_xy: Any,
    *,
    affine: Any | None = None,
    scale: float = 1.0,
) -> np.ndarray:
    """Return pairwise growth-consistency penalties."""

    affine_xy = (
        estimate_affine_growth_field(reference_centroids_xy, measurement_centroids_xy)
        if affine is None
        else affine
    )
    return affine_growth_penalty_matrix(
        reference_centroids_xy,
        measurement_centroids_xy,
        affine_xy,
        scale=scale,
    )


def radial_growth_penalty_matrix(
    reference_centroids_xy: Any,
    measurement_centroids_xy: Any,
    *,
    center_xy: Any | None = None,
    scale: float = 10.0,
) -> np.ndarray:
    """Return penalty for radial displacement inconsistency."""

    ref = np.asarray(reference_centroids_xy, dtype=float)
    meas = np.asarray(measurement_centroids_xy, dtype=float)
    scale = _positive_float(scale, name="scale")
    if center_xy is None:
        center = np.nanmean(ref, axis=0) if ref.size else np.zeros((2,), dtype=float)
    else:
        center = np.asarray(center_xy, dtype=float).reshape(2)
    ref_r = np.linalg.norm(ref - center[None, :], axis=1)
    meas_r = np.linalg.norm(meas - center[None, :], axis=1)
    radial_diff = np.abs(ref_r[:, None] - meas_r[None, :])
    return radial_diff / scale


def apply_growth_prior_to_costs(
    cost_matrix: Any,
    reference_centroids_xy: Any,
    measurement_centroids_xy: Any,
    *,
    affine_xy: Any | None = None,
    center_xy: Any | None = None,
    config: GrowthPriorConfig | None = None,
) -> np.ndarray:
    """Add affine/radial growth penalties to a cost matrix."""

    cfg = config or GrowthPriorConfig()
    costs = np.asarray(cost_matrix, dtype=float).copy()
    if affine_xy is not None and cfg.affine_weight > 0.0:
        costs += cfg.affine_weight * affine_growth_penalty_matrix(
            reference_centroids_xy,
            measurement_centroids_xy,
            affine_xy,
            scale=cfg.displacement_scale,
        )
    if cfg.radial_weight > 0.0:
        costs += cfg.radial_weight * radial_growth_penalty_matrix(
            reference_centroids_xy,
            measurement_centroids_xy,
            center_xy=center_xy,
            scale=cfg.displacement_scale,
        )
    return np.nan_to_num(costs, nan=1.0e6, posinf=1.0e6, neginf=1.0e6)


def estimate_growth_from_track_rows(
    track_rows: Any,
    position_tables: Sequence[Mapping[int, Any]],
    *,
    source_session: int = 0,
    target_session: int = -1,
    config: GrowthPriorConfig | None = None,
) -> np.ndarray:
    """Estimate an affine growth transform from complete links in a track matrix."""

    cfg = config or GrowthPriorConfig()
    rows = np.asarray(track_rows, dtype=int)
    if target_session < 0:
        target_session = rows.shape[1] + int(target_session)
    source_points: list[np.ndarray] = []
    target_points: list[np.ndarray] = []
    for row in rows:
        source_roi = int(row[source_session])
        target_roi = int(row[target_session])
        if source_roi < 0 or target_roi < 0:
            continue
        source_pos = position_tables[source_session].get(source_roi)
        target_pos = position_tables[target_session].get(target_roi)
        if source_pos is None or target_pos is None:
            continue
        source_points.append(np.asarray(source_pos, dtype=float).reshape(2))
        target_points.append(np.asarray(target_pos, dtype=float).reshape(2))
    if len(source_points) < 3:
        raise ValueError("At least three complete linked tracks are required")
    return fit_affine_growth_transform(
        np.vstack(source_points),
        np.vstack(target_points),
        regularization=cfg.regularization,
    )


def _nonnegative_float(value: Any, *, name: str) -> float:
    numeric = _finite_float(value, name=name)
    if numeric < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return numeric


def _positive_float(value: Any, *, name: str) -> float:
    numeric = _finite_float(value, name=name)
    if numeric <= 0.0:
        raise ValueError(f"{name} must be positive")
    return numeric


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric
