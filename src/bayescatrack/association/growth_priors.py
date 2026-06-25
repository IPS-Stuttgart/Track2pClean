"""Growth and deformation priors for longitudinal ROI association."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

_ROI_INDEX_ERROR = "track_rows entries must be non-negative integers or -1 missing sentinels"
_MISSING_VALUE_STRINGS = {"", "na", "nan", "none", "null", "-"}


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
    rows = _normalize_track_rows(track_rows)
    if rows.ndim != 2:
        raise ValueError("track_rows must have shape (n_tracks, n_sessions)")
    if rows.shape[1] == 0:
        raise ValueError("track_rows must contain at least one session")
    n_sessions = int(rows.shape[1])
    source_index = _normalize_session_index(
        source_session,
        name="source_session",
        n_sessions=n_sessions,
    )
    target_index = _normalize_session_index(
        target_session,
        name="target_session",
        n_sessions=n_sessions,
    )
    if len(position_tables) <= max(source_index, target_index):
        raise ValueError("position_tables must contain entries for the selected sessions")
    source_points: list[np.ndarray] = []
    target_points: list[np.ndarray] = []
    for row in rows:
        source_roi = int(row[source_index])
        target_roi = int(row[target_index])
        if source_roi < 0 or target_roi < 0:
            continue
        source_pos = position_tables[source_index].get(source_roi)
        target_pos = position_tables[target_index].get(target_roi)
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


def _normalize_track_rows(track_rows: Any) -> np.ndarray:
    raw_rows = np.asarray(track_rows, dtype=object)
    if raw_rows.ndim != 2:
        return raw_rows
    normalized = np.empty(raw_rows.shape, dtype=int)
    for index, value in np.ndenumerate(raw_rows):
        normalized[index] = _normalize_roi_index(value)
    return normalized


def _normalize_roi_index(value: Any) -> int:
    if value is None:
        return -1
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(_ROI_INDEX_ERROR)
    try:
        return _validate_roi_index(operator.index(value))
    except TypeError:
        pass
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if math.isnan(numeric):
            return -1
        if math.isfinite(numeric) and numeric.is_integer():
            return _validate_roi_index(int(numeric))
        raise ValueError(_ROI_INDEX_ERROR)
    if isinstance(value, str):
        return _normalize_roi_index_string(value)
    raise ValueError(_ROI_INDEX_ERROR)


def _normalize_roi_index_string(value: str) -> int:
    text = value.strip()
    if text.lower().replace(" ", "_") in _MISSING_VALUE_STRINGS:
        return -1
    try:
        numeric = float(text)
    except ValueError as exc:
        raise ValueError(_ROI_INDEX_ERROR) from exc
    if math.isnan(numeric):
        return -1
    if math.isfinite(numeric) and numeric.is_integer():
        return _validate_roi_index(int(numeric))
    raise ValueError(_ROI_INDEX_ERROR)


def _validate_roi_index(value: int) -> int:
    normalized = int(value)
    if normalized == -1 or normalized >= 0:
        return normalized
    raise ValueError(_ROI_INDEX_ERROR)


def _normalize_session_index(value: Any, *, name: str, n_sessions: int) -> int:
    raw_index = _integer_value(value, name=name)
    index = raw_index + n_sessions if raw_index < 0 else raw_index
    if index < 0 or index >= n_sessions:
        raise ValueError(f"{name} must select an existing session")
    return index


def _integer_value(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer session index")
    try:
        return int(operator.index(value))
    except TypeError:
        pass
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if math.isfinite(numeric) and numeric.is_integer():
            return int(numeric)
        raise ValueError(f"{name} must be an integer session index")
    if isinstance(value, str):
        text = value.strip()
        try:
            numeric = float(text)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer session index") from exc
        if math.isfinite(numeric) and numeric.is_integer():
            return int(numeric)
    raise ValueError(f"{name} must be an integer session index")


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
