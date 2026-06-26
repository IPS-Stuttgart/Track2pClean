"""Track-level smoothing and repair suggestions for longitudinal ROI tracks."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class TrackSmoothingConfig:
    """Configuration for geometry-based track refinement diagnostics."""

    residual_z_threshold: float = 3.5
    min_track_detections: int = 3
    min_edge_residual: float = 5.0
    split_bad_edges: bool = True
    fill_value: int = -1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "residual_z_threshold",
            _finite_positive_float(
                self.residual_z_threshold, name="residual_z_threshold"
            ),
        )
        object.__setattr__(
            self,
            "min_track_detections",
            _integer_at_least(
                self.min_track_detections, name="min_track_detections", minimum=2
            ),
        )
        object.__setattr__(
            self,
            "min_edge_residual",
            _finite_nonnegative_float(self.min_edge_residual, name="min_edge_residual"),
        )
        object.__setattr__(
            self,
            "split_bad_edges",
            _strict_bool(self.split_bad_edges, name="split_bad_edges"),
        )
        object.__setattr__(
            self,
            "fill_value",
            _negative_integer_sentinel(self.fill_value, name="fill_value"),
        )


@dataclass(frozen=True)
class TrackGeometryIssue:
    """One high-residual track edge or detection."""

    track_index: int
    session_index: int
    roi_index: int
    residual: float
    robust_z: float
    suggested_action: str


def roi_position_tables_from_sessions(
    sessions: Sequence[Any], *, order: str = "xy", weighted: bool = False
) -> tuple[dict[int, np.ndarray], ...]:
    """Return ``{suite2p_roi_index: xy_position}`` for each session."""

    tables: list[dict[int, np.ndarray]] = []
    for session in sessions:
        plane = session.plane_data
        centroids = np.asarray(
            plane.centroids(order=order, weighted=weighted), dtype=float
        )
        if centroids.ndim != 2 or centroids.shape[0] != 2:
            raise ValueError("session centroids must have shape (2, n_roi)")
        roi_indices = _position_table_roi_indices(plane, n_centroids=centroids.shape[1])
        tables.append(
            {
                int(roi_index): centroids[:, local_index].astype(float)
                for local_index, roi_index in enumerate(roi_indices)
            }
        )
    return tuple(tables)


def track_geometry_issues(
    track_rows: Any,
    position_tables: Sequence[Mapping[int, Any]],
    *,
    config: TrackSmoothingConfig | None = None,
) -> list[TrackGeometryIssue]:
    """Flag detections whose position is inconsistent with the full track."""

    cfg = config or TrackSmoothingConfig()
    rows = _validated_track_rows_and_position_tables(track_rows, position_tables)

    issues: list[TrackGeometryIssue] = []
    for track_index, row in enumerate(rows):
        session_indices, positions, roi_indices = _track_positions(
            row, position_tables, fill_value=cfg.fill_value
        )
        if positions.shape[0] < cfg.min_track_detections:
            continue
        fitted = _fit_linear_track(session_indices, positions)
        residuals = np.linalg.norm(positions - fitted, axis=1)
        z_scores = _robust_z_scores(residuals)
        for local_index, residual in enumerate(residuals):
            z_score = float(z_scores[local_index])
            if (
                residual >= cfg.min_edge_residual
                and z_score >= cfg.residual_z_threshold
            ):
                issues.append(
                    TrackGeometryIssue(
                        track_index=int(track_index),
                        session_index=int(session_indices[local_index]),
                        roi_index=int(roi_indices[local_index]),
                        residual=float(residual),
                        robust_z=z_score,
                        suggested_action="split_or_relink",
                    )
                )
    return issues


def smoothed_track_positions(
    track_rows: Any,
    position_tables: Sequence[Mapping[int, Any]],
    *,
    fill_value: int = -1,
) -> dict[int, dict[int, np.ndarray]]:
    """Return fitted per-track positions for present detections."""

    fill_value = _negative_integer_sentinel(fill_value, name="fill_value")
    rows = _validated_track_rows_and_position_tables(track_rows, position_tables)
    output: dict[int, dict[int, np.ndarray]] = {}
    for track_index, row in enumerate(rows):
        session_indices, positions, _roi_indices = _track_positions(
            row, position_tables, fill_value=fill_value
        )
        if positions.shape[0] < 2:
            continue
        fitted = _fit_linear_track(session_indices, positions)
        output[int(track_index)] = {
            int(session_index): fitted_position.astype(float)
            for session_index, fitted_position in zip(
                session_indices, fitted, strict=True
            )
        }
    return output


def split_tracks_at_issues(
    track_rows: Any,
    issues: Sequence[TrackGeometryIssue],
    *,
    fill_value: int = -1,
) -> np.ndarray:
    """Return a copy of ``track_rows`` split at flagged high-residual detections.

    The function is conservative: it does not invent replacement links.  It only
    turns a suspicious continuous row into two fragments, allowing benchmark code
    to determine whether avoiding a false continuation improves F1.
    """

    fill_value = _negative_integer_sentinel(fill_value, name="fill_value")
    rows = _validated_track_row_matrix(track_rows)
    issue_map: dict[int, list[int]] = {}
    for issue in issues:
        issue_map.setdefault(int(issue.track_index), []).append(
            int(issue.session_index)
        )

    output_rows: list[np.ndarray] = []
    for track_index, row in enumerate(rows):
        cut_points = sorted(
            {idx for idx in issue_map.get(track_index, []) if 0 < idx < rows.shape[1]}
        )
        if not cut_points:
            output_rows.append(row.copy())
            continue
        start = 0
        for cut_point in cut_points:
            fragment = np.full(row.shape, fill_value, dtype=int)
            fragment[start:cut_point] = row[start:cut_point]
            if np.any(fragment != fill_value):
                output_rows.append(fragment)
            start = cut_point
        fragment = np.full(row.shape, fill_value, dtype=int)
        fragment[start:] = row[start:]
        if np.any(fragment != fill_value):
            output_rows.append(fragment)
    return np.vstack(output_rows) if output_rows else rows[:0]


def geometry_issue_rows(
    issues: Sequence[TrackGeometryIssue],
) -> list[dict[str, float | int | str]]:
    """Serialize geometry issues for CSV/JSON reports."""

    return [
        {
            "track_index": issue.track_index,
            "session_index": issue.session_index,
            "roi_index": issue.roi_index,
            "residual": issue.residual,
            "robust_z": issue.robust_z,
            "suggested_action": issue.suggested_action,
        }
        for issue in issues
    ]


def _position_table_roi_indices(plane: Any, *, n_centroids: int) -> np.ndarray:
    n_rois = _integer_at_least(getattr(plane, "n_rois", n_centroids), name="plane.n_rois", minimum=0)
    if n_rois != n_centroids:
        raise ValueError("plane.n_rois must match the number of centroid columns")

    roi_indices = getattr(plane, "roi_indices", None)
    if roi_indices is None:
        return np.arange(n_centroids, dtype=int)

    normalized = _nonnegative_integer_vector(roi_indices, name="plane.roi_indices")
    if normalized.shape != (n_centroids,):
        raise ValueError("plane.roi_indices must have one entry per centroid")
    if len(set(normalized.tolist())) != n_centroids:
        raise ValueError("plane.roi_indices must contain unique ROI indices")
    return normalized


def _nonnegative_integer_vector(values: Any, *, name: str) -> np.ndarray:
    value_array = np.asarray(values, dtype=object).reshape(-1)
    normalized = np.empty(value_array.shape, dtype=int)
    for index, value in np.ndenumerate(value_array):
        integer_value = _integer_value(value, name=name)
        if integer_value < 0:
            raise ValueError(f"{name} must contain non-negative ROI indices")
        normalized[index] = integer_value
    return normalized


def _validated_track_rows_and_position_tables(
    track_rows: Any,
    position_tables: Sequence[Mapping[int, Any]],
) -> np.ndarray:
    rows = _validated_track_row_matrix(track_rows)
    if rows.shape[1] != len(position_tables):
        raise ValueError("position_tables must contain one table per session")
    return rows


def _validated_track_row_matrix(track_rows: Any) -> np.ndarray:
    rows = np.asarray(track_rows)
    if rows.ndim != 2:
        raise ValueError("track_rows must be two-dimensional")
    if _contains_ambiguous_track_row_tokens(rows):
        raise ValueError("track_rows must contain finite integer ROI indices")
    if np.issubdtype(rows.dtype, np.integer):
        return rows.astype(int, copy=False)

    try:
        numeric_rows = np.asarray(rows, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("track_rows must contain finite integer ROI indices") from exc
    if not np.all(np.isfinite(numeric_rows)) or not np.all(
        numeric_rows == np.floor(numeric_rows)
    ):
        raise ValueError("track_rows must contain finite integer ROI indices")
    return numeric_rows.astype(int, copy=False)


def _contains_ambiguous_track_row_tokens(rows: np.ndarray) -> bool:
    if np.issubdtype(rows.dtype, np.bool_) or rows.dtype.kind in {"S", "U"}:
        return True
    if rows.dtype != object:
        return False
    return any(isinstance(value, (bool, np.bool_, str, bytes)) for value in rows.ravel())


def _track_positions(
    row: np.ndarray,
    position_tables: Sequence[Mapping[int, Any]],
    *,
    fill_value: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fill_value = _negative_integer_sentinel(fill_value, name="fill_value")
    session_indices: list[int] = []
    positions: list[np.ndarray] = []
    roi_indices: list[int] = []
    for session_index, roi_index in enumerate(row):
        roi_int = int(roi_index)
        if roi_int == fill_value:
            continue
        position = position_tables[session_index].get(roi_int)
        if position is None:
            continue
        session_indices.append(session_index)
        roi_indices.append(roi_int)
        positions.append(
            _validated_position_vector(
                position, session_index=session_index, roi_index=roi_int
            )
        )
    if not positions:
        return (
            np.zeros((0,), dtype=int),
            np.zeros((0, 2), dtype=float),
            np.zeros((0,), dtype=int),
        )
    return (
        np.asarray(session_indices, dtype=int),
        np.vstack(positions),
        np.asarray(roi_indices, dtype=int),
    )


def _validated_position_vector(
    position: Any,
    *,
    session_index: int,
    roi_index: int,
) -> np.ndarray:
    try:
        vector = np.asarray(position, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"position_tables[{session_index}][{roi_index}] must be a finite 2-vector"
        ) from exc
    if vector.size != 2:
        raise ValueError(
            f"position_tables[{session_index}][{roi_index}] must be a finite 2-vector"
        )
    vector = vector.reshape(2)
    if not np.all(np.isfinite(vector)):
        raise ValueError(
            f"position_tables[{session_index}][{roi_index}] must be a finite 2-vector"
        )
    return vector


def _fit_linear_track(session_indices: np.ndarray, positions: np.ndarray) -> np.ndarray:
    t = np.asarray(session_indices, dtype=float)
    x = np.column_stack((np.ones_like(t), t))
    coef, *_ = np.linalg.lstsq(x, positions, rcond=None)
    return x @ coef


def _robust_z_scores(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    median = float(np.median(values))
    deviations = np.abs(values - median)
    mad = float(np.median(deviations))
    scale = 1.4826 * mad
    if np.isfinite(scale) and scale > 1.0e-12:
        return deviations / scale

    z_scores = np.zeros(values.shape, dtype=float)
    non_median = np.isfinite(deviations) & (deviations > 1.0e-12)
    z_scores[non_median] = np.inf
    z_scores[~np.isfinite(deviations)] = np.inf
    return z_scores


def _finite_positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite positive value")
    numeric_value = float(value)
    if not np.isfinite(numeric_value) or numeric_value <= 0.0:
        raise ValueError(f"{name} must be a finite positive value")
    return numeric_value


def _finite_nonnegative_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite non-negative value")
    numeric_value = float(value)
    if not np.isfinite(numeric_value) or numeric_value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative value")
    return numeric_value


def _integer_at_least(value: Any, *, name: str, minimum: int) -> int:
    integer_value = _integer_value(value, name=name)
    if integer_value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return integer_value


def _negative_integer_sentinel(value: Any, *, name: str) -> int:
    integer_value = _integer_value(value, name=name)
    if integer_value >= 0:
        raise ValueError(f"{name} must be a negative integer sentinel")
    return integer_value


def _integer_value(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or not float(value).is_integer():
            raise ValueError(f"{name} must be an integer")
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{name} must be an integer")
        try:
            numeric_value = float(stripped)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(f"{name} must be an integer")
        return int(numeric_value)
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _strict_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)
