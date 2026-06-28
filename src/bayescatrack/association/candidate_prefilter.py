"""Candidate prefilters for large pairwise ROI association problems."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CentroidCandidatePrefilterConfig:
    """Configuration for centroid-based pairwise candidate generation.

    The helper is deliberately conservative and opt-in: it only constructs a
    boolean candidate mask or applies a finite large-cost sentinel to an already
    dense matrix. Existing callers can keep dense cost matrices unchanged, while
    large benchmark runs can use the mask to avoid evaluating expensive ROI
    evidence for clearly impossible pairs.
    """

    max_distance: float | None = None
    row_top_k: int | None = None
    column_top_k: int | None = None
    include_diagonal_when_square: bool = False
    large_cost: float = 1.0e6

    def __post_init__(self) -> None:
        if self.max_distance is not None:
            object.__setattr__(
                self,
                "max_distance",
                _finite_nonnegative_float(self.max_distance, name="max_distance"),
            )
        for name, value in {
            "row_top_k": self.row_top_k,
            "column_top_k": self.column_top_k,
        }.items():
            if value is not None:
                object.__setattr__(self, name, _positive_int(value, name=name))
        object.__setattr__(
            self,
            "include_diagonal_when_square",
            _strict_bool(
                self.include_diagonal_when_square,
                name="include_diagonal_when_square",
            ),
        )
        object.__setattr__(
            self,
            "large_cost",
            _finite_positive_float(self.large_cost, name="large_cost"),
        )


def centroid_candidate_mask(
    reference_centroids: np.ndarray,
    measurement_centroids: np.ndarray,
    *,
    config: CentroidCandidatePrefilterConfig | None = None,
) -> np.ndarray:
    """Return a boolean pairwise mask from centroid radius and top-k rules.

    Parameters
    ----------
    reference_centroids, measurement_centroids
        Arrays with shape ``(2, n_roi)`` or ``(n_roi, 2)``. The function accepts
        either common convention and normalizes internally. Ambiguous ``(2, 2)``
        arrays inherit the peer centroid layout when the peer layout is
        unambiguous; otherwise they keep the historical coordinate-row layout.
    config
        Candidate policy. ``None`` returns a dense all-true mask.
    """

    cfg = config or CentroidCandidatePrefilterConfig()
    reference = _as_point_matrix(
        reference_centroids,
        name="reference_centroids",
        peer_values=measurement_centroids,
    )
    measurement = _as_point_matrix(
        measurement_centroids,
        name="measurement_centroids",
        peer_values=reference_centroids,
    )
    distances = _pairwise_distances(reference, measurement)
    mask = np.ones(distances.shape, dtype=bool)

    if cfg.max_distance is not None:
        mask &= distances <= float(cfg.max_distance)

    if cfg.row_top_k is not None:
        mask &= _top_k_mask(distances, axis=1, top_k=cfg.row_top_k)

    if cfg.column_top_k is not None:
        mask &= _top_k_mask(distances, axis=0, top_k=cfg.column_top_k)

    if cfg.include_diagonal_when_square and mask.shape[0] == mask.shape[1]:
        np.fill_diagonal(mask, True)

    return mask


def apply_candidate_mask(
    cost_matrix: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    large_cost: float = 1.0e6,
) -> np.ndarray:
    """Return ``cost_matrix`` with non-candidates replaced by ``large_cost``."""

    costs = np.asarray(cost_matrix, dtype=float)
    mask = _as_candidate_mask(candidate_mask)
    if costs.shape != mask.shape:
        raise ValueError(
            f"candidate_mask shape {mask.shape} does not match cost matrix shape {costs.shape}"
        )
    large_cost = _finite_positive_float(large_cost, name="large_cost")
    return np.where(mask, costs, large_cost)


def candidate_edges_from_mask(
    candidate_mask: np.ndarray,
) -> tuple[tuple[int, int], ...]:
    """Return sparse candidate edge coordinates from a pairwise mask."""

    rows, columns = np.nonzero(_as_candidate_mask(candidate_mask))
    return tuple(
        (int(row), int(column)) for row, column in zip(rows, columns, strict=True)
    )


def _as_point_matrix(
    values: np.ndarray,
    *,
    name: str,
    peer_values: np.ndarray | None = None,
) -> np.ndarray:
    points = np.asarray(values, dtype=float)
    if points.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if points.shape == (2, 2):
        peer_layout = _unambiguous_centroid_layout(peer_values)
        if peer_layout == "point_rows":
            return np.ascontiguousarray(points, dtype=float)
        return np.ascontiguousarray(points.T, dtype=float)
    if points.shape[0] == 2:
        points = points.T
    elif points.shape[1] != 2:
        raise ValueError(f"{name} must have shape (2, n) or (n, 2)")
    return np.ascontiguousarray(points, dtype=float)


def _unambiguous_centroid_layout(values: np.ndarray | None) -> str | None:
    if values is None:
        return None
    points = np.asarray(values)
    if points.ndim != 2:
        return None
    if points.shape[1] == 2 and points.shape[0] != 2:
        return "point_rows"
    if points.shape[0] == 2 and points.shape[1] != 2:
        return "coordinate_rows"
    return None


def _pairwise_distances(reference: np.ndarray, measurement: np.ndarray) -> np.ndarray:
    if reference.shape[0] == 0 or measurement.shape[0] == 0:
        return np.zeros((reference.shape[0], measurement.shape[0]), dtype=float)
    deltas = reference[:, None, :] - measurement[None, :, :]
    return np.linalg.norm(deltas, axis=2)


def _top_k_mask(distances: np.ndarray, *, axis: int, top_k: int) -> np.ndarray:
    top_k = _positive_int(top_k, name="top_k")
    distances = np.asarray(distances, dtype=float)
    if distances.size == 0:
        return np.zeros(distances.shape, dtype=bool)
    if axis == 1:
        return _row_top_k_mask(distances, top_k=top_k)
    if axis == 0:
        return _row_top_k_mask(distances.T, top_k=top_k).T
    raise ValueError("axis must be 0 or 1")


def _row_top_k_mask(distances: np.ndarray, *, top_k: int) -> np.ndarray:
    mask = np.zeros(distances.shape, dtype=bool)
    if distances.shape[1] == 0:
        return mask
    k = min(_positive_int(top_k, name="top_k"), distances.shape[1])
    for row_index, row in enumerate(distances):
        finite_columns = np.flatnonzero(np.isfinite(row))
        if finite_columns.size == 0:
            continue
        if finite_columns.size <= k:
            selected = finite_columns
        else:
            finite_costs = row[finite_columns]
            selected = finite_columns[np.argpartition(finite_costs, k - 1)[:k]]
        mask[row_index, selected] = True
    return mask


def _as_candidate_mask(candidate_mask: Any) -> np.ndarray:
    mask = np.asarray(candidate_mask)
    if mask.ndim != 2:
        raise ValueError(
            "candidate_mask must be a two-dimensional boolean or binary mask"
        )
    if mask.dtype == np.dtype(bool):
        return np.ascontiguousarray(mask, dtype=bool)
    if mask.dtype.kind in {"i", "u", "f"}:
        numeric_mask = np.asarray(mask, dtype=float)
        if not _is_binary_numeric_mask(numeric_mask):
            raise ValueError(
                "candidate_mask must contain only boolean or binary values"
            )
        return np.ascontiguousarray(numeric_mask == 1.0, dtype=bool)
    if mask.dtype.kind == "O":
        normalized = np.empty(mask.shape, dtype=bool)
        for index, value in np.ndenumerate(mask):
            normalized[index] = _candidate_mask_scalar_to_bool(value)
        return np.ascontiguousarray(normalized, dtype=bool)
    raise ValueError("candidate_mask must contain only boolean or binary values")


def _is_binary_numeric_mask(mask: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(mask)) and np.all((mask == 0.0) | (mask == 1.0)))


def _candidate_mask_scalar_to_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        integer_value = int(value)
        if integer_value in {0, 1}:
            return bool(integer_value)
        raise ValueError("candidate_mask must contain only boolean or binary values")
    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        if np.isfinite(numeric_value) and numeric_value in {0.0, 1.0}:
            return bool(numeric_value)
        raise ValueError("candidate_mask must contain only boolean or binary values")
    raise ValueError("candidate_mask must contain only boolean or binary values")


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
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


def _strict_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def _finite_nonnegative_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite non-negative value")
    numeric_value = float(value)
    if not np.isfinite(numeric_value) or numeric_value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative value")
    return numeric_value


def _finite_positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite positive value")
    numeric_value = float(value)
    if not np.isfinite(numeric_value) or numeric_value <= 0.0:
        raise ValueError(f"{name} must be a finite positive value")
    return numeric_value
