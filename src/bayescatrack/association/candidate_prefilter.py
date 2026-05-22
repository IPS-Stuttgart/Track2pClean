"""Candidate prefilters for large pairwise ROI association problems."""

from __future__ import annotations

from dataclasses import dataclass

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
        if self.max_distance is not None and float(self.max_distance) < 0.0:
            raise ValueError("max_distance must be non-negative when provided")
        for name, value in {
            "row_top_k": self.row_top_k,
            "column_top_k": self.column_top_k,
        }.items():
            if value is not None and int(value) < 1:
                raise ValueError(f"{name} must be at least 1 when provided")
        if float(self.large_cost) <= 0.0:
            raise ValueError("large_cost must be strictly positive")


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
        either common convention and normalizes internally.
    config
        Candidate policy. ``None`` returns a dense all-true mask.
    """

    cfg = config or CentroidCandidatePrefilterConfig()
    reference = _as_point_matrix(reference_centroids, name="reference_centroids")
    measurement = _as_point_matrix(measurement_centroids, name="measurement_centroids")
    distances = _pairwise_distances(reference, measurement)
    mask = np.ones(distances.shape, dtype=bool)

    if cfg.max_distance is not None:
        mask &= distances <= float(cfg.max_distance)

    if cfg.row_top_k is not None:
        mask &= _top_k_mask(distances, axis=1, top_k=int(cfg.row_top_k))

    if cfg.column_top_k is not None:
        mask &= _top_k_mask(distances, axis=0, top_k=int(cfg.column_top_k))

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
    mask = np.asarray(candidate_mask, dtype=bool)
    if costs.shape != mask.shape:
        raise ValueError(
            f"candidate_mask shape {mask.shape} does not match cost matrix shape {costs.shape}"
        )
    if large_cost <= 0.0:
        raise ValueError("large_cost must be strictly positive")
    return np.where(mask, costs, float(large_cost))


def candidate_edges_from_mask(candidate_mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Return sparse candidate edge coordinates from a pairwise mask."""

    rows, columns = np.nonzero(np.asarray(candidate_mask, dtype=bool))
    return tuple((int(row), int(column)) for row, column in zip(rows, columns, strict=True))


def _as_point_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    points = np.asarray(values, dtype=float)
    if points.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if points.shape[0] == 2:
        points = points.T
    elif points.shape[1] != 2:
        raise ValueError(f"{name} must have shape (2, n) or (n, 2)")
    return np.ascontiguousarray(points, dtype=float)


def _pairwise_distances(reference: np.ndarray, measurement: np.ndarray) -> np.ndarray:
    if reference.shape[0] == 0 or measurement.shape[0] == 0:
        return np.zeros((reference.shape[0], measurement.shape[0]), dtype=float)
    deltas = reference[:, None, :] - measurement[None, :, :]
    return np.linalg.norm(deltas, axis=2)


def _top_k_mask(distances: np.ndarray, *, axis: int, top_k: int) -> np.ndarray:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
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
    k = min(int(top_k), distances.shape[1])
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
