"""Conservative post-solve relinking for geometry-flagged track detections."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from bayescatrack.association.track_refinement import TrackGeometryIssue

SessionEdge = tuple[int, int]


@dataclass(frozen=True)
class PostSolveRelinkingConfig:
    """Controls for replacing high-residual detections with lower-cost candidates."""

    max_edge_cost: float | None = 6.0
    min_cost_improvement: float = 0.25
    enforce_unique_session_rois: bool = True
    fill_value: int = -1

    def __post_init__(self) -> None:
        if self.max_edge_cost is not None and self.max_edge_cost < 0.0:
            raise ValueError("max_edge_cost must be non-negative when provided")
        if self.min_cost_improvement < 0.0:
            raise ValueError("min_cost_improvement must be non-negative")


def relink_tracks_at_geometry_issues(
    track_rows: Any,
    issues: Sequence[TrackGeometryIssue],
    pairwise_costs: Mapping[SessionEdge, np.ndarray],
    *,
    roi_indices_by_session: Sequence[Sequence[int]],
    config: PostSolveRelinkingConfig | Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Return a track matrix with local relinks for flagged high-residual detections.

    The relinker is intentionally conservative.  It only replaces the ROI at a
    flagged session if the previous present detection has a finite pairwise edge
    to that session and a lower-cost unused candidate beats the current edge by
    at least ``min_cost_improvement``.  It does not invent missing sessions or
    repair unflagged detections.
    """

    cfg = _coerce_config(config)
    rows = np.asarray(track_rows, dtype=int).copy()
    if rows.ndim != 2:
        raise ValueError("track_rows must be two-dimensional")
    if rows.shape[1] != len(roi_indices_by_session):
        raise ValueError("roi_indices_by_session must contain one vector per session")
    if not issues:
        return rows

    roi_indices = tuple(np.asarray(values, dtype=int).reshape(-1) for values in roi_indices_by_session)
    local_index_by_session = tuple(_local_index_map(values) for values in roi_indices)
    occupied = _occupied_rois_by_session(rows, fill_value=cfg.fill_value)

    for issue in sorted(issues, key=lambda item: (item.track_index, item.session_index)):
        track_index = int(issue.track_index)
        session_index = int(issue.session_index)
        if not (0 <= track_index < rows.shape[0] and 0 <= session_index < rows.shape[1]):
            continue
        previous_session = _previous_present_session(rows[track_index], session_index, fill_value=cfg.fill_value)
        if previous_session is None:
            continue
        edge = (previous_session, session_index)
        cost_matrix = pairwise_costs.get(edge)
        if cost_matrix is None:
            continue
        matrix = np.asarray(cost_matrix, dtype=float)
        source_roi = int(rows[track_index, previous_session])
        source_local = local_index_by_session[previous_session].get(source_roi)
        if source_local is None or source_local >= matrix.shape[0]:
            continue

        current_roi = int(rows[track_index, session_index])
        current_local = local_index_by_session[session_index].get(current_roi)
        current_cost = (
            float(matrix[source_local, current_local])
            if current_local is not None and current_local < matrix.shape[1]
            else float("inf")
        )

        target_local = _best_relink_candidate(
            matrix[source_local],
            roi_indices[session_index],
            occupied[session_index] - {current_roi},
            current_cost=current_cost,
            config=cfg,
        )
        if target_local is None:
            continue
        new_roi = int(roi_indices[session_index][target_local])
        rows[track_index, session_index] = new_roi
        occupied[session_index].discard(current_roi)
        occupied[session_index].add(new_roi)
    return rows


def _coerce_config(
    config: PostSolveRelinkingConfig | Mapping[str, Any] | None,
) -> PostSolveRelinkingConfig:
    if config is None:
        return PostSolveRelinkingConfig()
    if isinstance(config, PostSolveRelinkingConfig):
        return config
    return PostSolveRelinkingConfig(**dict(config))


def _local_index_map(roi_indices: np.ndarray) -> dict[int, int]:
    return {int(roi_index): int(local_index) for local_index, roi_index in enumerate(roi_indices)}


def _occupied_rois_by_session(rows: np.ndarray, *, fill_value: int) -> tuple[set[int], ...]:
    occupied: list[set[int]] = []
    for session_index in range(rows.shape[1]):
        occupied.append(
            {int(value) for value in rows[:, session_index] if int(value) != int(fill_value)}
        )
    return tuple(occupied)


def _previous_present_session(
    row: np.ndarray,
    session_index: int,
    *,
    fill_value: int,
) -> int | None:
    for candidate in range(int(session_index) - 1, -1, -1):
        if int(row[candidate]) != int(fill_value):
            return candidate
    return None


def _best_relink_candidate(
    costs: np.ndarray,
    target_roi_indices: np.ndarray,
    occupied_target_rois: set[int],
    *,
    current_cost: float,
    config: PostSolveRelinkingConfig,
) -> int | None:
    costs = np.asarray(costs, dtype=float).reshape(-1)
    target_roi_indices = np.asarray(target_roi_indices, dtype=int).reshape(-1)
    limit = min(costs.size, target_roi_indices.size)
    if limit <= 0:
        return None
    costs = costs[:limit]
    target_roi_indices = target_roi_indices[:limit]
    finite = np.isfinite(costs)
    if config.max_edge_cost is not None:
        finite &= costs <= float(config.max_edge_cost)
    if config.enforce_unique_session_rois:
        for local_index, roi_index in enumerate(target_roi_indices):
            if int(roi_index) in occupied_target_rois:
                finite[local_index] = False
    candidates = np.flatnonzero(finite)
    if candidates.size == 0:
        return None
    best_local = int(candidates[np.argmin(costs[candidates])])
    best_cost = float(costs[best_local])
    if np.isfinite(current_cost) and current_cost - best_cost < config.min_cost_improvement:
        return None
    return best_local


__all__ = ("PostSolveRelinkingConfig", "relink_tracks_at_geometry_issues")
