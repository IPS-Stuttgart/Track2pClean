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
    bidirectional_next_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.max_edge_cost is not None and self.max_edge_cost < 0.0:
            raise ValueError("max_edge_cost must be non-negative when provided")
        if self.min_cost_improvement < 0.0:
            raise ValueError("min_cost_improvement must be non-negative")
        next_weight = float(self.bidirectional_next_weight)
        if not np.isfinite(next_weight) or next_weight < 0.0:
            raise ValueError("bidirectional_next_weight must be finite and non-negative")
        object.__setattr__(self, "bidirectional_next_weight", next_weight)


def relink_tracks_at_geometry_issues(
    track_rows: Any,
    issues: Sequence[TrackGeometryIssue],
    pairwise_costs: Mapping[SessionEdge, np.ndarray],
    *,
    roi_indices_by_session: Sequence[Sequence[int]],
    config: PostSolveRelinkingConfig | Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Return a track matrix with local relinks for flagged high-residual detections.

    The relinker is intentionally conservative. It only replaces the ROI at a
    flagged session if the previous present detection has a finite pairwise edge
    to that session and a lower-cost unused candidate beats the current edge by
    at least ``min_cost_improvement``. When the next present detection is also
    available, candidates are ranked by incoming plus outgoing edge evidence so
    a local repair remains compatible with the full track. It does not invent
    missing sessions or repair unflagged detections.
    """

    cfg = _coerce_config(config)
    rows = np.asarray(track_rows, dtype=int).copy()
    if rows.ndim != 2:
        raise ValueError("track_rows must be two-dimensional")
    if rows.shape[1] != len(roi_indices_by_session):
        raise ValueError("roi_indices_by_session must contain one vector per session")
    if not issues:
        return rows

    roi_indices = tuple(
        np.asarray(values, dtype=int).reshape(-1) for values in roi_indices_by_session
    )
    local_index_by_session = tuple(_local_index_map(values) for values in roi_indices)
    occupied = _occupied_rois_by_session(rows, fill_value=cfg.fill_value)

    for issue in sorted(
        issues, key=lambda item: (item.track_index, item.session_index)
    ):
        track_index = int(issue.track_index)
        session_index = int(issue.session_index)
        if not (
            0 <= track_index < rows.shape[0] and 0 <= session_index < rows.shape[1]
        ):
            continue
        previous_session = _previous_present_session(
            rows[track_index], session_index, fill_value=cfg.fill_value
        )
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
        next_costs, current_next_cost = _outgoing_relink_costs(
            rows[track_index],
            session_index,
            current_local,
            pairwise_costs,
            roi_indices,
            local_index_by_session,
            config=cfg,
        )

        target_local = _best_relink_candidate(
            matrix[source_local],
            roi_indices[session_index],
            occupied[session_index] - {current_roi},
            current_cost=current_cost,
            config=cfg,
            next_costs=next_costs,
            current_next_cost=current_next_cost,
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
    return {
        int(roi_index): int(local_index)
        for local_index, roi_index in enumerate(roi_indices)
    }


def _occupied_rois_by_session(
    rows: np.ndarray, *, fill_value: int
) -> tuple[set[int], ...]:
    occupied: list[set[int]] = []
    for session_index in range(rows.shape[1]):
        occupied.append(
            {
                int(value)
                for value in rows[:, session_index]
                if int(value) != int(fill_value)
            }
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


def _next_present_session(
    row: np.ndarray,
    session_index: int,
    *,
    fill_value: int,
) -> int | None:
    for candidate in range(int(session_index) + 1, row.shape[0]):
        if int(row[candidate]) != int(fill_value):
            return candidate
    return None


def _outgoing_relink_costs(
    row: np.ndarray,
    session_index: int,
    current_local: int | None,
    pairwise_costs: Mapping[SessionEdge, np.ndarray],
    roi_indices: Sequence[np.ndarray],
    local_index_by_session: Sequence[Mapping[int, int]],
    *,
    config: PostSolveRelinkingConfig,
) -> tuple[np.ndarray | None, float | None]:
    if config.bidirectional_next_weight <= 0.0:
        return None, None
    next_session = _next_present_session(
        row, session_index, fill_value=config.fill_value
    )
    if next_session is None:
        return None, None
    next_matrix = pairwise_costs.get((session_index, next_session))
    if next_matrix is None:
        return None, None
    matrix = np.asarray(next_matrix, dtype=float)
    if matrix.ndim != 2:
        return None, None
    next_roi = int(row[next_session])
    next_local = local_index_by_session[next_session].get(next_roi)
    if next_local is None or next_local >= matrix.shape[1]:
        return None, None
    n_candidates = len(roi_indices[session_index])
    outgoing = np.full(n_candidates, float("inf"), dtype=float)
    row_limit = min(n_candidates, matrix.shape[0])
    if row_limit > 0:
        outgoing[:row_limit] = matrix[:row_limit, next_local]
    current_next_cost = (
        float(outgoing[current_local])
        if current_local is not None and current_local < outgoing.shape[0]
        else float("inf")
    )
    return outgoing, current_next_cost


def _best_relink_candidate(
    costs: np.ndarray,
    target_roi_indices: np.ndarray,
    occupied_target_rois: set[int],
    *,
    current_cost: float,
    config: PostSolveRelinkingConfig,
    next_costs: np.ndarray | None = None,
    current_next_cost: float | None = None,
) -> int | None:
    costs = np.asarray(costs, dtype=float).reshape(-1)
    target_roi_indices = np.asarray(target_roi_indices, dtype=int).reshape(-1)
    limit = min(costs.size, target_roi_indices.size)
    if limit <= 0:
        return None
    costs = costs[:limit]
    target_roi_indices = target_roi_indices[:limit]
    candidate_scores = costs.copy()
    current_score = float(current_cost)
    finite = np.isfinite(costs)
    if config.max_edge_cost is not None:
        finite &= costs <= float(config.max_edge_cost)
    if next_costs is not None and config.bidirectional_next_weight > 0.0:
        next_costs = np.asarray(next_costs, dtype=float).reshape(-1)
        padded_next_costs = np.full(limit, float("inf"), dtype=float)
        padded_next_costs[: min(limit, next_costs.size)] = next_costs[:limit]
        finite &= np.isfinite(padded_next_costs)
        if config.max_edge_cost is not None:
            finite &= padded_next_costs <= float(config.max_edge_cost)
        next_weight = float(config.bidirectional_next_weight)
        candidate_scores = costs + next_weight * padded_next_costs
        current_score = _joint_current_cost(
            current_cost,
            current_next_cost,
            next_weight=next_weight,
        )
    if config.enforce_unique_session_rois:
        for local_index, roi_index in enumerate(target_roi_indices):
            if int(roi_index) in occupied_target_rois:
                finite[local_index] = False
    candidates = np.flatnonzero(finite)
    if candidates.size == 0:
        return None
    best_local = int(candidates[np.argmin(candidate_scores[candidates])])
    best_cost = float(candidate_scores[best_local])
    if (
        np.isfinite(current_score)
        and current_score - best_cost < config.min_cost_improvement
    ):
        return None
    return best_local


def _joint_current_cost(
    current_cost: float,
    current_next_cost: float | None,
    *,
    next_weight: float,
) -> float:
    if current_next_cost is None:
        return float(current_cost)
    if not np.isfinite(current_next_cost):
        return float("inf")
    return float(current_cost) + next_weight * float(current_next_cost)


__all__ = ("PostSolveRelinkingConfig", "relink_tracks_at_geometry_issues")
