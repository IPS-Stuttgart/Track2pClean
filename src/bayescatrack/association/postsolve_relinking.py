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
    use_next_context: bool = True
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

    The relinker is intentionally conservative. It only replaces the ROI at a
    flagged session when finite pairwise evidence around that session supports a
    lower-cost unused candidate by at least ``min_cost_improvement``. When
    ``use_next_context`` is enabled, an interior detection is scored with both
    its previous and next present neighbours, which prevents one-sided repairs
    from creating a new high-cost bridge in the following session. It does not
    invent missing sessions or repair unflagged detections.
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

        current_roi = int(rows[track_index, session_index])
        current_local = local_index_by_session[session_index].get(current_roi)
        candidate_costs = _candidate_relink_costs(
            rows[track_index],
            session_index,
            pairwise_costs,
            local_index_by_session=local_index_by_session,
            current_local=current_local,
            config=cfg,
        )
        if candidate_costs is None:
            continue

        target_local = _best_relink_candidate(
            candidate_costs.costs,
            roi_indices[session_index],
            occupied[session_index] - {current_roi},
            current_cost=candidate_costs.current_cost,
            config=cfg,
        )
        if target_local is None:
            continue
        new_roi = int(roi_indices[session_index][target_local])
        rows[track_index, session_index] = new_roi
        occupied[session_index].discard(current_roi)
        occupied[session_index].add(new_roi)
    return rows


@dataclass(frozen=True)
class _CandidateRelinkCosts:
    costs: np.ndarray
    current_cost: float


def _candidate_relink_costs(
    row: np.ndarray,
    session_index: int,
    pairwise_costs: Mapping[SessionEdge, np.ndarray],
    *,
    local_index_by_session: Sequence[Mapping[int, int]],
    current_local: int | None,
    config: PostSolveRelinkingConfig,
) -> _CandidateRelinkCosts | None:
    previous_session = _previous_present_session(
        row, session_index, fill_value=config.fill_value
    )
    next_session = (
        _next_present_session(row, session_index, fill_value=config.fill_value)
        if config.use_next_context
        else None
    )
    directional_costs: list[np.ndarray] = []
    current_cost = 0.0
    if previous_session is not None:
        previous_costs = _previous_to_candidate_costs(
            row,
            previous_session,
            session_index,
            pairwise_costs,
            local_index_by_session=local_index_by_session,
        )
        if previous_costs is not None:
            costs, current_edge_cost = _edge_current_cost(
                previous_costs, current_local=current_local
            )
            directional_costs.append(costs)
            current_cost += current_edge_cost
    if next_session is not None:
        next_costs = _candidate_to_next_costs(
            row,
            session_index,
            next_session,
            pairwise_costs,
            local_index_by_session=local_index_by_session,
        )
        if next_costs is not None:
            costs, current_edge_cost = _edge_current_cost(
                next_costs, current_local=current_local
            )
            directional_costs.append(costs)
            current_cost += current_edge_cost
    if not directional_costs:
        return None
    combined = _sum_aligned_cost_vectors(directional_costs)
    return _CandidateRelinkCosts(combined, current_cost)


def _previous_to_candidate_costs(
    row: np.ndarray,
    previous_session: int,
    session_index: int,
    pairwise_costs: Mapping[SessionEdge, np.ndarray],
    *,
    local_index_by_session: Sequence[Mapping[int, int]],
) -> np.ndarray | None:
    edge = (previous_session, session_index)
    cost_matrix = pairwise_costs.get(edge)
    if cost_matrix is None:
        return None
    matrix = np.asarray(cost_matrix, dtype=float)
    source_roi = int(row[previous_session])
    source_local = local_index_by_session[previous_session].get(source_roi)
    if source_local is None or source_local >= matrix.shape[0]:
        return None
    return np.asarray(matrix[source_local], dtype=float).reshape(-1)


def _candidate_to_next_costs(
    row: np.ndarray,
    session_index: int,
    next_session: int,
    pairwise_costs: Mapping[SessionEdge, np.ndarray],
    *,
    local_index_by_session: Sequence[Mapping[int, int]],
) -> np.ndarray | None:
    edge = (session_index, next_session)
    cost_matrix = pairwise_costs.get(edge)
    if cost_matrix is None:
        return None
    matrix = np.asarray(cost_matrix, dtype=float)
    target_roi = int(row[next_session])
    target_local = local_index_by_session[next_session].get(target_roi)
    if target_local is None or target_local >= matrix.shape[1]:
        return None
    return np.asarray(matrix[:, target_local], dtype=float).reshape(-1)


def _edge_current_cost(
    costs: np.ndarray,
    *,
    current_local: int | None,
) -> tuple[np.ndarray, float]:
    costs = np.asarray(costs, dtype=float).reshape(-1)
    current_cost = (
        float(costs[current_local])
        if current_local is not None and current_local < costs.size
        else float("inf")
    )
    return costs, current_cost


def _sum_aligned_cost_vectors(cost_vectors: Sequence[np.ndarray]) -> np.ndarray:
    limit = min(vector.size for vector in cost_vectors)
    if limit <= 0:
        return np.zeros((0,), dtype=float)
    combined = np.zeros((limit,), dtype=float)
    for vector in cost_vectors:
        combined += np.asarray(vector[:limit], dtype=float)
    return combined


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
    for candidate in range(int(session_index) + 1, row.size):
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
    if (
        np.isfinite(current_cost)
        and current_cost - best_cost < config.min_cost_improvement
    ):
        return None
    return best_local


__all__ = ("PostSolveRelinkingConfig", "relink_tracks_at_geometry_issues")
