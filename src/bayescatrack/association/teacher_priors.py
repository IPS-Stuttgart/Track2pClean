"""Track2p-teacher edge priors for controlled ablation experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from bayescatrack.core.bridge import Track2pSession

from ._numeric_validation import finite_nonnegative_float as _finite_nonnegative_float
from ._numeric_validation import finite_positive_float as _finite_positive_float
from ._numeric_validation import positive_integer as _positive_integer
from ._numeric_validation import validated_numeric_float as _validated_numeric_float

SessionEdge = tuple[int, int]


@dataclass(frozen=True)
class TeacherEdgePriorConfig:
    """Additive cost policy for Track2p-teacher edges."""

    relief: float = 0.5
    teacher_cost_cap: float | None = None
    non_teacher_penalty: float = 0.0
    min_cost: float = -5.0
    max_gap: int | None = None
    consecutive_only: bool = False
    large_cost: float = 1.0e6

    def __post_init__(self) -> None:
        object.__setattr__(self, "relief", _finite_nonnegative_float(self.relief, name="relief"))
        object.__setattr__(self, "non_teacher_penalty", _finite_nonnegative_float(self.non_teacher_penalty, name="non_teacher_penalty"))
        if self.teacher_cost_cap is not None:
            object.__setattr__(self, "teacher_cost_cap", _finite_nonnegative_float(self.teacher_cost_cap, name="teacher_cost_cap"))
        object.__setattr__(self, "min_cost", _validated_numeric_float(self.min_cost, name="min_cost"))
        if self.max_gap is not None:
            object.__setattr__(self, "max_gap", _positive_integer(self.max_gap, name="max_gap"))
        if not isinstance(self.consecutive_only, bool):
            raise ValueError("consecutive_only must be a boolean")
        object.__setattr__(self, "large_cost", _finite_positive_float(self.large_cost, name="large_cost"))


def teacher_edge_prior_config_from_mapping(value: TeacherEdgePriorConfig | Mapping[str, Any] | None) -> TeacherEdgePriorConfig | None:
    """Normalize optional teacher-prior configuration values."""

    if value is None:
        return None
    if isinstance(value, TeacherEdgePriorConfig):
        return value
    return TeacherEdgePriorConfig(**dict(value))


def apply_teacher_edge_priors(
    pairwise_costs: Mapping[SessionEdge, np.ndarray],
    sessions: Sequence[Track2pSession],
    *,
    teacher_track_matrix: Any,
    session_edges: Sequence[SessionEdge] | None = None,
    config: TeacherEdgePriorConfig | Mapping[str, Any] | None = None,
) -> dict[SessionEdge, np.ndarray]:
    """Return pairwise costs adjusted by Track2p-teacher edge priors."""

    cfg = teacher_edge_prior_config_from_mapping(config) or TeacherEdgePriorConfig()
    edges = tuple(session_edges) if session_edges is not None else tuple(pairwise_costs)
    teacher_masks = teacher_edge_masks_from_track_matrix(teacher_track_matrix, sessions, session_edges=edges, config=cfg)
    adjusted: dict[SessionEdge, np.ndarray] = {}
    for edge, matrix in pairwise_costs.items():
        costs = np.asarray(matrix, dtype=float).copy()
        mask = teacher_masks.get(edge)
        if mask is None or not np.any(mask):
            adjusted[edge] = costs
            continue
        if mask.shape != costs.shape:
            raise ValueError(f"Teacher mask for edge {edge!r} has shape {mask.shape}, expected {costs.shape}")
        finite = np.isfinite(costs) & (costs < float(cfg.large_cost))
        if cfg.non_teacher_penalty > 0.0:
            costs[finite & ~mask] += float(cfg.non_teacher_penalty)
        if cfg.teacher_cost_cap is not None:
            costs[mask] = np.minimum(costs[mask], float(cfg.teacher_cost_cap))
        if cfg.relief > 0.0:
            costs[mask] -= float(cfg.relief)
        costs[mask] = np.maximum(costs[mask], float(cfg.min_cost))
        adjusted[edge] = costs
    return adjusted


def teacher_edge_masks_from_track_matrix(
    teacher_track_matrix: Any,
    sessions: Sequence[Track2pSession],
    *,
    session_edges: Sequence[SessionEdge],
    config: TeacherEdgePriorConfig | Mapping[str, Any] | None = None,
) -> dict[SessionEdge, np.ndarray]:
    """Return boolean teacher-edge masks in loaded ROI coordinates."""

    cfg = teacher_edge_prior_config_from_mapping(config) or TeacherEdgePriorConfig()
    sessions = tuple(sessions)
    tracks = _normalize_track_matrix(teacher_track_matrix)
    if tracks.shape[1] != len(sessions):
        raise ValueError("teacher_track_matrix must have one column per loaded session: " f"got {tracks.shape[1]} columns for {len(sessions)} sessions")
    roi_position_by_session = tuple(_suite2p_to_loaded_position(session) for session in sessions)
    masks: dict[SessionEdge, np.ndarray] = {}
    for edge in session_edges:
        source, target = (int(edge[0]), int(edge[1]))
        if source < 0 or target <= source or target >= len(sessions):
            raise ValueError(f"Invalid teacher-prior session edge {edge!r}")
        gap = target - source
        if cfg.consecutive_only and gap != 1:
            continue
        if cfg.max_gap is not None and gap > int(cfg.max_gap):
            continue
        masks[(source, target)] = np.zeros((int(sessions[source].plane_data.n_rois), int(sessions[target].plane_data.n_rois)), dtype=bool)

    for row in tracks:
        for edge, mask in masks.items():
            source, target = edge
            roi_a = row[source]
            roi_b = row[target]
            if not _is_valid_roi_index(roi_a) or not _is_valid_roi_index(roi_b):
                continue
            source_position = roi_position_by_session[source].get(int(roi_a))
            target_position = roi_position_by_session[target].get(int(roi_b))
            if source_position is None or target_position is None:
                continue
            mask[source_position, target_position] = True
    return masks


def _suite2p_to_loaded_position(session: Track2pSession) -> dict[int, int]:
    plane = session.plane_data
    if plane.roi_indices is None:
        roi_indices = np.arange(int(plane.n_rois), dtype=int)
    else:
        roi_indices = np.asarray(plane.roi_indices, dtype=int).reshape(-1)
    return {int(suite2p_index): int(position) for position, suite2p_index in enumerate(roi_indices)}


def _normalize_track_matrix(track_matrix: Any) -> np.ndarray:
    matrix = np.asarray(track_matrix, dtype=object)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError("track matrix must be one- or two-dimensional")
    normalized = np.empty(matrix.shape, dtype=object)
    for index, value in np.ndenumerate(matrix):
        normalized[index] = _parse_roi_index(value)
    return normalized


def _parse_roi_index(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        text = value.strip()
        if text.casefold() in {"", "none", "nan", "null"}:
            return None
        try:
            numeric = float(text)
        except ValueError:
            return None
    elif isinstance(value, (int, np.integer)):
        numeric = float(int(value))
    elif isinstance(value, (float, np.floating)):
        numeric = float(value)
    else:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
    if not np.isfinite(numeric) or not numeric.is_integer():
        return None
    parsed = int(numeric)
    return parsed if parsed >= 0 else None


def _is_valid_roi_index(value: Any) -> bool:
    return _parse_roi_index(value) is not None
