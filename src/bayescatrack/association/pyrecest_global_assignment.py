"""BayesCaTrack adapter for PyRecEst global multi-session assignment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np

try:  # pragma: no cover - exercised in normal benchmark environments
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - defensive fallback only
    linear_sum_assignment = None

from bayescatrack.association.activity_similarity import (
    add_activity_similarity_components,
)
from bayescatrack.association.activity_tie_breaker import (
    activity_tie_breaker_cost_matrix,
)
from bayescatrack.association.calibrated_costs import (
    CalibratedAssociationModel,
    calibrated_cost_matrix_from_bundle,
)
from bayescatrack.association.edge_thresholds import (
    EdgeThresholdPolicy,
    apply_edge_cost_thresholds,
    compute_otsu_edge_cost_thresholds,
)
from bayescatrack.association.higher_order_consistency import (
    HigherOrderConsistencyConfig,
    apply_higher_order_consistency,
)
from bayescatrack.association.registered_masks import (
    replace_empty_registered_masks,
)
from bayescatrack.core.bridge import (
    Track2pSession,
    build_session_pair_association_bundle,
)
from bayescatrack.track2p_registration import register_plane_pair

AssociationCost = Literal[
    "registered-iou",
    "registered-soft-iou",
    "registered-shifted-iou",
    "roi-aware",
    "calibrated",
    "monotone-ranked",
]
SessionEdge = tuple[int, int]


@dataclass(frozen=True)
class GlobalAssignmentRun:
    """Global assignment result plus the pairwise evidence used to build it."""

    result: Any
    pairwise_costs: dict[SessionEdge, np.ndarray]
    session_sizes: tuple[int, ...]
    session_edges: tuple[SessionEdge, ...]
    edge_cost_thresholds: dict[SessionEdge, float | None] | None = None


def registered_iou_cost_kwargs(
    *, similarity_epsilon: float = 1.0e-6
) -> dict[str, float]:
    """Return cost kwargs for a Track2p-style registered IoU ablation."""

    return {
        "centroid_weight": 0.0,
        "iou_weight": 1.0,
        "mask_cosine_weight": 0.0,
        "area_weight": 0.0,
        "roi_feature_weight": 0.0,
        "cell_probability_weight": 0.0,
        "similarity_epsilon": float(similarity_epsilon),
    }


def roi_aware_cost_kwargs() -> dict[str, float]:
    """Return the default BayesCaTrack ROI-aware cost configuration."""

    return {}


def session_edge_pairs(
    num_sessions: int, *, max_gap: int = 2
) -> tuple[SessionEdge, ...]:
    """Return forward session edges admitted by the max-gap skip-session policy."""

    if num_sessions < 0:
        raise ValueError("num_sessions must be non-negative")
    if max_gap < 1:
        raise ValueError("max_gap must be at least 1")
    return tuple(
        (source, target)
        for source in range(max(0, num_sessions - 1))
        for target in range(source + 1, min(num_sessions, source + max_gap + 1))
    )


# pylint: disable=too-many-arguments,too-many-locals
def build_registered_pairwise_costs(
    sessions: Sequence[Track2pSession],
    *,
    max_gap: int = 2,
    cost: AssociationCost = "registered-iou",
    calibrated_model: CalibratedAssociationModel | None = None,
    transform_type: str = "affine",
    order: str = "xy",
    weighted_centroids: bool = False,
    velocity_variance: float = 25.0,
    regularization: float = 1.0e-6,
    pairwise_cost_kwargs: Mapping[str, Any] | None = None,
    return_pairwise_components: bool = False,
    activity_tie_breaker_weight: float = 0.0,
    activity_tie_breaker_component: str = "activity_tiebreaker_cost",
    activity_trace_source: str = "auto",
    activity_event_threshold: float = 0.0,
) -> dict[SessionEdge, np.ndarray]:
    """Build registered pairwise cost matrices for consecutive and skip-session edges."""

    sessions = list(sessions)
    if cost == "calibrated" and calibrated_model is None:
        raise ValueError("calibrated_model is required when cost='calibrated'")
    if activity_tie_breaker_weight < 0.0:
        raise ValueError("activity_tie_breaker_weight must be non-negative")

    needs_activity_components = (
        return_pairwise_components
        or cost == "calibrated"
        or activity_tie_breaker_weight > 0.0
    )

    base_cost_kwargs = _cost_kwargs_for_method(cost)
    if pairwise_cost_kwargs is not None:
        base_cost_kwargs.update(dict(pairwise_cost_kwargs))

    pairwise_costs: dict[SessionEdge, np.ndarray] = {}
    for source_session, target_session in session_edge_pairs(
        len(sessions), max_gap=max_gap
    ):
        registered_measurement_plane = register_plane_pair(
            sessions[source_session].plane_data,
            sessions[target_session].plane_data,
            transform_type=transform_type,
        )
        registered_measurement_plane, empty_registered_rois = (
            replace_empty_registered_masks(registered_measurement_plane)
        )
        bundle = build_session_pair_association_bundle(
            sessions[source_session],
            sessions[target_session],
            measurement_plane_in_reference_frame=registered_measurement_plane,
            order=order,
            weighted_centroids=weighted_centroids,
            velocity_variance=velocity_variance,
            regularization=regularization,
            pairwise_cost_kwargs=base_cost_kwargs,
            return_pairwise_components=needs_activity_components,
        )
        if needs_activity_components:
            add_activity_similarity_components(
                bundle.pairwise_components,
                sessions[source_session].plane_data,
                registered_measurement_plane,
                trace_source=activity_trace_source,
                event_threshold=activity_event_threshold,
            )
        if cost == "calibrated":
            assert calibrated_model is not None
            cost_matrix = calibrated_cost_matrix_from_bundle(
                bundle,
                calibrated_model,
                session_gap=target_session - source_session,
            )
        else:
            cost_matrix = np.asarray(bundle.pairwise_cost_matrix, dtype=float)
        if activity_tie_breaker_weight > 0.0:
            cost_matrix = np.asarray(
                cost_matrix, dtype=float
            ) + activity_tie_breaker_cost_matrix(
                bundle.pairwise_components,
                component_name=activity_tie_breaker_component,
                weight=activity_tie_breaker_weight,
            )
        pairwise_costs[(source_session, target_session)] = (
            _penalize_empty_registered_roi_columns(
                cost_matrix,
                empty_registered_rois,
                large_cost=float(base_cost_kwargs.get("large_cost", 1.0e6)),
            )
        )
    return pairwise_costs


# pylint: disable=too-many-arguments,too-many-locals
def solve_global_assignment_for_sessions(
    sessions: Sequence[Track2pSession],
    *,
    max_gap: int = 2,
    cost: AssociationCost = "registered-iou",
    calibrated_model: CalibratedAssociationModel | None = None,
    transform_type: str = "affine",
    start_cost: float = 5.0,
    end_cost: float = 5.0,
    gap_penalty: float = 1.0,
    cost_threshold: float | None = 6.0,
    order: str = "xy",
    weighted_centroids: bool = False,
    velocity_variance: float = 25.0,
    regularization: float = 1.0e-6,
    pairwise_cost_kwargs: Mapping[str, Any] | None = None,
    activity_tie_breaker_weight: float = 0.0,
    activity_tie_breaker_component: str = "activity_tiebreaker_cost",
    activity_trace_source: str = "auto",
    activity_event_threshold: float = 0.0,
    higher_order_consistency_config: (
        HigherOrderConsistencyConfig | Mapping[str, Any] | None
    ) = None,
    edge_threshold_policy: EdgeThresholdPolicy = "none",
    edge_threshold_otsu_bins: int = 256,
    edge_threshold_otsu_max_cost: float | None = None,
) -> GlobalAssignmentRun:
    """Run PyRecEst's global path-cover assignment on registered BayesCaTrack costs."""

    sessions = list(sessions)
    pairwise_costs = build_registered_pairwise_costs(
        sessions,
        max_gap=max_gap,
        cost=cost,
        calibrated_model=calibrated_model,
        transform_type=transform_type,
        order=order,
        weighted_centroids=weighted_centroids,
        velocity_variance=velocity_variance,
        regularization=regularization,
        pairwise_cost_kwargs=pairwise_cost_kwargs,
        activity_tie_breaker_weight=activity_tie_breaker_weight,
        activity_tie_breaker_component=activity_tie_breaker_component,
        activity_trace_source=activity_trace_source,
        activity_event_threshold=activity_event_threshold,
    )
    session_sizes = tuple(int(session.plane_data.n_rois) for session in sessions)
    if higher_order_consistency_config is not None:
        pairwise_costs = apply_higher_order_consistency(
            pairwise_costs,
            session_sizes=session_sizes,
            config=higher_order_consistency_config,
        )
    edge_cost_thresholds: dict[SessionEdge, float | None] | None = None
    if edge_threshold_policy == "otsu":
        edge_cost_thresholds = compute_otsu_edge_cost_thresholds(
            pairwise_costs,
            bins=edge_threshold_otsu_bins,
            max_cost=edge_threshold_otsu_max_cost,
        )
    elif edge_threshold_policy == "manual-oracle":
        raise ValueError(
            "edge_threshold_policy='manual-oracle' requires benchmark ground truth; "
            "use bayescatrack.experiments.track2p_benchmark"
        )
    elif edge_threshold_policy != "none":
        raise ValueError(f"Unsupported edge_threshold_policy: {edge_threshold_policy!r}")
    session_edges = session_edge_pairs(len(sessions), max_gap=max_gap)
    return solve_global_assignment_from_pairwise_costs(
        pairwise_costs,
        session_sizes=session_sizes,
        session_edges=session_edges,
        start_cost=start_cost,
        end_cost=end_cost,
        gap_penalty=gap_penalty,
        cost_threshold=cost_threshold,
        edge_cost_thresholds=edge_cost_thresholds,
    )


def solve_global_assignment_from_pairwise_costs(
    pairwise_costs: Mapping[SessionEdge, np.ndarray],
    *,
    session_sizes: Sequence[int],
    session_edges: Sequence[SessionEdge] | None = None,
    start_cost: float = 5.0,
    end_cost: float = 5.0,
    gap_penalty: float = 1.0,
    cost_threshold: float | None = 6.0,
    edge_cost_thresholds: Mapping[SessionEdge, float | None] | None = None,
) -> GlobalAssignmentRun:
    """Run PyRecEst's global assignment using already-built pairwise costs.

    This is useful for fold-internal solver-prior tuning: registration and
    pairwise cost construction are expensive, while start/end/gap/threshold
    sweeps only require re-running the path-cover solver on the same cost
    matrices.
    """

    costs = dict(pairwise_costs)
    sizes = tuple(int(size) for size in session_sizes)
    edges = tuple(session_edges) if session_edges is not None else tuple(sorted(costs))
    threshold_map = (
        dict(edge_cost_thresholds) if edge_cost_thresholds is not None else None
    )
    if threshold_map is not None:
        costs = apply_edge_cost_thresholds(costs, threshold_map)

    result = _load_pyrecest_multisession_solver()(
        costs,
        session_sizes=sizes,
        start_cost=float(start_cost),
        end_cost=float(end_cost),
        gap_penalty=float(gap_penalty),
        cost_threshold=None if cost_threshold is None else float(cost_threshold),
    )
    return GlobalAssignmentRun(
        result=result,
        pairwise_costs=costs,
        session_sizes=sizes,
        session_edges=edges,
        edge_cost_thresholds=threshold_map,
    )


# pylint: disable=too-many-arguments
def solve_track2p_style_propagation_for_sessions(
    sessions: Sequence[Track2pSession],
    *,
    cost: AssociationCost = "registered-iou",
    calibrated_model: CalibratedAssociationModel | None = None,
    transform_type: str = "affine",
    cost_threshold: float | None = 6.0,
    order: str = "xy",
    weighted_centroids: bool = False,
    velocity_variance: float = 25.0,
    regularization: float = 1.0e-6,
    pairwise_cost_kwargs: Mapping[str, Any] | None = None,
    seed_session: int = 0,
    seed_detection_indices: Sequence[int] | None = None,
) -> GlobalAssignmentRun:
    """Run a seed-restricted Track2p-style propagation baseline.

    This deliberately separates pairwise evidence quality from the fully global
    path-cover objective.  It builds only consecutive registered cost matrices,
    solves one Hungarian assignment per consecutive edge, gates each edge by
    ``cost_threshold``, and then grows one row per seed ROI without allowing
    later-session births to compete with the evaluated seed population.

    The returned tracks use loaded detection indices, matching PyRecEst's track
    convention.  Convert them to Suite2p row indices with
    :func:`tracks_to_suite2p_index_matrix` before comparing against Track2p or
    manual ground-truth rows.
    """

    sessions = list(sessions)
    pairwise_costs = build_registered_pairwise_costs(
        sessions,
        max_gap=1,
        cost=cost,
        calibrated_model=calibrated_model,
        transform_type=transform_type,
        order=order,
        weighted_centroids=weighted_centroids,
        velocity_variance=velocity_variance,
        regularization=regularization,
        pairwise_cost_kwargs=pairwise_cost_kwargs,
    )
    return solve_track2p_style_propagation_from_pairwise_costs(
        pairwise_costs,
        session_sizes=tuple(int(session.plane_data.n_rois) for session in sessions),
        seed_session=seed_session,
        seed_detection_indices=seed_detection_indices,
        cost_threshold=cost_threshold,
    )


def solve_track2p_style_propagation_from_pairwise_costs(
    pairwise_costs: Mapping[SessionEdge, np.ndarray],
    *,
    session_sizes: Sequence[int],
    seed_session: int = 0,
    seed_detection_indices: Sequence[int] | None = None,
    cost_threshold: float | None = 6.0,
) -> GlobalAssignmentRun:
    """Solve consecutive edge assignments and propagate only seed rows.

    Unlike :func:`solve_global_assignment_from_pairwise_costs`, this helper does
    not optimize starts, ends, skip links, or births.  It is intended as a
    diagnostic row that mirrors the Track2p matching pattern: solve each
    consecutive pair independently, threshold links, then propagate identities
    from a chosen seed session until a gated link is missing.
    """

    costs = dict(pairwise_costs)
    sizes = tuple(int(size) for size in session_sizes)
    if not sizes:
        result = SimpleNamespace(tracks=tuple(), matched_edges=tuple(), total_cost=0.0)
        return GlobalAssignmentRun(
            result=result,
            pairwise_costs=costs,
            session_sizes=sizes,
            session_edges=tuple(),
        )

    seed_session = _validate_seed_session(seed_session, n_sessions=len(sizes))
    seed_detection_indices = _normalize_seed_detection_indices(
        seed_detection_indices,
        n_detections=sizes[seed_session],
    )
    max_cost = _coerce_track2p_style_max_cost(cost_threshold)
    consecutive_edges = tuple((index, index + 1) for index in range(len(sizes) - 1))

    forward_matches: dict[SessionEdge, dict[int, tuple[int, float]]] = {}
    matched_edges: list[tuple[int, int, int, int, float]] = []
    for edge in consecutive_edges:
        if edge not in costs:
            raise KeyError(
                f"Missing consecutive cost matrix for edge {edge}; "
                "Track2p-style propagation requires all adjacent session pairs"
            )
        source_session, target_session = edge
        cost_matrix = np.asarray(costs[edge], dtype=float)
        _validate_pairwise_cost_shape(
            cost_matrix,
            edge=edge,
            session_sizes=sizes,
        )
        source_positions, target_positions, edge_costs = (
            _solve_cost_matrix_linear_assignment(cost_matrix, max_cost=max_cost)
        )
        edge_matches: dict[int, tuple[int, float]] = {}
        for source_position, target_position, edge_cost in zip(
            source_positions,
            target_positions,
            edge_costs,
            strict=True,
        ):
            source_detection = int(source_position)
            target_detection = int(target_position)
            cost_value = float(edge_cost)
            edge_matches[source_detection] = (target_detection, cost_value)
            matched_edges.append(
                (
                    int(source_session),
                    source_detection,
                    int(target_session),
                    target_detection,
                    cost_value,
                )
            )
        forward_matches[edge] = edge_matches

    tracks, total_cost = _build_seed_restricted_tracks_from_matches(
        forward_matches,
        session_sizes=sizes,
        seed_session=seed_session,
        seed_detection_indices=seed_detection_indices,
    )
    result = SimpleNamespace(
        tracks=tracks,
        matched_edges=tuple(matched_edges),
        total_cost=float(total_cost),
    )
    return GlobalAssignmentRun(
        result=result,
        pairwise_costs=costs,
        session_sizes=sizes,
        session_edges=consecutive_edges,
    )


def _validate_seed_session(seed_session: int, *, n_sessions: int) -> int:
    seed_session = int(seed_session)
    if seed_session < 0 or seed_session >= n_sessions:
        raise IndexError(
            f"seed_session {seed_session} out of bounds for {n_sessions} sessions"
        )
    return seed_session


def _normalize_seed_detection_indices(
    seed_detection_indices: Sequence[int] | None,
    *,
    n_detections: int,
) -> tuple[int, ...]:
    if n_detections < 0:
        raise ValueError("session sizes must be non-negative")
    if seed_detection_indices is None:
        return tuple(range(n_detections))

    normalized: list[int] = []
    seen: set[int] = set()
    for raw_index in seed_detection_indices:
        try:
            detection_index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise TypeError("seed_detection_indices must contain integers") from exc
        if detection_index < 0 or detection_index >= n_detections:
            raise IndexError(
                f"seed detection index {detection_index} out of bounds for "
                f"seed session with {n_detections} detections"
            )
        if detection_index not in seen:
            normalized.append(detection_index)
            seen.add(detection_index)
    return tuple(normalized)


def _coerce_track2p_style_max_cost(max_cost: float | None) -> float | None:
    if max_cost is None:
        return None
    max_cost = float(max_cost)
    if not np.isfinite(max_cost) or max_cost < 0.0:
        raise ValueError("cost_threshold must be a finite non-negative value or None")
    return max_cost


def _validate_pairwise_cost_shape(
    cost_matrix: np.ndarray,
    *,
    edge: SessionEdge,
    session_sizes: Sequence[int],
) -> None:
    source_session, target_session = edge
    expected_shape = (
        int(session_sizes[source_session]),
        int(session_sizes[target_session]),
    )
    if cost_matrix.shape != expected_shape:
        raise ValueError(
            f"Cost matrix for edge {edge} has shape {cost_matrix.shape}, "
            f"expected {expected_shape} from session_sizes"
        )


def _solve_cost_matrix_linear_assignment(
    cost_matrix: np.ndarray,
    *,
    max_cost: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if linear_sum_assignment is None:
        raise ImportError(
            "Track2p-style propagation requires scipy.optimize.linear_sum_assignment"
        )
    if cost_matrix.ndim != 2:
        raise ValueError("cost_matrix must be two-dimensional")
    if cost_matrix.shape[0] == 0 or cost_matrix.shape[1] == 0:
        empty_indices = np.zeros((0,), dtype=int)
        empty_costs = np.zeros((0,), dtype=float)
        return empty_indices, empty_indices, empty_costs

    valid_assignment_mask = np.isfinite(cost_matrix)
    if max_cost is not None:
        valid_assignment_mask &= cost_matrix <= max_cost
    if not np.any(valid_assignment_mask):
        empty_indices = np.zeros((0,), dtype=int)
        empty_costs = np.zeros((0,), dtype=float)
        return empty_indices, empty_indices, empty_costs

    assignment_cost_matrix = _gate_cost_matrix_for_track2p_style_assignment(
        cost_matrix,
        valid_assignment_mask,
        max_cost=max_cost,
    )
    source_positions, target_positions = linear_sum_assignment(assignment_cost_matrix)
    keep = valid_assignment_mask[source_positions, target_positions]
    source_positions = np.asarray(source_positions[keep], dtype=int)
    target_positions = np.asarray(target_positions[keep], dtype=int)
    assignment_costs = np.asarray(
        cost_matrix[source_positions, target_positions],
        dtype=float,
    )
    return source_positions, target_positions, assignment_costs


def _gate_cost_matrix_for_track2p_style_assignment(
    cost_matrix: np.ndarray,
    valid_assignment_mask: np.ndarray,
    *,
    max_cost: float | None,
) -> np.ndarray:
    valid_costs = np.asarray(cost_matrix[valid_assignment_mask], dtype=float)
    if valid_costs.size == 0:
        raise ValueError("valid_assignment_mask must contain at least one True entry")

    valid_min = float(np.min(valid_costs))
    valid_max = float(np.max(valid_costs))
    cost_span = max(valid_max - valid_min, 1.0)
    threshold_scale = valid_max if max_cost is None else float(max_cost)
    cost_scale = max(
        abs(valid_min),
        abs(valid_max),
        abs(threshold_scale),
        cost_span,
        1.0,
    )
    max_assignments = min(cost_matrix.shape)
    invalid_penalty = (max_assignments + 1) * (cost_scale + cost_span + 1.0)
    return np.where(valid_assignment_mask, cost_matrix, invalid_penalty)


def _build_seed_restricted_tracks_from_matches(
    forward_matches: Mapping[SessionEdge, Mapping[int, tuple[int, float]]],
    *,
    session_sizes: Sequence[int],
    seed_session: int,
    seed_detection_indices: Sequence[int],
) -> tuple[tuple[dict[int, int], ...], float]:
    n_sessions = len(session_sizes)
    reverse_matches: dict[SessionEdge, dict[int, tuple[int, float]]] = {
        edge: {
            int(target_detection): (int(source_detection), float(edge_cost))
            for source_detection, (target_detection, edge_cost) in mapping.items()
        }
        for edge, mapping in forward_matches.items()
    }

    tracks: list[dict[int, int]] = []
    total_cost = 0.0
    for seed_detection in seed_detection_indices:
        track: dict[int, int] = {int(seed_session): int(seed_detection)}
        path_cost = 0.0

        current_detection = int(seed_detection)
        for source_session in range(seed_session, n_sessions - 1):
            edge = (source_session, source_session + 1)
            next_step = forward_matches.get(edge, {}).get(current_detection)
            if next_step is None:
                break
            next_detection, edge_cost = next_step
            track[source_session + 1] = int(next_detection)
            path_cost += float(edge_cost)
            current_detection = int(next_detection)

        current_detection = int(seed_detection)
        for source_session in range(seed_session - 1, -1, -1):
            edge = (source_session, source_session + 1)
            previous_step = reverse_matches.get(edge, {}).get(current_detection)
            if previous_step is None:
                break
            previous_detection, edge_cost = previous_step
            track[source_session] = int(previous_detection)
            path_cost += float(edge_cost)
            current_detection = int(previous_detection)

        tracks.append(track)
        total_cost += path_cost
    return tuple(tracks), float(total_cost)


def tracks_to_suite2p_index_matrix(
    tracks: Sequence[Mapping[int, int]], sessions: Sequence[Track2pSession]
) -> np.ndarray:
    """Convert solver tracks in loaded-ROI coordinates to original Suite2p indices."""
    sessions = list(sessions)
    detection_matrix = np.asarray(
        _load_pyrecest_tracks_to_index_matrix()(
            list(tracks),
            session_sizes=tuple(int(session.plane_data.n_rois) for session in sessions),
            fill_value=-1,
        ),
        dtype=int,
    )

    matrix = np.empty(detection_matrix.shape, dtype=object)
    matrix[:] = None
    roi_indices_by_session = [_roi_indices_for_session(session) for session in sessions]

    for session_index, roi_indices in enumerate(roi_indices_by_session):
        detection_indices = detection_matrix[:, session_index]
        present = detection_indices >= 0
        if not np.any(present):
            continue

        present_detection_indices = detection_indices[present]
        invalid = present_detection_indices >= roi_indices.shape[0]
        if np.any(invalid):
            raise IndexError(
                f"detection index {int(present_detection_indices[invalid][0])} "
                f"out of bounds for session {session_index}"
            )
        matrix[present, session_index] = [
            int(value) for value in roi_indices[present_detection_indices]
        ]
    return matrix


def _cost_kwargs_for_method(cost: AssociationCost) -> dict[str, Any]:
    if cost == "registered-iou":
        return registered_iou_cost_kwargs()
    if cost in {"roi-aware", "calibrated"}:
        return roi_aware_cost_kwargs()
    if cost == "monotone-ranked":
        raise ValueError(
            "cost='monotone-ranked' is a LOSO training mode. Fit a monotone "
            "ranker and pass it back as calibrated_model with cost='calibrated'."
        )
    raise ValueError(f"Unsupported association cost: {cost}")


def _penalize_empty_registered_roi_columns(
    cost_matrix: np.ndarray, empty_registered_rois: np.ndarray, *, large_cost: float
) -> np.ndarray:
    cost_matrix = np.asarray(cost_matrix, dtype=float).copy()
    if empty_registered_rois.shape != (cost_matrix.shape[1],):
        raise ValueError(
            "empty_registered_rois must have one entry for each measurement ROI"
        )
    cost_matrix[:, empty_registered_rois] = large_cost
    return cost_matrix


def _roi_indices_for_session(session: Track2pSession) -> np.ndarray:
    plane = session.plane_data
    if plane.roi_indices is not None:
        return np.asarray(plane.roi_indices, dtype=int)
    return np.arange(plane.n_rois, dtype=int)


def _load_pyrecest_tracks_to_index_matrix() -> Any:
    try:
        from pyrecest.utils import tracks_to_index_matrix
    except ImportError:
        try:
            from pyrecest.utils.multisession_assignment_score import (
                tracks_to_index_matrix,
            )
        except ImportError:
            return _local_tracks_to_index_matrix
    return tracks_to_index_matrix


def _local_tracks_to_index_matrix(
    tracks: Sequence[Mapping[int, int]],
    session_sizes: Sequence[int] | None = None,
    *,
    fill_value: int = -1,
) -> np.ndarray:
    """Small fallback matching PyRecEst's dense track matrix convention."""

    max_session = max(
        (int(session_index) for track in tracks for session_index in track),
        default=-1,
    )
    n_sessions = max(max_session + 1, len(session_sizes or ()))
    matrix = np.full((len(tracks), n_sessions), fill_value, dtype=int)
    for track_index, track in enumerate(tracks):
        for session_index, detection_index in track.items():
            matrix[track_index, int(session_index)] = int(detection_index)
    return matrix


def _load_pyrecest_multisession_solver() -> Any:
    try:
        from pyrecest.utils import solve_multisession_assignment
    except ImportError:
        try:
            from pyrecest.utils.multisession_assignment import (
                solve_multisession_assignment,
            )
        except (
            ImportError
        ) as exc:  # pragma: no cover - exercised in runtime environments without PyRecEst
            raise ImportError(
                "PyRecEst with pyrecest.utils.solve_multisession_assignment is required "
                "for global-assignment benchmarks."
            ) from exc
    return solve_multisession_assignment
