"""Track2p-policy dynamic-programming rescue benchmark.

This variant keeps the strong Track2p-policy defaults, but replaces brittle
first-session greedy propagation with a small deterministic beam search.  The
candidate graph contains threshold-minimum accepted Hungarian links, top-k local
rescue candidates, and optional one-gap repair edges.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from bayescatrack.core.bridge import Track2pSession
from bayescatrack.experiments.track2p_benchmark import (
    OutputFormat,
    SubjectBenchmarkResult,
    Track2pBenchmarkConfig,
    _load_reference_for_subject,
    _load_subject_sessions,
    _score_prediction_against_reference,
    _validate_reference_for_benchmark,
    _validate_reference_roi_indices,
    discover_subject_dirs,
    write_results,
)
from bayescatrack.experiments.track2p_emulation_benchmark import (
    ThresholdMethod,
    _roi_indices,
    _threshold_assigned_iou,
    _track2p_cross_iou_matrix,
)
from bayescatrack.experiments.track2p_policy_benchmark import (
    TRACK2P_POLICY_DEFAULT_CELL_PROBABILITY_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE,
    track2p_policy_config,
)
from bayescatrack.track2p_registration import register_plane_pair
from scipy.optimize import linear_sum_assignment

TRACK2P_POLICY_DP_METHOD = "track2p-policy-dp"


@dataclass(frozen=True)
class Track2pPolicyDPConfig:
    """Configuration for the small Track2p-policy DP/beam rescue."""

    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD
    iou_distance_threshold: float = TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD
    row_top_k: int = 2
    rescue_min_iou: float = 0.10
    threshold_rescue_margin: float = 0.15
    accepted_bonus: float = 0.25
    rescue_penalty: float = 0.25
    gap_penalty: float = 1.0
    threshold_margin_weight: float = 0.5
    beam_width: int = 8
    max_gap: int = 2
    path_candidates_per_seed: int = 4
    path_selection_beam_width: int = 256
    logit_epsilon: float = 1.0e-3
    fill_value: int = -1

    def __post_init__(self) -> None:
        if self.threshold_method not in {"otsu", "min"}:
            raise ValueError("threshold_method must be 'otsu' or 'min'")
        if self.iou_distance_threshold < 0.0:
            raise ValueError("iou_distance_threshold must be non-negative")
        if self.row_top_k <= 0:
            raise ValueError("row_top_k must be positive")
        if not 0.0 <= self.rescue_min_iou <= 1.0:
            raise ValueError("rescue_min_iou must lie in [0, 1]")
        if self.threshold_rescue_margin < 0.0:
            raise ValueError("threshold_rescue_margin must be non-negative")
        if self.beam_width <= 0:
            raise ValueError("beam_width must be positive")
        if self.max_gap < 1:
            raise ValueError("max_gap must be at least 1")
        if self.path_candidates_per_seed <= 0:
            raise ValueError("path_candidates_per_seed must be positive")
        if self.path_selection_beam_width <= 0:
            raise ValueError("path_selection_beam_width must be positive")
        if not 0.0 < self.logit_epsilon < 0.5:
            raise ValueError("logit_epsilon must lie in (0, 0.5)")


@dataclass(frozen=True)
class PolicyCandidate:
    """One Track2p-policy edge candidate in local ROI index space."""

    source_session: int
    target_session: int
    source_roi: int
    target_roi: int
    score: float
    iou: float
    threshold: float
    accepted_by_threshold: bool
    selected_by_hungarian: bool

    @property
    def session_gap(self) -> int:
        return int(self.target_session - self.source_session)


@dataclass(frozen=True)
class TrackPath:
    """One local-index track path and its accumulated score."""

    row: tuple[int, ...]
    score: float


@dataclass(frozen=True)
class _BeamState:
    row: tuple[int, ...]
    current_session: int
    current_roi: int
    score: float

    def as_path(self) -> TrackPath:
        return TrackPath(row=self.row, score=float(self.score))


# pylint: disable=too-many-arguments,too-many-locals
def track2p_policy_dp_tracks(
    sessions: Sequence[Track2pSession],
    *,
    transform_type: str = TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE,
    config: Track2pPolicyDPConfig | None = None,
) -> np.ndarray:
    """Return Suite2p-indexed tracks from the DP-rescued Track2p policy."""

    session_list = list(sessions)
    cfg = config or Track2pPolicyDPConfig()
    if not session_list:
        return np.zeros((0, 0), dtype=int)
    if len(session_list) == 1:
        return _roi_indices(session_list[0]).reshape(-1, 1)

    candidates = build_policy_candidate_edges(
        session_list,
        transform_type=transform_type,
        config=cfg,
    )
    by_source = _candidates_by_source(candidates)
    start_rois = sorted(
        candidate.source_roi
        for candidate in candidates.get((0, 1), ())
        if candidate.accepted_by_threshold
    )
    if cfg.max_gap > 1:
        start_rois = sorted(
            set(start_rois)
            | {
                candidate.source_roi
                for edge, edge_candidates in candidates.items()
                if edge[0] == 0
                for candidate in edge_candidates
            }
        )
    if not start_rois:
        return np.zeros((0, len(session_list)), dtype=int)

    paths: list[TrackPath] = []
    for start_roi in start_rois:
        paths.extend(
            _best_track_paths(
                start_roi=start_roi,
                n_sessions=len(session_list),
                candidates_by_source=by_source,
                config=cfg,
                max_paths=cfg.path_candidates_per_seed,
            )
        )
    selected_paths = select_non_conflicting_paths(
        paths,
        fill_value=cfg.fill_value,
        beam_width=cfg.path_selection_beam_width,
    )
    if not selected_paths:
        return np.zeros((0, len(session_list)), dtype=int)
    local_tracks = np.asarray([path.row for path in selected_paths], dtype=int)
    return _local_tracks_to_suite2p(
        local_tracks, session_list, fill_value=cfg.fill_value
    )


def build_policy_candidate_edges(
    sessions: Sequence[Track2pSession],
    *,
    transform_type: str,
    config: Track2pPolicyDPConfig,
) -> dict[tuple[int, int], tuple[PolicyCandidate, ...]]:
    """Build accepted and rescue candidates for consecutive and skip edges."""

    session_list = list(sessions)
    output: dict[tuple[int, int], tuple[PolicyCandidate, ...]] = {}
    max_gap = min(int(config.max_gap), max(len(session_list) - 1, 1))
    for source_session in range(len(session_list) - 1):
        for gap in range(1, max_gap + 1):
            target_session = source_session + gap
            if target_session >= len(session_list):
                continue
            edge = (source_session, target_session)
            output[edge] = _policy_candidates_for_pair(
                session_list[source_session],
                session_list[target_session],
                edge=edge,
                transform_type=transform_type,
                config=config,
            )
    return output


def _policy_candidates_for_pair(
    reference_session: Track2pSession,
    moving_session: Track2pSession,
    *,
    edge: tuple[int, int],
    transform_type: str,
    config: Track2pPolicyDPConfig,
) -> tuple[PolicyCandidate, ...]:
    registered = register_plane_pair(
        reference_session.plane_data,
        moving_session.plane_data,
        transform_type=transform_type,
    )
    iou = _track2p_cross_iou_matrix(
        np.asarray(reference_session.plane_data.roi_masks) > 0,
        np.asarray(registered.roi_masks) > 0,
        distance_threshold=float(config.iou_distance_threshold),
    )
    if iou.size == 0:
        return ()

    row_ind, col_ind = linear_sum_assignment(1.0 - iou)
    assigned_iou = iou[row_ind, col_ind]
    threshold = _threshold_assigned_iou(assigned_iou, method=config.threshold_method)
    selected_pairs = {
        (int(row), int(col)) for row, col in zip(row_ind, col_ind, strict=True)
    }
    accepted_pairs = {
        (int(row), int(col))
        for row, col, value in zip(row_ind, col_ind, assigned_iou, strict=True)
        if float(value) > float(threshold)
    }

    candidates: dict[tuple[int, int], PolicyCandidate] = {}
    for row, col in accepted_pairs:
        _add_candidate(
            candidates,
            edge=edge,
            row=row,
            col=col,
            iou=float(iou[row, col]),
            threshold=float(threshold),
            accepted_by_threshold=True,
            selected_by_hungarian=True,
            config=config,
        )

    threshold_floor = max(0.0, float(threshold) - float(config.threshold_rescue_margin))
    rescue_floor = max(float(config.rescue_min_iou), threshold_floor)
    for row_index, row_values in enumerate(iou):
        finite = np.isfinite(row_values) & (row_values >= rescue_floor)
        columns = np.flatnonzero(finite)
        if columns.size == 0:
            continue
        ordered = columns[np.argsort(row_values[columns])[::-1]][: config.row_top_k]
        for col_index in ordered:
            pair = (int(row_index), int(col_index))
            _add_candidate(
                candidates,
                edge=edge,
                row=pair[0],
                col=pair[1],
                iou=float(iou[pair]),
                threshold=float(threshold),
                accepted_by_threshold=pair in accepted_pairs,
                selected_by_hungarian=pair in selected_pairs,
                config=config,
            )

    return tuple(
        sorted(
            candidates.values(),
            key=lambda item: (
                item.source_session,
                item.source_roi,
                -item.score,
                item.target_session,
                item.target_roi,
            ),
        )
    )


def _add_candidate(
    candidates: dict[tuple[int, int], PolicyCandidate],
    *,
    edge: tuple[int, int],
    row: int,
    col: int,
    iou: float,
    threshold: float,
    accepted_by_threshold: bool,
    selected_by_hungarian: bool,
    config: Track2pPolicyDPConfig,
) -> None:
    source_session, target_session = edge
    candidate = PolicyCandidate(
        source_session=int(source_session),
        target_session=int(target_session),
        source_roi=int(row),
        target_roi=int(col),
        score=_candidate_score(
            iou=iou,
            threshold=threshold,
            accepted_by_threshold=accepted_by_threshold,
            session_gap=target_session - source_session,
            config=config,
        ),
        iou=float(iou),
        threshold=float(threshold),
        accepted_by_threshold=bool(accepted_by_threshold),
        selected_by_hungarian=bool(selected_by_hungarian),
    )
    key = (int(row), int(col))
    previous = candidates.get(key)
    if previous is None or candidate.score > previous.score:
        candidates[key] = candidate


def _candidate_score(
    *,
    iou: float,
    threshold: float,
    accepted_by_threshold: bool,
    session_gap: int,
    config: Track2pPolicyDPConfig,
) -> float:
    clipped = float(np.clip(iou, config.logit_epsilon, 1.0 - config.logit_epsilon))
    logit = float(np.log(clipped / (1.0 - clipped)))
    score = logit + float(config.threshold_margin_weight) * (
        float(iou) - float(threshold)
    )
    if accepted_by_threshold:
        score += float(config.accepted_bonus)
    else:
        score -= float(config.rescue_penalty)
    if session_gap > 1:
        score -= float(config.gap_penalty) * float(session_gap - 1)
    return float(score)


def _candidates_by_source(
    candidates: Mapping[tuple[int, int], Sequence[PolicyCandidate]],
) -> dict[tuple[int, int], tuple[PolicyCandidate, ...]]:
    grouped: dict[tuple[int, int], list[PolicyCandidate]] = {}
    for edge_candidates in candidates.values():
        for candidate in edge_candidates:
            key = (int(candidate.source_session), int(candidate.source_roi))
            grouped.setdefault(key, []).append(candidate)
    return {
        key: tuple(
            sorted(
                values,
                key=lambda item: (-item.score, item.target_session, item.target_roi),
            )
        )
        for key, values in grouped.items()
    }


def _best_track_path(
    *,
    start_roi: int,
    n_sessions: int,
    candidates_by_source: Mapping[tuple[int, int], Sequence[PolicyCandidate]],
    config: Track2pPolicyDPConfig,
) -> TrackPath | None:
    paths = _best_track_paths(
        start_roi=start_roi,
        n_sessions=n_sessions,
        candidates_by_source=candidates_by_source,
        config=config,
        max_paths=1,
    )
    return paths[0] if paths else None


def _best_track_paths(
    *,
    start_roi: int,
    n_sessions: int,
    candidates_by_source: Mapping[tuple[int, int], Sequence[PolicyCandidate]],
    config: Track2pPolicyDPConfig,
    max_paths: int,
) -> tuple[TrackPath, ...]:
    if max_paths <= 0:
        raise ValueError("max_paths must be positive")
    row = [int(config.fill_value)] * int(n_sessions)
    row[0] = int(start_roi)
    active = [
        _BeamState(
            row=tuple(row), current_session=0, current_roi=int(start_roi), score=0.0
        )
    ]
    complete: list[_BeamState] = []

    while active:
        expanded: list[_BeamState] = []
        for state in active:
            if state.current_session >= n_sessions - 1:
                complete.append(state)
                continue
            candidates = candidates_by_source.get(
                (int(state.current_session), int(state.current_roi)), ()
            )
            if not candidates:
                complete.append(state)
                continue
            for candidate in candidates:
                if candidate.target_session <= state.current_session:
                    continue
                next_row = list(state.row)
                if next_row[candidate.target_session] not in (
                    int(config.fill_value),
                    candidate.target_roi,
                ):
                    continue
                next_row[candidate.target_session] = int(candidate.target_roi)
                expanded.append(
                    _BeamState(
                        row=tuple(next_row),
                        current_session=int(candidate.target_session),
                        current_roi=int(candidate.target_roi),
                        score=float(state.score + candidate.score),
                    )
                )
        if not expanded:
            break
        expanded.sort(key=lambda item: (-item.score, item.row))
        active = expanded[: int(config.beam_width)]

    pool = complete + active
    if not pool:
        return ()

    best_by_row: dict[tuple[int, ...], _BeamState] = {}
    for state in pool:
        previous = best_by_row.get(state.row)
        if previous is None or _state_is_better(state, previous):
            best_by_row[state.row] = state

    ordered = sorted(
        best_by_row.values(),
        key=lambda item: (-item.score, -_path_length(item.row), item.row),
    )
    return tuple(state.as_path() for state in ordered[: int(max_paths)])


def select_non_conflicting_paths(
    paths: Sequence[TrackPath],
    *,
    fill_value: int = -1,
    beam_width: int | None = None,
) -> tuple[TrackPath, ...]:
    """Select high-scoring paths while enforcing one ROI use per session.

    The previous implementation was greedy: once the top-scoring path for a seed
    occupied an ROI, lower-scoring compatible alternatives from other seeds were
    never considered.  That is fragile for complete-track F1 because one locally
    attractive path can block two or more slightly weaker but mutually
    compatible paths.  This selector decomposes the conflict graph into connected
    components and solves each component as a maximum-weight independent-set
    problem.  Small components are solved exactly; large components use a bounded
    deterministic beam to keep the benchmark runtime predictable.
    """

    if beam_width is not None and int(beam_width) <= 0:
        raise ValueError("beam_width must be positive when provided")

    candidate_paths, observation_sets = _prepare_candidate_paths(
        paths, fill_value=fill_value
    )
    if not candidate_paths:
        return ()

    selected: list[TrackPath] = []
    for component in _conflict_components(observation_sets):
        if len(component) == 1:
            selected.append(candidate_paths[component[0]])
            continue
        component_paths = tuple(candidate_paths[index] for index in component)
        component_observations = tuple(observation_sets[index] for index in component)
        selected.extend(
            _select_non_conflicting_component(
                component_paths,
                component_observations,
                beam_width=beam_width,
            )
        )

    return tuple(
        sorted(selected, key=lambda item: (-item.score, -_path_length(item.row), item.row))
    )


def _prepare_candidate_paths(
    paths: Sequence[TrackPath], *, fill_value: int
) -> tuple[tuple[TrackPath, ...], tuple[frozenset[tuple[int, int]], ...]]:
    best_by_row: dict[tuple[int, ...], TrackPath] = {}
    for path in paths:
        row = tuple(int(value) for value in path.row)
        observations = _path_observations(row, fill_value=fill_value)
        if not observations:
            continue
        normalized = TrackPath(row=row, score=float(path.score))
        previous = best_by_row.get(row)
        if previous is None or _track_path_sort_key(normalized) < _track_path_sort_key(
            previous
        ):
            best_by_row[row] = normalized

    ordered_paths = tuple(sorted(best_by_row.values(), key=_track_path_sort_key))
    observation_sets = tuple(
        _path_observations(path.row, fill_value=fill_value)
        for path in ordered_paths
    )
    return ordered_paths, observation_sets


def _path_observations(
    row: Sequence[int], *, fill_value: int
) -> frozenset[tuple[int, int]]:
    return frozenset(
        {
            (session_index, int(roi))
            for session_index, roi in enumerate(row)
            if int(roi) != int(fill_value)
        }
    )


def _conflict_components(
    observation_sets: Sequence[frozenset[tuple[int, int]]],
) -> tuple[tuple[int, ...], ...]:
    observation_to_paths: dict[tuple[int, int], list[int]] = {}
    for path_index, observations in enumerate(observation_sets):
        for observation in observations:
            observation_to_paths.setdefault(observation, []).append(path_index)

    neighbors = [set() for _ in observation_sets]
    for path_indices in observation_to_paths.values():
        if len(path_indices) < 2:
            continue
        path_index_set = set(path_indices)
        for path_index in path_indices:
            neighbors[path_index].update(path_index_set - {path_index})

    components: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for start in range(len(observation_sets)):
        if start in seen:
            continue
        stack = [start]
        component: list[int] = []
        seen.add(start)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(neighbors[current]):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                stack.append(neighbor)
        components.append(tuple(sorted(component)))
    return tuple(components)


def _select_non_conflicting_component(
    paths: Sequence[TrackPath],
    observation_sets: Sequence[frozenset[tuple[int, int]]],
    *,
    beam_width: int | None,
) -> tuple[TrackPath, ...]:
    if len(paths) <= 20 or beam_width is None:
        return _select_non_conflicting_component_exact(paths, observation_sets)
    return _select_non_conflicting_component_beam(
        paths, observation_sets, beam_width=int(beam_width)
    )


def _select_non_conflicting_component_exact(
    paths: Sequence[TrackPath],
    observation_sets: Sequence[frozenset[tuple[int, int]]],
) -> tuple[TrackPath, ...]:
    best_index = min(range(len(paths)), key=lambda index: _track_path_sort_key(paths[index]))
    best_indices: tuple[int, ...] = (best_index,)
    best_score = float(paths[best_index].score)
    positive_suffix = [0.0] * (len(paths) + 1)
    for index in range(len(paths) - 1, -1, -1):
        positive_suffix[index] = positive_suffix[index + 1] + max(
            0.0, float(paths[index].score)
        )

    def update(score: float, selected_indices: tuple[int, ...]) -> None:
        nonlocal best_indices, best_score
        if not selected_indices:
            return
        if _selection_is_better(
            score,
            selected_indices,
            best_score,
            best_indices,
            paths,
        ):
            best_score = float(score)
            best_indices = selected_indices

    def recurse(
        index: int,
        score: float,
        selected_indices: tuple[int, ...],
        occupied: frozenset[tuple[int, int]],
    ) -> None:
        if index >= len(paths):
            update(score, selected_indices)
            return
        if score + positive_suffix[index] < best_score:
            return

        observations = observation_sets[index]
        if not observations & occupied:
            recurse(
                index + 1,
                score + float(paths[index].score),
                selected_indices + (index,),
                occupied | observations,
            )
        recurse(index + 1, score, selected_indices, occupied)

    recurse(0, 0.0, (), frozenset())
    return tuple(paths[index] for index in best_indices)


def _select_non_conflicting_component_beam(
    paths: Sequence[TrackPath],
    observation_sets: Sequence[frozenset[tuple[int, int]]],
    *,
    beam_width: int,
) -> tuple[TrackPath, ...]:
    states: list[tuple[float, tuple[int, ...], frozenset[tuple[int, int]]]] = [
        (0.0, (), frozenset())
    ]
    for index, observations in enumerate(observation_sets):
        next_states = list(states)
        for score, selected_indices, occupied in states:
            if observations & occupied:
                continue
            next_states.append(
                (
                    score + float(paths[index].score),
                    selected_indices + (index,),
                    occupied | observations,
                )
            )
        states = _prune_selection_beam(next_states, paths, beam_width=beam_width)

    nonempty_states = [state for state in states if state[1]]
    if not nonempty_states:
        best_index = min(
            range(len(paths)), key=lambda index: _track_path_sort_key(paths[index])
        )
        return (paths[best_index],)
    best = min(
        nonempty_states,
        key=lambda state: _selection_sort_key(state[0], state[1], paths),
    )
    return tuple(paths[index] for index in best[1])


def _prune_selection_beam(
    states: Sequence[tuple[float, tuple[int, ...], frozenset[tuple[int, int]]]],
    paths: Sequence[TrackPath],
    *,
    beam_width: int,
) -> list[tuple[float, tuple[int, ...], frozenset[tuple[int, int]]]]:
    best_by_occupied: dict[
        frozenset[tuple[int, int]],
        tuple[float, tuple[int, ...], frozenset[tuple[int, int]]],
    ] = {}
    for state in states:
        previous = best_by_occupied.get(state[2])
        if previous is None or _selection_sort_key(
            state[0], state[1], paths
        ) < _selection_sort_key(previous[0], previous[1], paths):
            best_by_occupied[state[2]] = state
    ordered = sorted(
        best_by_occupied.values(),
        key=lambda state: _selection_sort_key(state[0], state[1], paths),
    )
    return ordered[: int(beam_width)]


def _state_is_better(candidate: _BeamState, incumbent: _BeamState) -> bool:
    return (
        float(candidate.score),
        _path_length(candidate.row),
        tuple(-int(value) for value in candidate.row),
    ) > (
        float(incumbent.score),
        _path_length(incumbent.row),
        tuple(-int(value) for value in incumbent.row),
    )


def _track_path_sort_key(path: TrackPath) -> tuple[float, int, tuple[int, ...]]:
    return (-float(path.score), -_path_length(path.row), path.row)


def _selection_sort_key(
    score: float, selected_indices: Sequence[int], paths: Sequence[TrackPath]
) -> tuple[float, int, tuple[tuple[int, ...], ...]]:
    return (
        -float(score),
        -sum(_path_length(paths[index].row) for index in selected_indices),
        tuple(paths[index].row for index in selected_indices),
    )


def _selection_is_better(
    candidate_score: float,
    candidate_indices: Sequence[int],
    incumbent_score: float,
    incumbent_indices: Sequence[int],
    paths: Sequence[TrackPath],
) -> bool:
    return _selection_sort_key(candidate_score, candidate_indices, paths) < (
        _selection_sort_key(incumbent_score, incumbent_indices, paths)
    )


def _path_length(row: Sequence[int]) -> int:
    return sum(1 for value in row if int(value) >= 0)


def _local_tracks_to_suite2p(
    local_tracks: np.ndarray,
    sessions: Sequence[Track2pSession],
    *,
    fill_value: int,
) -> np.ndarray:
    suite2p_tracks = np.full(local_tracks.shape, int(fill_value), dtype=int)
    roi_indices_by_session = [_roi_indices(session) for session in sessions]
    for session_index, roi_indices in enumerate(roi_indices_by_session):
        valid = local_tracks[:, session_index] >= 0
        if np.any(valid):
            suite2p_tracks[valid, session_index] = roi_indices[
                local_tracks[valid, session_index]
            ]
    return suite2p_tracks


def run_track2p_policy_dp_benchmark(
    config: Track2pBenchmarkConfig,
    *,
    dp_config: Track2pPolicyDPConfig | None = None,
    transform_type: str | None = None,
    cell_probability_threshold: float | None = None,
) -> list[SubjectBenchmarkResult]:
    """Run the DP-rescued Track2p-policy benchmark row."""

    cfg = dp_config or Track2pPolicyDPConfig()
    policy_config = track2p_policy_config(
        config,
        transform_type=transform_type,
        cell_probability_threshold=cell_probability_threshold,
    )
    policy_config = replace(policy_config, max_gap=int(cfg.max_gap))
    subject_dirs = discover_subject_dirs(policy_config.data)
    if not subject_dirs:
        raise ValueError(
            f"No Track2p-style subject directories found under {policy_config.data}"
        )

    results: list[SubjectBenchmarkResult] = []
    for subject_dir in subject_dirs:
        reference = _load_reference_for_subject(
            subject_dir, data_root=policy_config.data, config=policy_config
        )
        _validate_reference_for_benchmark(
            reference, subject_dir=subject_dir, config=policy_config
        )
        sessions = _load_subject_sessions(subject_dir, policy_config)
        _validate_reference_roi_indices(reference, sessions)
        predicted = track2p_policy_dp_tracks(
            sessions,
            transform_type=policy_config.transform_type,
            config=cfg,
        )
        scores = _score_prediction_against_reference(
            predicted, reference, config=policy_config
        )
        scores = {
            **scores,
            "track2p_policy_variant": TRACK2P_POLICY_DP_METHOD,
            "track2p_policy_threshold_method": str(cfg.threshold_method),
            "track2p_policy_iou_distance_threshold": float(cfg.iou_distance_threshold),
            "track2p_policy_cell_probability_threshold": float(
                policy_config.cell_probability_threshold
            ),
            "track2p_policy_transform_type": str(policy_config.transform_type),
            "track2p_policy_dp_row_top_k": int(cfg.row_top_k),
            "track2p_policy_dp_rescue_min_iou": float(cfg.rescue_min_iou),
            "track2p_policy_dp_threshold_rescue_margin": float(
                cfg.threshold_rescue_margin
            ),
            "track2p_policy_dp_gap_penalty": float(cfg.gap_penalty),
            "track2p_policy_dp_beam_width": int(cfg.beam_width),
            "track2p_policy_dp_max_gap": int(cfg.max_gap),
            "track2p_policy_dp_path_candidates_per_seed": int(
                cfg.path_candidates_per_seed
            ),
            "track2p_policy_dp_path_selection_beam_width": int(
                cfg.path_selection_beam_width
            ),
        }
        results.append(
            SubjectBenchmarkResult(
                subject=subject_dir.name,
                variant="Track2p-policy DP",
                method=cast(Any, TRACK2P_POLICY_DP_METHOD),
                scores=scores,
                n_sessions=len(sessions),
                reference_source=reference.source,
            )
        )
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bayescatrack.experiments.track2p_policy_dp_benchmark",
        description="Run the DP-rescued Track2p-policy benchmark method.",
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument(
        "--reference-kind",
        choices=("auto", "manual-gt", "track2p-output", "aligned-subject-rows"),
        default="manual-gt",
    )
    parser.add_argument("--plane", dest="plane_name", default="plane0")
    parser.add_argument(
        "--input-format", choices=("auto", "suite2p", "npy"), default="suite2p"
    )
    parser.add_argument(
        "--transform-type", default=TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE
    )
    parser.add_argument(
        "--threshold-method",
        choices=("otsu", "min"),
        default=TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    )
    parser.add_argument("--iou-distance-threshold", type=float, default=12.0)
    parser.add_argument(
        "--cell-probability-threshold",
        type=float,
        default=TRACK2P_POLICY_DEFAULT_CELL_PROBABILITY_THRESHOLD,
    )
    parser.add_argument("--row-top-k", type=int, default=2)
    parser.add_argument("--rescue-min-iou", type=float, default=0.10)
    parser.add_argument("--threshold-rescue-margin", type=float, default=0.15)
    parser.add_argument("--accepted-bonus", type=float, default=0.25)
    parser.add_argument("--rescue-penalty", type=float, default=0.25)
    parser.add_argument("--gap-penalty", type=float, default=1.0)
    parser.add_argument("--threshold-margin-weight", type=float, default=0.5)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument("--path-candidates-per-seed", type=int, default=4)
    parser.add_argument("--path-selection-beam-width", type=int, default=256)
    parser.add_argument(
        "--restrict-to-reference-seed-rois",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed-session", type=int, default=0)
    parser.add_argument(
        "--allow-track2p-as-reference-for-smoke-test", action="store_true"
    )
    parser.add_argument(
        "--include-behavior", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("table", "json", "csv"), default="table")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    dp_config = Track2pPolicyDPConfig(
        threshold_method=cast(Literal["otsu", "min"], args.threshold_method),
        iou_distance_threshold=args.iou_distance_threshold,
        row_top_k=args.row_top_k,
        rescue_min_iou=args.rescue_min_iou,
        threshold_rescue_margin=args.threshold_rescue_margin,
        accepted_bonus=args.accepted_bonus,
        rescue_penalty=args.rescue_penalty,
        gap_penalty=args.gap_penalty,
        threshold_margin_weight=args.threshold_margin_weight,
        beam_width=args.beam_width,
        max_gap=args.max_gap,
        path_candidates_per_seed=args.path_candidates_per_seed,
        path_selection_beam_width=args.path_selection_beam_width,
    )
    config = Track2pBenchmarkConfig(
        data=args.data,
        method="global-assignment",
        plane_name=args.plane_name,
        input_format=args.input_format,
        reference=args.reference,
        reference_kind=args.reference_kind,
        allow_track2p_as_reference_for_smoke_test=args.allow_track2p_as_reference_for_smoke_test,
        seed_session=args.seed_session,
        restrict_to_reference_seed_rois=args.restrict_to_reference_seed_rois,
        transform_type=args.transform_type,
        max_gap=dp_config.max_gap,
        include_behavior=args.include_behavior,
        include_non_cells=False,
        cell_probability_threshold=args.cell_probability_threshold,
        weighted_masks=False,
        weighted_centroids=False,
        exclude_overlapping_pixels=False,
    )
    results = run_track2p_policy_dp_benchmark(
        config,
        dp_config=dp_config,
        transform_type=args.transform_type,
        cell_probability_threshold=args.cell_probability_threshold,
    )
    rows = [result.to_dict() for result in results]
    if args.output is not None:
        write_results(rows, args.output, cast(OutputFormat, args.format))
    else:
        from bayescatrack.experiments.track2p_benchmark import _write_stdout

        _write_stdout(rows, cast(OutputFormat, args.format))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
