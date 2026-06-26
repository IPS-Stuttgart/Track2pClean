"""Tracklet-hypothesis graph benchmark for Track2p-style data.

This runner is a bounded alternative to raw scan-assignment MHT.  It first
builds high-precision local tracklets from label-free adjacent-session evidence,
then solves a small hypothesis-selection problem over possible tracklet joins.
The no-join hypothesis is always present, so uncertain boundaries remain
fragmented instead of being forced into false continuations.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from bayescatrack.experiments import track2p_policy_full_mht_benchmark as full_mht
from bayescatrack.experiments import track2p_policy_suffix_stitch_ranking_audit as rank
from bayescatrack.experiments.track2p_benchmark import (
    GROUND_TRUTH_REFERENCE_SOURCE,
    OutputFormat,
    SubjectBenchmarkResult,
    Track2pBenchmarkConfig,
    _load_reference_for_subject,
    _load_subject_sessions,
    _reference_matrix,
    _score_prediction_against_reference,
    _validate_reference_for_benchmark,
    _validate_reference_roi_indices,
    discover_subject_dirs,
    write_results,
)
from bayescatrack.experiments.track2p_emulation_benchmark import ThresholdMethod
from bayescatrack.experiments.track2p_policy_benchmark import (
    TRACK2P_POLICY_DEFAULT_CELL_PROBABILITY_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE,
    track2p_policy_config,
)


METHOD = "track2p-policy-tracklet-graph-mht"
SeedSource = Literal["reference", "all-cells"]


@dataclass(frozen=True)
class TrackletGraphConfig:
    """Controls for conservative tracklet construction and graph selection."""

    seed_source: SeedSource = "reference"
    max_seed_tracks: int | None = None
    beam_width: int = 64
    path_hypotheses: int = 16
    edge_top_k: int = 4
    max_join_gap: int = 3
    local_min_edge_score: float = 1.25
    local_min_registered_iou: float = 0.45
    local_min_shifted_iou: float = 0.35
    local_min_area_ratio: float = 0.70
    local_max_centroid_distance: float = 8.0
    local_max_growth_residual: float = 3.5
    local_min_ambiguity_margin: float = 0.15
    join_min_edge_score: float = 0.75
    join_min_registered_iou: float = 0.20
    join_min_shifted_iou: float = 0.20
    join_min_area_ratio: float = 0.55
    join_max_centroid_distance: float = 14.0
    join_max_growth_residual: float = 8.0
    join_score_frontier_min_edge_score: float = 0.0
    join_score_frontier_min_registered_iou: float = 0.30
    join_score_frontier_min_shifted_iou: float = 0.50
    join_score_frontier_min_area_ratio: float = 0.65
    join_score_frontier_max_centroid_distance: float = 4.5
    join_score_frontier_max_growth_residual: float = 4.5
    join_complexity_penalty: float = 0.35
    gap_penalty: float = 0.20
    component_incoherence_weight: float = 0.25
    registered_iou_weight: float = 1.0
    shifted_iou_weight: float = 1.5
    area_ratio_weight: float = 0.25
    cell_probability_weight: float = 0.25
    centroid_distance_weight: float = 0.05
    threshold_margin_weight: float = 0.50
    growth_residual_weight: float = 0.10
    growth_mahalanobis_weight: float = 0.25
    growth_anchor_min_registered_iou: float = 0.55
    growth_anchor_min_shifted_iou: float = 0.30
    growth_anchor_min_cell_probability: float = 0.80


@dataclass(frozen=True)
class Tracklet:
    """A conservative local fragment used as an atomic graph node."""

    tracklet_id: int
    rois: tuple[int, ...]
    start_session: int

    @property
    def end_session(self) -> int:
        return int(self.start_session + len(self.rois) - 1)

    @property
    def length(self) -> int:
        return int(len(self.rois))

    def roi_at(self, session_index: int) -> int | None:
        offset = int(session_index) - int(self.start_session)
        if offset < 0 or offset >= len(self.rois):
            return None
        return int(self.rois[offset])


@dataclass(frozen=True)
class TrackletEdge:
    """A candidate same-identity continuation between two tracklets."""

    source_id: int
    target_id: int
    score: float
    raw_score: float
    gap: int
    registered_iou: float
    shifted_iou: float
    centroid_distance: float
    area_ratio: float
    growth_residual: float
    growth_mahalanobis: float
    endpoint_cell_probability_min: float
    duplicate_source_rank: int
    duplicate_target_rank: int
    would_complete_track: bool
    suspicious_complete_component: bool
    component_incoherence: float


@dataclass(frozen=True)
class _PathHypothesis:
    tracklet_ids: tuple[int, ...]
    edge_ids: tuple[tuple[int, int], ...]
    score: float


@dataclass(frozen=True)
class TrackletGraphResult:
    """Benchmark rows plus graph diagnostics."""

    results: tuple[SubjectBenchmarkResult, ...]
    diagnostic_rows: tuple[dict[str, Any], ...]
    summary_rows: tuple[dict[str, Any], ...]


def run_track2p_policy_tracklet_graph_mht(
    config: Track2pBenchmarkConfig,
    *,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    iou_distance_threshold: float = TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    transform_type: str | None = None,
    cell_probability_threshold: float | None = None,
    graph_config: TrackletGraphConfig | None = None,
    progress: bool = False,
) -> TrackletGraphResult:
    """Run the tracklet-hypothesis graph benchmark row."""

    policy_config = track2p_policy_config(
        config,
        transform_type=transform_type,
        cell_probability_threshold=cell_probability_threshold,
    )
    graph_config = graph_config or TrackletGraphConfig()
    subject_dirs = discover_subject_dirs(policy_config.data)
    if not subject_dirs:
        raise ValueError(
            f"No Track2p-style subject directories found under {policy_config.data}"
        )

    results: list[SubjectBenchmarkResult] = []
    diagnostic_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for subject_index, subject_dir in enumerate(subject_dirs, start=1):
        if progress:
            print(
                f"{METHOD}: subject {subject_index}/{len(subject_dirs)} "
                f"{subject_dir.name}",
                flush=True,
            )
        output = _run_subject_tracklet_graph(
            subject_dir,
            config=policy_config,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
            graph_config=graph_config,
            progress=progress,
        )
        results.append(output["result"])
        diagnostic_rows.extend(output["diagnostic_rows"])
        summary_rows.append(output["summary_row"])
    summary_rows.append(_all_summary_row(summary_rows))
    return TrackletGraphResult(
        tuple(results), tuple(diagnostic_rows), tuple(summary_rows)
    )


def _run_subject_tracklet_graph(
    subject_dir: Path,
    *,
    config: Track2pBenchmarkConfig,
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
    graph_config: TrackletGraphConfig,
    progress: bool = False,
) -> dict[str, Any]:
    reference = _load_reference_for_subject(
        subject_dir, data_root=config.data, config=config
    )
    _validate_reference_for_benchmark(reference, subject_dir=subject_dir, config=config)
    if reference.source != GROUND_TRUTH_REFERENCE_SOURCE:
        raise ValueError(f"{METHOD} requires independent manual-GT references")
    sessions = _load_subject_sessions(subject_dir, config)
    _validate_reference_roi_indices(reference, sessions)
    reference_tracks = _reference_matrix(reference, curated_only=config.curated_only)
    n_sessions = len(sessions)
    if n_sessions < 2:
        raise ValueError(f"{subject_dir.name} has fewer than two sessions")

    seed_rois = full_mht._seed_rois(
        sessions,
        reference_tracks,
        seed_session=int(config.seed_session),
        seed_source=graph_config.seed_source,
        cell_probability_threshold=float(config.cell_probability_threshold),
    )
    if graph_config.max_seed_tracks is not None:
        seed_rois = seed_rois[: max(0, int(graph_config.max_seed_tracks))]
    if not seed_rois:
        raise ValueError(f"{subject_dir.name}: no seed ROIs for tracklet graph")

    feature_cache = rank._FeatureCache(
        sessions=sessions,
        transform_type=str(config.transform_type),
        threshold_method=threshold_method,
        iou_distance_threshold=float(iou_distance_threshold),
        cell_probability_threshold=float(config.cell_probability_threshold),
        matrices={},
    )
    matrix_config = _matrix_config(graph_config)
    forced_rois = {int(config.seed_session): tuple(seed_rois)}
    tracklets, local_rows = _build_conservative_tracklets(
        sessions,
        feature_cache=feature_cache,
        matrix_config=matrix_config,
        graph_config=graph_config,
        forced_rois_by_session=forced_rois,
        progress=progress,
        subject=subject_dir.name,
    )
    seed_tracklet_ids = _seed_tracklet_ids(
        tracklets,
        seed_rois=seed_rois,
        seed_session=int(config.seed_session),
    )
    edges, edge_rows, expanded_source_ids = _build_tracklet_edges(
        sessions,
        tracklets,
        feature_cache=feature_cache,
        matrix_config=matrix_config,
        graph_config=graph_config,
        seed_tracklet_ids=seed_tracklet_ids,
        n_sessions=n_sessions,
        subject=subject_dir.name,
        progress=progress,
    )
    selected_paths = _select_seed_paths(
        tracklets,
        edges,
        seed_rois=seed_rois,
        seed_session=int(config.seed_session),
        graph_config=graph_config,
    )
    prediction = _paths_to_prediction_matrix(selected_paths, tracklets, n_sessions)
    coverage_rows, coverage_summary = _coverage_audit_rows(
        subject_dir.name,
        reference_tracks,
        tracklets=tracklets,
        edges=edges,
        selected_paths=selected_paths,
        sessions=sessions,
        feature_cache=feature_cache,
        matrix_config=matrix_config,
        graph_config=graph_config,
        expanded_source_ids=expanded_source_ids,
        seed_session=int(config.seed_session),
        n_sessions=n_sessions,
    )
    scores = _score_prediction_against_reference(prediction, reference, config=config)
    scores = {
        **dict(scores),
        "tracklet_graph_mht_n_tracklets": int(len(tracklets)),
        "tracklet_graph_mht_n_candidate_edges": int(len(edges)),
        "tracklet_graph_mht_n_selected_paths": int(len(selected_paths)),
        "tracklet_graph_mht_n_selected_joins": int(
            sum(len(path.edge_ids) for path in selected_paths)
        ),
        "tracklet_graph_mht_beam_width": int(graph_config.beam_width),
        "tracklet_graph_mht_path_hypotheses": int(graph_config.path_hypotheses),
        "tracklet_graph_mht_max_join_gap": int(graph_config.max_join_gap),
        **coverage_summary,
    }
    result = SubjectBenchmarkResult(
        subject=subject_dir.name,
        variant="Tracklet hypothesis graph MHT",
        method=cast(Any, METHOD),
        scores=scores,
        n_sessions=n_sessions,
        reference_source=GROUND_TRUTH_REFERENCE_SOURCE,
    )
    path_rows = _path_diagnostic_rows(subject_dir.name, selected_paths)
    summary_row = {
        "subject": subject_dir.name,
        "n_seed_tracks": int(len(seed_rois)),
        "n_tracklets": int(len(tracklets)),
        "n_candidate_edges": int(len(edges)),
        "n_selected_paths": int(len(selected_paths)),
        "n_selected_joins": int(sum(len(path.edge_ids) for path in selected_paths)),
        "best_score": float(sum(float(path.score) for path in selected_paths)),
        "pairwise_f1": float(scores.get("pairwise_f1", float("nan"))),
        "complete_track_f1": float(scores.get("complete_track_f1", float("nan"))),
        **coverage_summary,
    }
    return {
        "result": result,
        "diagnostic_rows": [*local_rows, *edge_rows, *path_rows, *coverage_rows],
        "summary_row": summary_row,
    }


def _matrix_config(config: TrackletGraphConfig) -> full_mht.FullMHTConfig:
    return full_mht.FullMHTConfig(
        edge_top_k=max(1, int(config.edge_top_k)),
        registered_iou_weight=float(config.registered_iou_weight),
        shifted_iou_weight=float(config.shifted_iou_weight),
        area_ratio_weight=float(config.area_ratio_weight),
        cell_probability_weight=float(config.cell_probability_weight),
        centroid_distance_weight=float(config.centroid_distance_weight),
        threshold_margin_weight=float(config.threshold_margin_weight),
        growth_residual_weight=float(config.growth_residual_weight),
        growth_mahalanobis_weight=float(config.growth_mahalanobis_weight),
        growth_anchor_min_registered_iou=float(
            config.growth_anchor_min_registered_iou
        ),
        growth_anchor_min_shifted_iou=float(config.growth_anchor_min_shifted_iou),
        growth_anchor_min_cell_probability=float(
            config.growth_anchor_min_cell_probability
        ),
    )


def _build_conservative_tracklets(
    sessions: Sequence[Any],
    *,
    feature_cache: rank._FeatureCache,
    matrix_config: full_mht.FullMHTConfig,
    graph_config: TrackletGraphConfig,
    forced_rois_by_session: Mapping[int, Sequence[int]],
    progress: bool,
    subject: str,
) -> tuple[tuple[Tracklet, ...], list[dict[str, Any]]]:
    eligible = {
        session_index: _eligible_rois(
            sessions,
            session_index,
            threshold=float(feature_cache.cell_probability_threshold),
            forced_rois=forced_rois_by_session.get(session_index, ()),
        )
        for session_index in range(len(sessions))
    }
    successor: dict[tuple[int, int], tuple[int, int]] = {}
    predecessor: dict[tuple[int, int], tuple[int, int]] = {}
    diagnostic_rows: list[dict[str, Any]] = []
    for session_index in range(len(sessions) - 1):
        if progress:
            print(
                f"{METHOD}: {subject} local tracklets "
                f"{session_index}->{session_index + 1}",
                flush=True,
            )
        links, rows = _conservative_adjacent_links(
            sessions,
            feature_cache=feature_cache,
            matrix_config=matrix_config,
            graph_config=graph_config,
            source_session=session_index,
            target_session=session_index + 1,
            source_rois=eligible[session_index],
            subject=subject,
        )
        diagnostic_rows.extend(rows)
        for source_roi, target_roi in links.items():
            source_key = (int(session_index), int(source_roi))
            target_key = (int(session_index) + 1, int(target_roi))
            if source_key in successor or target_key in predecessor:
                continue
            successor[source_key] = target_key
            predecessor[target_key] = source_key

    visited: set[tuple[int, int]] = set()
    tracklets: list[Tracklet] = []
    for session_index in range(len(sessions)):
        for roi in eligible[session_index]:
            key = (int(session_index), int(roi))
            if key in visited or key in predecessor:
                continue
            chain = [key]
            visited.add(key)
            current = key
            while current in successor:
                current = successor[current]
                if current in visited:
                    break
                chain.append(current)
                visited.add(current)
            tracklets.append(
                Tracklet(
                    tracklet_id=len(tracklets),
                    start_session=int(chain[0][0]),
                    rois=tuple(int(item[1]) for item in chain),
                )
            )
    for session_index in range(len(sessions)):
        for roi in eligible[session_index]:
            key = (int(session_index), int(roi))
            if key in visited:
                continue
            visited.add(key)
            tracklets.append(
                Tracklet(
                    tracklet_id=len(tracklets),
                    start_session=int(session_index),
                    rois=(int(roi),),
                )
            )
    diagnostic_rows.extend(_tracklet_diagnostic_rows(subject, tracklets))
    return tuple(tracklets), diagnostic_rows


def _eligible_rois(
    sessions: Sequence[Any],
    session_index: int,
    *,
    threshold: float,
    forced_rois: Sequence[int],
) -> tuple[int, ...]:
    forced = {int(roi) for roi in forced_rois}
    output: list[int] = []
    for roi in full_mht._roi_indices(sessions[int(session_index)]):
        if int(roi) in forced or (
            full_mht._cell_probability(sessions, int(session_index), int(roi))
            >= float(threshold)
        ):
            output.append(int(roi))
    return tuple(sorted(set(output)))


def _conservative_adjacent_links(
    sessions: Sequence[Any],
    *,
    feature_cache: rank._FeatureCache,
    matrix_config: full_mht.FullMHTConfig,
    graph_config: TrackletGraphConfig,
    source_session: int,
    target_session: int,
    source_rois: Sequence[int],
    subject: str,
) -> tuple[dict[int, int], list[dict[str, Any]]]:
    if not source_rois:
        return {}, []
    matrices = full_mht._sparse_pair_matrices(
        sessions,
        feature_cache,
        source_session=int(source_session),
        target_session=int(target_session),
        source_rois=source_rois,
        edge_top_k=max(1, int(graph_config.edge_top_k)),
        config=matrix_config,
    )
    score_matrix = _score_matrix(
        sessions,
        matrices,
        target_session=int(target_session),
        matrix_config=matrix_config,
    )
    valid = _local_valid_mask(matrices, score_matrix, graph_config)
    pairs = _mutual_local_links(
        score_matrix,
        valid,
        min_margin=float(graph_config.local_min_ambiguity_margin),
    )
    links: dict[int, int] = {}
    rows: list[dict[str, Any]] = []
    pair_set = set(pairs)
    source_indices = np.asarray(matrices.source_indices, dtype=int)
    for row_index, source_roi in enumerate(source_indices):
        candidates = np.flatnonzero(valid[int(row_index)])
        for col_index in candidates:
            selected = (int(row_index), int(col_index)) in pair_set
            target_roi = int(matrices.target_indices[int(col_index)])
            if selected:
                links[int(source_roi)] = target_roi
            rows.append(
                _edge_feature_row(
                    "local_link_candidate",
                    subject,
                    source_session=int(source_session),
                    source_roi=int(source_roi),
                    target_session=int(target_session),
                    target_roi=target_roi,
                    score=float(score_matrix[int(row_index), int(col_index)]),
                    matrices=matrices,
                    row_index=int(row_index),
                    col_index=int(col_index),
                    selected=selected,
                )
            )
    return links, rows


def _score_matrix(
    sessions: Sequence[Any],
    matrices: full_mht._FullMHTPairMatrices,
    *,
    target_session: int,
    matrix_config: full_mht.FullMHTConfig,
) -> np.ndarray:
    scores = np.full(matrices.registered_iou.shape, -np.inf, dtype=float)
    for row_index in range(scores.shape[0]):
        for col_index in range(scores.shape[1]):
            scores[int(row_index), int(col_index)] = full_mht._edge_score(
                sessions,
                matrices,
                target_session=int(target_session),
                source_local=int(row_index),
                target_local=int(col_index),
                config=matrix_config,
            )
    return scores


def _local_valid_mask(
    matrices: full_mht._FullMHTPairMatrices,
    score_matrix: np.ndarray,
    config: TrackletGraphConfig,
) -> np.ndarray:
    return (
        np.isfinite(score_matrix)
        & (score_matrix >= float(config.local_min_edge_score))
        & (matrices.registered_iou >= float(config.local_min_registered_iou))
        & (matrices.shifted_iou >= float(config.local_min_shifted_iou))
        & (matrices.area_ratio >= float(config.local_min_area_ratio))
        & (matrices.centroid_distance <= float(config.local_max_centroid_distance))
        & (matrices.growth_residual <= float(config.local_max_growth_residual))
    )


def _join_valid_mask(
    matrices: full_mht._FullMHTPairMatrices,
    score_matrix: np.ndarray,
    config: TrackletGraphConfig,
) -> np.ndarray:
    strict = (
        np.isfinite(score_matrix)
        & (score_matrix >= float(config.join_min_edge_score))
        & (matrices.registered_iou >= float(config.join_min_registered_iou))
        & (matrices.shifted_iou >= float(config.join_min_shifted_iou))
        & (matrices.area_ratio >= float(config.join_min_area_ratio))
        & (matrices.centroid_distance <= float(config.join_max_centroid_distance))
        & (matrices.growth_residual <= float(config.join_max_growth_residual))
    )
    score_frontier = (
        np.isfinite(score_matrix)
        & (score_matrix >= float(config.join_score_frontier_min_edge_score))
        & (
            matrices.registered_iou
            >= float(config.join_score_frontier_min_registered_iou)
        )
        & (matrices.shifted_iou >= float(config.join_score_frontier_min_shifted_iou))
        & (matrices.area_ratio >= float(config.join_score_frontier_min_area_ratio))
        & (
            matrices.centroid_distance
            <= float(config.join_score_frontier_max_centroid_distance)
        )
        & (
            matrices.growth_residual
            <= float(config.join_score_frontier_max_growth_residual)
        )
    )
    return strict | score_frontier


def _mutual_local_links(
    score_matrix: np.ndarray,
    valid_mask: np.ndarray,
    *,
    min_margin: float,
) -> tuple[tuple[int, int], ...]:
    """Return mutual-best links whose row and column margins are conservative."""

    scores = np.asarray(score_matrix, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool)
    if scores.size == 0 or scores.shape != valid.shape:
        return tuple()
    masked = np.where(valid, scores, -np.inf)
    if not np.isfinite(masked).any():
        return tuple()
    row_best = np.argmax(masked, axis=1)
    col_best = np.argmax(masked, axis=0)
    pairs: list[tuple[int, int]] = []
    for row_index, col_index in enumerate(row_best):
        row = int(row_index)
        col = int(col_index)
        if not np.isfinite(masked[row, col]):
            continue
        if int(col_best[col]) != row:
            continue
        if _selection_margin(masked[row, :], col) < float(min_margin):
            continue
        if _selection_margin(masked[:, col], row) < float(min_margin):
            continue
        pairs.append((row, col))
    return tuple(pairs)


def _selection_margin(values: np.ndarray, selected_index: int) -> float:
    finite = np.asarray(values, dtype=float)
    selected = float(finite[int(selected_index)])
    competitors = np.delete(finite, int(selected_index))
    competitors = competitors[np.isfinite(competitors)]
    if competitors.size == 0:
        return float("inf")
    return float(selected - float(np.max(competitors)))


def _seed_tracklet_ids(
    tracklets: Sequence[Tracklet],
    *,
    seed_rois: Sequence[int],
    seed_session: int,
) -> tuple[int, ...]:
    output: list[int] = []
    for seed_roi in seed_rois:
        tracklet_id = _tracklet_containing(tracklets, int(seed_session), int(seed_roi))
        if tracklet_id is not None:
            output.append(int(tracklet_id))
    return tuple(dict.fromkeys(output))


def _build_tracklet_edges(
    sessions: Sequence[Any],
    tracklets: Sequence[Tracklet],
    *,
    feature_cache: rank._FeatureCache,
    matrix_config: full_mht.FullMHTConfig,
    graph_config: TrackletGraphConfig,
    seed_tracklet_ids: Sequence[int],
    n_sessions: int,
    subject: str,
    progress: bool,
) -> tuple[tuple[TrackletEdge, ...], list[dict[str, Any]], tuple[int, ...]]:
    edges: list[TrackletEdge] = []
    tracklets_by_id = {int(tracklet.tracklet_id): tracklet for tracklet in tracklets}
    tracklets_by_start: dict[int, list[Tracklet]] = {}
    for tracklet in tracklets:
        tracklets_by_start.setdefault(int(tracklet.start_session), []).append(tracklet)
    queue = list(dict.fromkeys(int(item) for item in seed_tracklet_ids))
    expanded_sources: set[int] = set()
    seen_edges: set[tuple[int, int]] = set()
    while queue:
        source_id = int(queue.pop(0))
        if source_id in expanded_sources:
            continue
        source = tracklets_by_id.get(source_id)
        if source is None:
            continue
        expanded_sources.add(source_id)
        if progress:
            print(
                f"{METHOD}: {subject} graph source={source_id} "
                f"end_session={source.end_session}",
                flush=True,
            )
        scored_for_source = _candidate_edges_for_source(
            sessions,
            source,
            tracklets_by_start=tracklets_by_start,
            feature_cache=feature_cache,
            matrix_config=matrix_config,
            graph_config=graph_config,
            n_sessions=int(n_sessions),
        )
        for edge in scored_for_source[: max(1, int(graph_config.edge_top_k))]:
            edge_key = (int(edge.source_id), int(edge.target_id))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            edges.append(edge)
            if int(edge.target_id) not in expanded_sources:
                queue.append(int(edge.target_id))
    ranked_edges = _rank_duplicate_edges(edges)
    rows: list[dict[str, Any]] = []
    for edge in ranked_edges:
        rows.append(_tracklet_edge_row(subject, edge, tracklets_by_id))
    return tuple(ranked_edges), rows, tuple(sorted(expanded_sources))


def _candidate_edges_for_source(
    sessions: Sequence[Any],
    source: Tracklet,
    *,
    tracklets_by_start: Mapping[int, Sequence[Tracklet]],
    feature_cache: rank._FeatureCache,
    matrix_config: full_mht.FullMHTConfig,
    graph_config: TrackletGraphConfig,
    n_sessions: int,
) -> list[TrackletEdge]:
    scored: list[TrackletEdge] = []
    max_gap = max(1, int(graph_config.max_join_gap))
    for target_session in range(
        int(source.end_session) + 1,
        min(int(n_sessions), int(source.end_session) + max_gap + 1),
    ):
        target_candidates = list(tracklets_by_start.get(int(target_session), ()))
        if not target_candidates:
            continue
        matrices = full_mht._sparse_pair_matrices(
            sessions,
            feature_cache,
            source_session=int(source.end_session),
            target_session=int(target_session),
            source_rois=(int(source.rois[-1]),),
            edge_top_k=max(1, int(graph_config.edge_top_k)),
            config=matrix_config,
        )
        score_matrix = _score_matrix(
            sessions,
            matrices,
            target_session=int(target_session),
            matrix_config=matrix_config,
        )
        valid = _join_valid_mask(matrices, score_matrix, graph_config)
        if valid.shape[0] == 0:
            continue
        target_lookup = {
            int(roi): index
            for index, roi in enumerate(np.asarray(matrices.target_indices))
        }
        for target in target_candidates:
            target_local = target_lookup.get(int(target.rois[0]))
            if target_local is None or not bool(valid[0, target_local]):
                continue
            scored.append(
                _make_tracklet_edge(
                    sessions,
                    source,
                    target,
                    matrices=matrices,
                    source_local=0,
                    target_local=int(target_local),
                    raw_score=float(score_matrix[0, target_local]),
                    graph_config=graph_config,
                    n_sessions=int(n_sessions),
                )
            )
    scored.sort(key=lambda edge: -float(edge.score))
    return scored


def _make_tracklet_edge(
    sessions: Sequence[Any],
    source: Tracklet,
    target: Tracklet,
    *,
    matrices: full_mht._FullMHTPairMatrices,
    source_local: int,
    target_local: int,
    raw_score: float,
    graph_config: TrackletGraphConfig,
    n_sessions: int,
) -> TrackletEdge:
    gap = int(target.start_session - source.end_session)
    component_incoherence = _component_incoherence(
        raw_score=float(raw_score),
        source=source,
        target=target,
        n_sessions=int(n_sessions),
        config=graph_config,
    )
    score = (
        float(raw_score)
        - float(graph_config.join_complexity_penalty)
        - float(graph_config.gap_penalty) * float(max(0, gap - 1))
        - float(graph_config.component_incoherence_weight) * component_incoherence
    )
    return TrackletEdge(
        source_id=int(source.tracklet_id),
        target_id=int(target.tracklet_id),
        score=float(score),
        raw_score=float(raw_score),
        gap=gap,
        registered_iou=float(matrices.registered_iou[source_local, target_local]),
        shifted_iou=float(matrices.shifted_iou[source_local, target_local]),
        centroid_distance=float(matrices.centroid_distance[source_local, target_local]),
        area_ratio=float(matrices.area_ratio[source_local, target_local]),
        growth_residual=float(matrices.growth_residual[source_local, target_local]),
        growth_mahalanobis=float(
            matrices.growth_mahalanobis[source_local, target_local]
        ),
        endpoint_cell_probability_min=min(
            full_mht._cell_probability(
                sessions, source.end_session, int(source.rois[-1])
            ),
            full_mht._cell_probability(
                sessions, target.start_session, int(target.rois[0])
            ),
        ),
        duplicate_source_rank=0,
        duplicate_target_rank=0,
        would_complete_track=_would_complete_track(
            source, target, n_sessions=int(n_sessions)
        ),
        suspicious_complete_component=component_incoherence > 0.0,
        component_incoherence=component_incoherence,
    )


def _component_incoherence(
    *,
    raw_score: float,
    source: Tracklet,
    target: Tracklet,
    n_sessions: int,
    config: TrackletGraphConfig,
) -> float:
    if not _would_complete_track(source, target, n_sessions=int(n_sessions)):
        return 0.0
    return max(0.0, float(config.local_min_edge_score) - float(raw_score))


def _would_complete_track(
    source: Tracklet, target: Tracklet, *, n_sessions: int
) -> bool:
    return (
        int(source.start_session) == 0
        and int(target.end_session) >= int(n_sessions) - 1
    )


def _rank_duplicate_edges(edges: Sequence[TrackletEdge]) -> tuple[TrackletEdge, ...]:
    source_groups: dict[int, list[TrackletEdge]] = {}
    target_groups: dict[int, list[TrackletEdge]] = {}
    for edge in edges:
        source_groups.setdefault(int(edge.source_id), []).append(edge)
        target_groups.setdefault(int(edge.target_id), []).append(edge)
    source_rank = _rank_map(source_groups)
    target_rank = _rank_map(target_groups)
    ranked: list[TrackletEdge] = []
    for edge in edges:
        ranked.append(
            replace(
                edge,
                duplicate_source_rank=source_rank[
                    (int(edge.source_id), int(edge.target_id))
                ],
                duplicate_target_rank=target_rank[
                    (int(edge.source_id), int(edge.target_id))
                ],
            )
        )
    ranked.sort(
        key=lambda edge: (
            -float(edge.score),
            int(edge.source_id),
            int(edge.target_id),
        )
    )
    return tuple(ranked)


def _rank_map(
    groups: Mapping[int, Sequence[TrackletEdge]],
) -> dict[tuple[int, int], int]:
    output: dict[tuple[int, int], int] = {}
    for group in groups.values():
        ordered = sorted(group, key=lambda edge: -float(edge.score))
        for rank_index, edge in enumerate(ordered, start=1):
            output[(int(edge.source_id), int(edge.target_id))] = int(rank_index)
    return output


def _select_seed_paths(
    tracklets: Sequence[Tracklet],
    edges: Sequence[TrackletEdge],
    *,
    seed_rois: Sequence[int],
    seed_session: int,
    graph_config: TrackletGraphConfig,
) -> tuple[_PathHypothesis, ...]:
    tracklets_by_id = {int(tracklet.tracklet_id): tracklet for tracklet in tracklets}
    seed_start_ids: list[int] = []
    for seed_roi in seed_rois:
        tracklet_id = _tracklet_containing(tracklets, int(seed_session), int(seed_roi))
        if tracklet_id is not None:
            seed_start_ids.append(int(tracklet_id))
    outgoing: dict[int, list[TrackletEdge]] = {}
    for edge in edges:
        outgoing.setdefault(int(edge.source_id), []).append(edge)
    for candidates in outgoing.values():
        candidates.sort(key=lambda edge: -float(edge.score))

    path_options = {
        int(start_id): _enumerate_paths(
            int(start_id),
            outgoing=outgoing,
            graph_config=graph_config,
        )
        for start_id in seed_start_ids
    }
    beams: list[tuple[tuple[_PathHypothesis, ...], frozenset[int], float]] = [
        (tuple(), frozenset(), 0.0)
    ]
    for start_id in seed_start_ids:
        options = path_options.get(
            int(start_id),
            (_PathHypothesis((int(start_id),), tuple(), 0.0),),
        )
        expanded: list[tuple[tuple[_PathHypothesis, ...], frozenset[int], float]] = []
        for selected_paths, used_tracklets, score in beams:
            for path in options:
                path_tracklets = frozenset(
                    int(tracklet_id) for tracklet_id in path.tracklet_ids
                )
                if used_tracklets.intersection(path_tracklets):
                    continue
                expanded.append(
                    (
                        selected_paths + (path,),
                        used_tracklets.union(path_tracklets),
                        float(score) + float(path.score),
                    )
                )
        if not expanded:
            continue
        expanded.sort(key=lambda item: -float(item[2]))
        beams = expanded[: max(1, int(graph_config.beam_width))]
    if not beams:
        return tuple()
    best_paths = beams[0][0]
    return tuple(
        sorted(
            best_paths,
            key=lambda path: (
                tracklets_by_id[int(path.tracklet_ids[0])].start_session,
                tracklets_by_id[int(path.tracklet_ids[0])].rois[0],
            ),
        )
    )


def _enumerate_paths(
    start_id: int,
    *,
    outgoing: Mapping[int, Sequence[TrackletEdge]],
    graph_config: TrackletGraphConfig,
) -> tuple[_PathHypothesis, ...]:
    completed: list[_PathHypothesis] = []
    stack = [_PathHypothesis((int(start_id),), tuple(), 0.0)]
    while stack:
        path = stack.pop()
        completed.append(path)
        last_id = int(path.tracklet_ids[-1])
        outgoing_edges = list(outgoing.get(last_id, ()))
        for edge in outgoing_edges[: max(1, int(graph_config.edge_top_k))]:
            target_id = int(edge.target_id)
            if target_id in path.tracklet_ids:
                continue
            stack.append(
                _PathHypothesis(
                    tracklet_ids=path.tracklet_ids + (target_id,),
                    edge_ids=path.edge_ids + ((int(edge.source_id), target_id),),
                    score=float(path.score) + float(edge.score),
                )
            )
    completed.sort(
        key=lambda path: (
            -float(path.score),
            -int(len(path.tracklet_ids)),
            tuple(int(tracklet_id) for tracklet_id in path.tracklet_ids),
        )
    )
    return tuple(completed[: max(1, int(graph_config.path_hypotheses))])


def _tracklet_containing(
    tracklets: Sequence[Tracklet], session_index: int, roi: int
) -> int | None:
    for tracklet in tracklets:
        if tracklet.roi_at(int(session_index)) == int(roi):
            return int(tracklet.tracklet_id)
    return None


def _paths_to_prediction_matrix(
    paths: Sequence[_PathHypothesis],
    tracklets: Sequence[Tracklet],
    n_sessions: int,
) -> np.ndarray:
    tracklets_by_id = {int(tracklet.tracklet_id): tracklet for tracklet in tracklets}
    matrix = np.full((len(paths), int(n_sessions)), -1, dtype=int)
    for row_index, path in enumerate(paths):
        for tracklet_id in path.tracklet_ids:
            tracklet = tracklets_by_id[int(tracklet_id)]
            for offset, roi in enumerate(tracklet.rois):
                session_index = int(tracklet.start_session) + int(offset)
                matrix[int(row_index), session_index] = int(roi)
    return matrix


def _coverage_audit_rows(
    subject: str,
    reference_tracks: np.ndarray,
    *,
    tracklets: Sequence[Tracklet],
    edges: Sequence[TrackletEdge],
    selected_paths: Sequence[_PathHypothesis],
    sessions: Sequence[Any] | None = None,
    feature_cache: rank._FeatureCache | None = None,
    matrix_config: full_mht.FullMHTConfig | None = None,
    graph_config: TrackletGraphConfig | None = None,
    expanded_source_ids: Sequence[int] = (),
    seed_session: int = 0,
    n_sessions: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return GT-only coverage diagnostics after graph selection is fixed."""

    reference = np.asarray(reference_tracks, dtype=int)
    audit_n_sessions = (
        int(reference.shape[1]) if n_sessions is None else int(n_sessions)
    )
    observation_to_tracklet = _observation_tracklet_map(tracklets)
    tracklets_by_id = {int(tracklet.tracklet_id): tracklet for tracklet in tracklets}
    tracklets_by_start: dict[int, list[Tracklet]] = {}
    for tracklet in tracklets:
        tracklets_by_start.setdefault(int(tracklet.start_session), []).append(tracklet)
    candidate_edges = {(int(edge.source_id), int(edge.target_id)) for edge in edges}
    expanded_sources = {int(tracklet_id) for tracklet_id in expanded_source_ids}
    selected_edges = {
        (int(source_id), int(target_id))
        for path in selected_paths
        for source_id, target_id in path.edge_ids
    }
    selected_successor_by_source = {
        int(source_id): int(target_id) for source_id, target_id in selected_edges
    }
    selected_predecessor_by_target = {
        int(target_id): int(source_id) for source_id, target_id in selected_edges
    }

    rows: list[dict[str, Any]] = []
    summary = {
        "tracklet_graph_audit_reference_tracks": int(reference.shape[0]),
        "tracklet_graph_audit_reference_tracks_with_seed_observation": 0,
        "tracklet_graph_audit_reference_tracks_with_seed_tracklet": 0,
        "tracklet_graph_audit_reference_links": 0,
        "tracklet_graph_audit_reference_links_preserved_in_tracklets": 0,
        "tracklet_graph_audit_reference_breaks": 0,
        "tracklet_graph_audit_break_correct_join_present": 0,
        "tracklet_graph_audit_break_correct_join_selected": 0,
        "tracklet_graph_audit_break_solver_rejected": 0,
        "tracklet_graph_audit_break_candidate_missing": 0,
        "tracklet_graph_audit_break_join_ineligible": 0,
        "tracklet_graph_audit_failure_tracklet_builder_too_strict": 0,
        "tracklet_graph_audit_failure_graph_candidate_missing": 0,
        "tracklet_graph_audit_failure_solver_too_conservative": 0,
        "tracklet_graph_audit_failure_conflict_issue": 0,
        "tracklet_graph_audit_recovered_correct_joins": 0,
        "tracklet_graph_audit_missing_source_not_reachable": 0,
        "tracklet_graph_audit_missing_outside_max_join_gap": 0,
        "tracklet_graph_audit_missing_source_not_in_sparse_support": 0,
        "tracklet_graph_audit_missing_target_cell_probability_below_threshold": 0,
        "tracklet_graph_audit_missing_target_not_in_sparse_support": 0,
        "tracklet_graph_audit_missing_registered_iou_below_gate": 0,
        "tracklet_graph_audit_missing_shifted_iou_below_gate": 0,
        "tracklet_graph_audit_missing_area_ratio_below_gate": 0,
        "tracklet_graph_audit_missing_centroid_distance_above_gate": 0,
        "tracklet_graph_audit_missing_growth_residual_above_gate": 0,
        "tracklet_graph_audit_missing_score_below_gate": 0,
        "tracklet_graph_audit_missing_lost_by_edge_top_k": 0,
        "tracklet_graph_audit_missing_materialization_unexpected": 0,
        "tracklet_graph_audit_missing_unclassified": 0,
        "tracklet_graph_audit_tracks_split_0": 0,
        "tracklet_graph_audit_tracks_split_1": 0,
        "tracklet_graph_audit_tracks_split_2": 0,
        "tracklet_graph_audit_tracks_split_3": 0,
        "tracklet_graph_audit_tracks_split_4plus": 0,
    }
    for reference_index, reference_row in enumerate(reference):
        track_row, break_rows = _reference_track_coverage_rows(
            subject,
            int(reference_index),
            np.asarray(reference_row, dtype=int),
            observation_to_tracklet=observation_to_tracklet,
            tracklets_by_id=tracklets_by_id,
            candidate_edges=candidate_edges,
            selected_edges=selected_edges,
            selected_successor_by_source=selected_successor_by_source,
            selected_predecessor_by_target=selected_predecessor_by_target,
            sessions=sessions,
            feature_cache=feature_cache,
            matrix_config=matrix_config,
            graph_config=graph_config,
            expanded_sources=expanded_sources,
            tracklets_by_start=tracklets_by_start,
            seed_session=int(seed_session),
            n_sessions=audit_n_sessions,
        )
        rows.append(track_row)
        rows.extend(break_rows)
        _accumulate_coverage_summary(summary, track_row, break_rows)
    rows.append({"row_type": "coverage_summary", "subject": subject, **summary})
    return rows, summary


def _observation_tracklet_map(
    tracklets: Sequence[Tracklet],
) -> dict[tuple[int, int], int]:
    output: dict[tuple[int, int], int] = {}
    for tracklet in tracklets:
        for offset, roi in enumerate(tracklet.rois):
            session_index = int(tracklet.start_session) + int(offset)
            output[(session_index, int(roi))] = int(tracklet.tracklet_id)
    return output


def _reference_track_coverage_rows(
    subject: str,
    reference_index: int,
    reference_row: np.ndarray,
    *,
    observation_to_tracklet: Mapping[tuple[int, int], int],
    tracklets_by_id: Mapping[int, Tracklet],
    candidate_edges: set[tuple[int, int]],
    selected_edges: set[tuple[int, int]],
    selected_successor_by_source: Mapping[int, int],
    selected_predecessor_by_target: Mapping[int, int],
    sessions: Sequence[Any] | None,
    feature_cache: rank._FeatureCache | None,
    matrix_config: full_mht.FullMHTConfig | None,
    graph_config: TrackletGraphConfig | None,
    expanded_sources: set[int],
    tracklets_by_start: Mapping[int, Sequence[Tracklet]],
    seed_session: int,
    n_sessions: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    observations = [
        (int(session_index), int(roi))
        for session_index, roi in enumerate(np.asarray(reference_row, dtype=int))
        if int(roi) >= 0
    ]
    observation_tracklets = [
        observation_to_tracklet.get((int(session_index), int(roi)))
        for session_index, roi in observations
    ]
    seed_roi = (
        int(reference_row[int(seed_session)])
        if 0 <= int(seed_session) < int(reference_row.shape[0])
        else -1
    )
    seed_tracklet_id = (
        observation_to_tracklet.get((int(seed_session), int(seed_roi)))
        if seed_roi >= 0
        else None
    )
    reference_links = 0
    preserved_links = 0
    break_rows: list[dict[str, Any]] = []
    for session_index in range(max(0, int(reference_row.shape[0]) - 1)):
        source_roi = int(reference_row[int(session_index)])
        target_roi = int(reference_row[int(session_index) + 1])
        if source_roi < 0 or target_roi < 0:
            continue
        reference_links += 1
        source_tid = observation_to_tracklet.get((int(session_index), source_roi))
        target_tid = observation_to_tracklet.get((int(session_index) + 1, target_roi))
        if source_tid is not None and source_tid == target_tid:
            preserved_links += 1
            continue
        break_rows.append(
            _reference_break_row(
                subject,
                reference_index,
                source_session=int(session_index),
                source_roi=source_roi,
                target_session=int(session_index) + 1,
                target_roi=target_roi,
                source_tracklet_id=source_tid,
                target_tracklet_id=target_tid,
                tracklets_by_id=tracklets_by_id,
                candidate_edges=candidate_edges,
                selected_edges=selected_edges,
                selected_successor_by_source=selected_successor_by_source,
                selected_predecessor_by_target=selected_predecessor_by_target,
                sessions=sessions,
                feature_cache=feature_cache,
                matrix_config=matrix_config,
                graph_config=graph_config,
                expanded_sources=expanded_sources,
                tracklets_by_start=tracklets_by_start,
                n_sessions=int(n_sessions),
            )
        )
    covered_tracklet_ids = [
        int(tracklet_id)
        for tracklet_id in observation_tracklets
        if tracklet_id is not None
    ]
    covered_tracklet_count = len(set(covered_tracklet_ids))
    tracklet_fragments = _tracklet_fragment_count(observation_tracklets)
    track_row = {
        "row_type": "reference_track_coverage",
        "subject": subject,
        "reference_track_index": int(reference_index),
        "seed_session": int(seed_session),
        "seed_roi": int(seed_roi),
        "seed_tracklet_id": "" if seed_tracklet_id is None else int(seed_tracklet_id),
        "has_seed_observation": int(seed_roi >= 0),
        "has_seed_tracklet": int(seed_tracklet_id is not None),
        "reference_observations": int(len(observations)),
        "covered_reference_observations": int(len(covered_tracklet_ids)),
        "covered_tracklet_count": int(covered_tracklet_count),
        "covered_tracklet_fragments": int(tracklet_fragments),
        "reference_links": int(reference_links),
        "reference_links_preserved_in_tracklets": int(preserved_links),
        "reference_breaks": int(reference_links - preserved_links),
        "tracklet_ids": " ".join(str(item) for item in covered_tracklet_ids),
    }
    return track_row, break_rows


def _reference_break_row(
    subject: str,
    reference_index: int,
    *,
    source_session: int,
    source_roi: int,
    target_session: int,
    target_roi: int,
    source_tracklet_id: int | None,
    target_tracklet_id: int | None,
    tracklets_by_id: Mapping[int, Tracklet],
    candidate_edges: set[tuple[int, int]],
    selected_edges: set[tuple[int, int]],
    selected_successor_by_source: Mapping[int, int],
    selected_predecessor_by_target: Mapping[int, int],
    sessions: Sequence[Any] | None,
    feature_cache: rank._FeatureCache | None,
    matrix_config: full_mht.FullMHTConfig | None,
    graph_config: TrackletGraphConfig | None,
    expanded_sources: set[int],
    tracklets_by_start: Mapping[int, Sequence[Tracklet]],
    n_sessions: int | None,
) -> dict[str, Any]:
    source_tracklet = (
        tracklets_by_id.get(int(source_tracklet_id))
        if source_tracklet_id is not None
        else None
    )
    target_tracklet = (
        tracklets_by_id.get(int(target_tracklet_id))
        if target_tracklet_id is not None
        else None
    )
    join_eligible = (
        source_tracklet is not None
        and target_tracklet is not None
        and int(source_tracklet.end_session) == int(source_session)
        and int(target_tracklet.start_session) == int(target_session)
    )
    edge_key = (
        int(source_tracklet_id) if source_tracklet_id is not None else -1,
        int(target_tracklet_id) if target_tracklet_id is not None else -1,
    )
    join_present = bool(join_eligible and edge_key in candidate_edges)
    join_selected = bool(join_present and edge_key in selected_edges)
    if source_tracklet_id is None:
        reason = "source_observation_uncovered"
    elif target_tracklet_id is None:
        reason = "target_observation_uncovered"
    elif not join_eligible:
        reason = "join_ineligible_requires_split_or_overlap"
    elif not join_present:
        reason = "correct_join_candidate_missing"
    elif not join_selected:
        reason = "correct_join_present_solver_rejected"
    else:
        reason = "correct_join_selected"
    failure_class = _reference_break_failure_class(
        reason,
        edge_key=edge_key,
        selected_successor_by_source=selected_successor_by_source,
        selected_predecessor_by_target=selected_predecessor_by_target,
    )
    missing_candidate_audit = _missing_candidate_audit(
        sessions,
        source_tracklet,
        target_tracklet,
        feature_cache=feature_cache,
        matrix_config=matrix_config,
        graph_config=graph_config,
        expanded_sources=expanded_sources,
        tracklets_by_start=tracklets_by_start,
        candidate_edges=candidate_edges,
        edge_key=edge_key,
        break_reason=reason,
        n_sessions=n_sessions,
    )
    return {
        "row_type": "reference_break",
        "subject": subject,
        "reference_track_index": int(reference_index),
        "source_session": int(source_session),
        "source_roi": int(source_roi),
        "target_session": int(target_session),
        "target_roi": int(target_roi),
        "source_tracklet": (
            "" if source_tracklet_id is None else int(source_tracklet_id)
        ),
        "target_tracklet": (
            "" if target_tracklet_id is None else int(target_tracklet_id)
        ),
        "source_tracklet_start": (
            "" if source_tracklet is None else int(source_tracklet.start_session)
        ),
        "source_tracklet_end": (
            "" if source_tracklet is None else int(source_tracklet.end_session)
        ),
        "target_tracklet_start": (
            "" if target_tracklet is None else int(target_tracklet.start_session)
        ),
        "target_tracklet_end": (
            "" if target_tracklet is None else int(target_tracklet.end_session)
        ),
        "correct_join_eligible": int(join_eligible),
        "correct_join_present": int(join_present),
        "correct_join_selected": int(join_selected),
        "solver_rejected_correct_join": int(join_present and not join_selected),
        "break_reason": reason,
        "failure_class": failure_class,
        **missing_candidate_audit,
    }


def _missing_candidate_audit(
    sessions: Sequence[Any],
    source_tracklet: Tracklet | None,
    target_tracklet: Tracklet | None,
    *,
    feature_cache: rank._FeatureCache | None,
    matrix_config: full_mht.FullMHTConfig | None,
    graph_config: TrackletGraphConfig | None,
    expanded_sources: set[int],
    tracklets_by_start: Mapping[int, Sequence[Tracklet]],
    candidate_edges: set[tuple[int, int]],
    edge_key: tuple[int, int],
    break_reason: str,
    n_sessions: int | None,
) -> dict[str, Any]:
    fields = _empty_missing_candidate_audit()
    if break_reason != "correct_join_candidate_missing":
        return fields
    if (
        sessions is None
        or feature_cache is None
        or matrix_config is None
        or graph_config is None
        or n_sessions is None
    ):
        return fields
    if source_tracklet is None or target_tracklet is None:
        return {**fields, "missing_candidate_primary_reason": "tracklet_uncovered"}

    gap = int(target_tracklet.start_session - source_tracklet.end_session)
    source_expanded = int(source_tracklet.tracklet_id) in expanded_sources
    within_gap = 1 <= gap <= max(1, int(graph_config.max_join_gap))
    fields.update(
        {
            "missing_candidate_source_expanded": int(source_expanded),
            "missing_candidate_gap": int(gap),
            "missing_candidate_within_max_join_gap": int(within_gap),
            "missing_candidate_source_cell_probability": full_mht._cell_probability(
                sessions, source_tracklet.end_session, int(source_tracklet.rois[-1])
            ),
            "missing_candidate_target_cell_probability": full_mht._cell_probability(
                sessions, target_tracklet.start_session, int(target_tracklet.rois[0])
            ),
        }
    )
    if not source_expanded:
        fields["missing_candidate_primary_reason"] = "source_not_reachable"
    elif not within_gap:
        fields["missing_candidate_primary_reason"] = "outside_max_join_gap"
        return fields

    matrices = full_mht._sparse_pair_matrices(
        sessions,
        feature_cache,
        source_session=int(source_tracklet.end_session),
        target_session=int(target_tracklet.start_session),
        source_rois=(int(source_tracklet.rois[-1]),),
        edge_top_k=max(1, int(graph_config.edge_top_k)),
        config=matrix_config,
    )
    source_lookup = {
        int(roi): index for index, roi in enumerate(np.asarray(matrices.source_indices))
    }
    target_lookup = {
        int(roi): index for index, roi in enumerate(np.asarray(matrices.target_indices))
    }
    source_local = source_lookup.get(int(source_tracklet.rois[-1]))
    target_local = target_lookup.get(int(target_tracklet.rois[0]))
    if source_local is None or source_local >= int(matrices.registered_iou.shape[0]):
        if source_expanded:
            fields["missing_candidate_primary_reason"] = "source_not_in_sparse_support"
        return fields
    if target_local is None:
        if fields["missing_candidate_primary_reason"] == "":
            if (
                float(fields["missing_candidate_target_cell_probability"])
                < float(feature_cache.cell_probability_threshold)
            ):
                reason = "target_cell_probability_below_threshold"
            else:
                reason = "target_not_in_sparse_support"
            fields["missing_candidate_primary_reason"] = reason
        return fields

    score_matrix = _score_matrix(
        sessions,
        matrices,
        target_session=int(target_tracklet.start_session),
        matrix_config=matrix_config,
    )
    raw_score = float(score_matrix[int(source_local), int(target_local)])
    fields.update(
        {
            "missing_candidate_target_in_sparse_support": 1,
            "missing_candidate_raw_score": raw_score,
            "missing_candidate_registered_iou": float(
                matrices.registered_iou[int(source_local), int(target_local)]
            ),
            "missing_candidate_shifted_iou": float(
                matrices.shifted_iou[int(source_local), int(target_local)]
            ),
            "missing_candidate_centroid_distance": float(
                matrices.centroid_distance[int(source_local), int(target_local)]
            ),
            "missing_candidate_area_ratio": float(
                matrices.area_ratio[int(source_local), int(target_local)]
            ),
            "missing_candidate_growth_residual": float(
                matrices.growth_residual[int(source_local), int(target_local)]
            ),
            "missing_candidate_growth_mahalanobis": float(
                matrices.growth_mahalanobis[int(source_local), int(target_local)]
            ),
            "missing_candidate_endpoint_cell_probability_min": min(
                float(fields["missing_candidate_source_cell_probability"]),
                float(fields["missing_candidate_target_cell_probability"]),
            ),
        }
    )
    gate_failures = _missing_candidate_gate_failures(fields, graph_config)
    fields["missing_candidate_gate_failures"] = ";".join(gate_failures)
    if fields["missing_candidate_primary_reason"] == "" and gate_failures:
        fields["missing_candidate_primary_reason"] = gate_failures[0]

    valid_edges = _candidate_edges_for_source(
        sessions,
        source_tracklet,
        tracklets_by_start=tracklets_by_start,
        feature_cache=feature_cache,
        matrix_config=matrix_config,
        graph_config=graph_config,
        n_sessions=int(n_sessions),
    )
    fields["missing_candidate_valid_candidate_count_for_source"] = len(valid_edges)
    for rank_index, edge in enumerate(valid_edges, start=1):
        if int(edge.target_id) != int(target_tracklet.tracklet_id):
            continue
        fields["missing_candidate_rank_for_source"] = int(rank_index)
        fields["missing_candidate_candidate_score"] = float(edge.score)
        break
    if fields["missing_candidate_primary_reason"] == "":
        rank_value = fields["missing_candidate_rank_for_source"]
        if rank_value != "" and int(rank_value) > max(1, int(graph_config.edge_top_k)):
            fields["missing_candidate_primary_reason"] = "lost_by_edge_top_k"
        elif edge_key not in candidate_edges:
            fields["missing_candidate_primary_reason"] = "materialization_unexpected"
    return fields


def _empty_missing_candidate_audit() -> dict[str, Any]:
    return {
        "missing_candidate_primary_reason": "",
        "missing_candidate_gate_failures": "",
        "missing_candidate_source_expanded": "",
        "missing_candidate_gap": "",
        "missing_candidate_within_max_join_gap": "",
        "missing_candidate_target_in_sparse_support": "",
        "missing_candidate_source_cell_probability": "",
        "missing_candidate_target_cell_probability": "",
        "missing_candidate_endpoint_cell_probability_min": "",
        "missing_candidate_registered_iou": "",
        "missing_candidate_shifted_iou": "",
        "missing_candidate_centroid_distance": "",
        "missing_candidate_area_ratio": "",
        "missing_candidate_growth_residual": "",
        "missing_candidate_growth_mahalanobis": "",
        "missing_candidate_raw_score": "",
        "missing_candidate_candidate_score": "",
        "missing_candidate_rank_for_source": "",
        "missing_candidate_valid_candidate_count_for_source": "",
    }


def _missing_candidate_gate_failures(
    fields: Mapping[str, Any], graph_config: TrackletGraphConfig
) -> list[str]:
    failures: list[str] = []
    if float(fields["missing_candidate_registered_iou"]) < float(
        graph_config.join_min_registered_iou
    ):
        failures.append("registered_iou_below_gate")
    if float(fields["missing_candidate_shifted_iou"]) < float(
        graph_config.join_min_shifted_iou
    ):
        failures.append("shifted_iou_below_gate")
    if float(fields["missing_candidate_area_ratio"]) < float(
        graph_config.join_min_area_ratio
    ):
        failures.append("area_ratio_below_gate")
    if float(fields["missing_candidate_centroid_distance"]) > float(
        graph_config.join_max_centroid_distance
    ):
        failures.append("centroid_distance_above_gate")
    if float(fields["missing_candidate_growth_residual"]) > float(
        graph_config.join_max_growth_residual
    ):
        failures.append("growth_residual_above_gate")
    if float(fields["missing_candidate_raw_score"]) < float(
        graph_config.join_min_edge_score
    ):
        failures.append("score_below_gate")
    return failures


def _reference_break_failure_class(
    reason: str,
    *,
    edge_key: tuple[int, int],
    selected_successor_by_source: Mapping[int, int],
    selected_predecessor_by_target: Mapping[int, int],
) -> str:
    if reason == "correct_join_selected":
        return "recovered_correct_join"
    if reason in {
        "source_observation_uncovered",
        "target_observation_uncovered",
        "join_ineligible_requires_split_or_overlap",
    }:
        return "tracklet_builder_too_strict"
    if reason == "correct_join_candidate_missing":
        return "graph_candidate_missing"
    if reason == "correct_join_present_solver_rejected":
        source_id, target_id = edge_key
        selected_target = selected_successor_by_source.get(int(source_id))
        selected_source = selected_predecessor_by_target.get(int(target_id))
        if (
            selected_target is not None
            and int(selected_target) != int(target_id)
        ) or (
            selected_source is not None
            and int(selected_source) != int(source_id)
        ):
            return "conflict_issue"
        return "solver_too_conservative"
    return "unclassified"


def _tracklet_fragment_count(tracklet_ids: Sequence[int | None]) -> int:
    fragments = 0
    previous: int | None = None
    for tracklet_id in tracklet_ids:
        if tracklet_id is None:
            previous = None
            continue
        current = int(tracklet_id)
        if previous is None or current != previous:
            fragments += 1
        previous = current
    return int(fragments)


def _accumulate_coverage_summary(
    summary: dict[str, int],
    track_row: Mapping[str, Any],
    break_rows: Sequence[Mapping[str, Any]],
) -> None:
    summary["tracklet_graph_audit_reference_tracks_with_seed_observation"] += int(
        track_row.get("has_seed_observation", 0)
    )
    summary["tracklet_graph_audit_reference_tracks_with_seed_tracklet"] += int(
        track_row.get("has_seed_tracklet", 0)
    )
    summary["tracklet_graph_audit_reference_links"] += int(
        track_row.get("reference_links", 0)
    )
    summary["tracklet_graph_audit_reference_links_preserved_in_tracklets"] += int(
        track_row.get("reference_links_preserved_in_tracklets", 0)
    )
    summary["tracklet_graph_audit_reference_breaks"] += int(
        track_row.get("reference_breaks", 0)
    )
    fragment_count = int(track_row.get("covered_tracklet_count", 0))
    if fragment_count <= 0:
        summary["tracklet_graph_audit_tracks_split_0"] += 1
    elif fragment_count == 1:
        summary["tracklet_graph_audit_tracks_split_1"] += 1
    elif fragment_count == 2:
        summary["tracklet_graph_audit_tracks_split_2"] += 1
    elif fragment_count == 3:
        summary["tracklet_graph_audit_tracks_split_3"] += 1
    else:
        summary["tracklet_graph_audit_tracks_split_4plus"] += 1
    for row in break_rows:
        summary["tracklet_graph_audit_break_correct_join_present"] += int(
            row.get("correct_join_present", 0)
        )
        summary["tracklet_graph_audit_break_correct_join_selected"] += int(
            row.get("correct_join_selected", 0)
        )
        summary["tracklet_graph_audit_break_solver_rejected"] += int(
            row.get("solver_rejected_correct_join", 0)
        )
        reason = str(row.get("break_reason", ""))
        if reason == "correct_join_candidate_missing":
            summary["tracklet_graph_audit_break_candidate_missing"] += 1
        elif reason == "join_ineligible_requires_split_or_overlap":
            summary["tracklet_graph_audit_break_join_ineligible"] += 1
        failure_class = str(row.get("failure_class", ""))
        if failure_class == "tracklet_builder_too_strict":
            summary["tracklet_graph_audit_failure_tracklet_builder_too_strict"] += 1
        elif failure_class == "graph_candidate_missing":
            summary["tracklet_graph_audit_failure_graph_candidate_missing"] += 1
        elif failure_class == "solver_too_conservative":
            summary["tracklet_graph_audit_failure_solver_too_conservative"] += 1
        elif failure_class == "conflict_issue":
            summary["tracklet_graph_audit_failure_conflict_issue"] += 1
        elif failure_class == "recovered_correct_join":
            summary["tracklet_graph_audit_recovered_correct_joins"] += 1
        missing_reason = str(row.get("missing_candidate_primary_reason", ""))
        if not missing_reason:
            continue
        _increment_missing_candidate_summary(summary, missing_reason)


def _increment_missing_candidate_summary(
    summary: dict[str, int], missing_reason: str
) -> None:
    key_by_reason = {
        "source_not_reachable": "tracklet_graph_audit_missing_source_not_reachable",
        "outside_max_join_gap": "tracklet_graph_audit_missing_outside_max_join_gap",
        "source_not_in_sparse_support": (
            "tracklet_graph_audit_missing_source_not_in_sparse_support"
        ),
        "target_cell_probability_below_threshold": (
            "tracklet_graph_audit_missing_target_cell_probability_below_threshold"
        ),
        "target_not_in_sparse_support": (
            "tracklet_graph_audit_missing_target_not_in_sparse_support"
        ),
        "registered_iou_below_gate": (
            "tracklet_graph_audit_missing_registered_iou_below_gate"
        ),
        "shifted_iou_below_gate": (
            "tracklet_graph_audit_missing_shifted_iou_below_gate"
        ),
        "area_ratio_below_gate": (
            "tracklet_graph_audit_missing_area_ratio_below_gate"
        ),
        "centroid_distance_above_gate": (
            "tracklet_graph_audit_missing_centroid_distance_above_gate"
        ),
        "growth_residual_above_gate": (
            "tracklet_graph_audit_missing_growth_residual_above_gate"
        ),
        "score_below_gate": "tracklet_graph_audit_missing_score_below_gate",
        "lost_by_edge_top_k": "tracklet_graph_audit_missing_lost_by_edge_top_k",
        "materialization_unexpected": (
            "tracklet_graph_audit_missing_materialization_unexpected"
        ),
    }
    key = key_by_reason.get(
        missing_reason, "tracklet_graph_audit_missing_unclassified"
    )
    summary[key] += 1


def _edge_feature_row(
    row_type: str,
    subject: str,
    *,
    source_session: int,
    source_roi: int,
    target_session: int,
    target_roi: int,
    score: float,
    matrices: full_mht._FullMHTPairMatrices,
    row_index: int,
    col_index: int,
    selected: bool,
) -> dict[str, Any]:
    return {
        "row_type": row_type,
        "subject": subject,
        "source_session": int(source_session),
        "source_roi": int(source_roi),
        "target_session": int(target_session),
        "target_roi": int(target_roi),
        "score": float(score),
        "registered_iou": float(matrices.registered_iou[row_index, col_index]),
        "shifted_iou": float(matrices.shifted_iou[row_index, col_index]),
        "centroid_distance": float(matrices.centroid_distance[row_index, col_index]),
        "area_ratio": float(matrices.area_ratio[row_index, col_index]),
        "growth_residual": float(matrices.growth_residual[row_index, col_index]),
        "growth_residual_mahalanobis": float(
            matrices.growth_mahalanobis[row_index, col_index]
        ),
        "selected": int(bool(selected)),
    }


def _tracklet_diagnostic_rows(
    subject: str, tracklets: Sequence[Tracklet]
) -> list[dict[str, Any]]:
    return [
        {
            "row_type": "tracklet",
            "subject": subject,
            "tracklet_id": int(tracklet.tracklet_id),
            "start_session": int(tracklet.start_session),
            "end_session": int(tracklet.end_session),
            "length": int(tracklet.length),
            "roi_sequence": " ".join(str(int(roi)) for roi in tracklet.rois),
        }
        for tracklet in tracklets
    ]


def _tracklet_edge_row(
    subject: str,
    edge: TrackletEdge,
    tracklets_by_id: Mapping[int, Tracklet],
) -> dict[str, Any]:
    source = tracklets_by_id[int(edge.source_id)]
    target = tracklets_by_id[int(edge.target_id)]
    return {
        "row_type": "tracklet_join_candidate",
        "subject": subject,
        "source_tracklet": int(edge.source_id),
        "target_tracklet": int(edge.target_id),
        "source_session": int(source.end_session),
        "source_roi": int(source.rois[-1]),
        "target_session": int(target.start_session),
        "target_roi": int(target.rois[0]),
        "gap": int(edge.gap),
        "score": float(edge.score),
        "raw_score": float(edge.raw_score),
        "registered_iou": float(edge.registered_iou),
        "shifted_iou": float(edge.shifted_iou),
        "centroid_distance": float(edge.centroid_distance),
        "area_ratio": float(edge.area_ratio),
        "growth_residual": float(edge.growth_residual),
        "growth_residual_mahalanobis": float(edge.growth_mahalanobis),
        "endpoint_cell_probability_min": float(edge.endpoint_cell_probability_min),
        "duplicate_source_rank": int(edge.duplicate_source_rank),
        "duplicate_target_rank": int(edge.duplicate_target_rank),
        "would_complete_track": int(bool(edge.would_complete_track)),
        "suspicious_complete_component": int(bool(edge.suspicious_complete_component)),
        "component_incoherence": float(edge.component_incoherence),
    }


def _path_diagnostic_rows(
    subject: str, paths: Sequence[_PathHypothesis]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path_index, path in enumerate(paths, start=1):
        rows.append(
            {
                "row_type": "selected_path",
                "subject": subject,
                "path_rank": int(path_index),
                "score": float(path.score),
                "tracklet_ids": " ".join(str(int(item)) for item in path.tracklet_ids),
                "edge_ids": " ".join(
                    f"{int(source)}->{int(target)}" for source, target in path.edge_ids
                ),
                "n_joins": int(len(path.edge_ids)),
            }
        )
    return rows


def _all_summary_row(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"subject": "ALL"}
    return {
        "subject": "ALL",
        "n_seed_tracks": int(sum(int(row.get("n_seed_tracks", 0)) for row in rows)),
        "n_tracklets": int(sum(int(row.get("n_tracklets", 0)) for row in rows)),
        "n_candidate_edges": int(
            sum(int(row.get("n_candidate_edges", 0)) for row in rows)
        ),
        "n_selected_joins": int(
            sum(int(row.get("n_selected_joins", 0)) for row in rows)
        ),
        "best_score": float(sum(float(row.get("best_score", 0.0)) for row in rows)),
    }


def _write_rows(
    rows: Sequence[Mapping[str, Any]],
    output: Path,
    *,
    output_format: Literal["csv", "json"],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output.write_text(json.dumps(list(rows), indent=2), encoding="utf-8")
        return
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    import csv

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the tracklet-graph benchmark parser."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-policy-tracklet-graph-mht",
        description="Run conservative tracklets plus bounded graph-MHT selection.",
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
        "--threshold-method",
        choices=("otsu", "min"),
        default=TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    )
    parser.add_argument(
        "--transform-type", default=TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE
    )
    parser.add_argument(
        "--iou-distance-threshold",
        type=float,
        default=TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    )
    parser.add_argument(
        "--cell-probability-threshold",
        type=float,
        default=TRACK2P_POLICY_DEFAULT_CELL_PROBABILITY_THRESHOLD,
    )
    parser.add_argument("--seed-session", type=int, default=0)
    parser.add_argument(
        "--restrict-to-reference-seed-rois",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--allow-track2p-as-reference-for-smoke-test", action="store_true"
    )
    parser.add_argument(
        "--include-behavior", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--seed-source", choices=("reference", "all-cells"), default="reference"
    )
    parser.add_argument("--max-seed-tracks", type=int, default=None)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--path-hypotheses", type=int, default=16)
    parser.add_argument("--edge-top-k", type=int, default=4)
    parser.add_argument("--max-join-gap", type=int, default=3)
    parser.add_argument("--local-min-edge-score", type=float, default=1.25)
    parser.add_argument("--local-min-registered-iou", type=float, default=0.45)
    parser.add_argument("--local-min-shifted-iou", type=float, default=0.35)
    parser.add_argument("--local-min-area-ratio", type=float, default=0.70)
    parser.add_argument("--local-max-centroid-distance", type=float, default=8.0)
    parser.add_argument("--local-max-growth-residual", type=float, default=3.5)
    parser.add_argument("--local-min-ambiguity-margin", type=float, default=0.15)
    parser.add_argument("--join-min-edge-score", type=float, default=0.75)
    parser.add_argument("--join-min-registered-iou", type=float, default=0.20)
    parser.add_argument("--join-min-shifted-iou", type=float, default=0.20)
    parser.add_argument("--join-min-area-ratio", type=float, default=0.55)
    parser.add_argument("--join-max-centroid-distance", type=float, default=14.0)
    parser.add_argument("--join-max-growth-residual", type=float, default=8.0)
    parser.add_argument("--join-score-frontier-min-edge-score", type=float, default=0.0)
    parser.add_argument(
        "--join-score-frontier-min-registered-iou", type=float, default=0.30
    )
    parser.add_argument(
        "--join-score-frontier-min-shifted-iou", type=float, default=0.50
    )
    parser.add_argument(
        "--join-score-frontier-min-area-ratio", type=float, default=0.65
    )
    parser.add_argument(
        "--join-score-frontier-max-centroid-distance", type=float, default=4.5
    )
    parser.add_argument(
        "--join-score-frontier-max-growth-residual", type=float, default=4.5
    )
    parser.add_argument("--join-complexity-penalty", type=float, default=0.35)
    parser.add_argument("--gap-penalty", type=float, default=0.20)
    parser.add_argument("--component-incoherence-weight", type=float, default=0.25)
    parser.add_argument("--registered-iou-weight", type=float, default=1.0)
    parser.add_argument("--shifted-iou-weight", type=float, default=1.5)
    parser.add_argument("--area-ratio-weight", type=float, default=0.25)
    parser.add_argument("--cell-probability-weight", type=float, default=0.25)
    parser.add_argument("--centroid-distance-weight", type=float, default=0.05)
    parser.add_argument("--threshold-margin-weight", type=float, default=0.50)
    parser.add_argument("--growth-residual-weight", type=float, default=0.10)
    parser.add_argument("--growth-mahalanobis-weight", type=float, default=0.25)
    parser.add_argument("--growth-anchor-min-registered-iou", type=float, default=0.55)
    parser.add_argument("--growth-anchor-min-shifted-iou", type=float, default=0.30)
    parser.add_argument(
        "--growth-anchor-min-cell-probability", type=float, default=0.80
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=("table", "json", "csv"), default="csv")
    parser.add_argument("--diagnostics-output", type=Path, default=None)
    parser.add_argument("--diagnostics-format", choices=("csv", "json"), default="csv")
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument(
        "--progress", action=argparse.BooleanOptionalAction, default=False
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the tracklet-hypothesis graph benchmark CLI."""

    args = build_arg_parser().parse_args(argv)
    benchmark_config = Track2pBenchmarkConfig(
        data=args.data,
        method="global-assignment",
        input_format=args.input_format,
        reference=args.reference,
        reference_kind=args.reference_kind,
        plane_name=args.plane_name,
        seed_session=args.seed_session,
        restrict_to_reference_seed_rois=args.restrict_to_reference_seed_rois,
        transform_type=args.transform_type,
        allow_track2p_as_reference_for_smoke_test=(
            args.allow_track2p_as_reference_for_smoke_test
        ),
        include_behavior=args.include_behavior,
        include_non_cells=False,
        cell_probability_threshold=args.cell_probability_threshold,
        exclude_overlapping_pixels=False,
        weighted_masks=False,
        weighted_centroids=False,
    )
    result = run_track2p_policy_tracklet_graph_mht(
        benchmark_config,
        threshold_method=cast(ThresholdMethod, args.threshold_method),
        iou_distance_threshold=float(args.iou_distance_threshold),
        transform_type=args.transform_type,
        cell_probability_threshold=float(args.cell_probability_threshold),
        graph_config=TrackletGraphConfig(
            seed_source=cast(SeedSource, args.seed_source),
            max_seed_tracks=args.max_seed_tracks,
            beam_width=max(1, int(args.beam_width)),
            path_hypotheses=max(1, int(args.path_hypotheses)),
            edge_top_k=max(1, int(args.edge_top_k)),
            max_join_gap=max(1, int(args.max_join_gap)),
            local_min_edge_score=float(args.local_min_edge_score),
            local_min_registered_iou=float(args.local_min_registered_iou),
            local_min_shifted_iou=float(args.local_min_shifted_iou),
            local_min_area_ratio=float(args.local_min_area_ratio),
            local_max_centroid_distance=float(args.local_max_centroid_distance),
            local_max_growth_residual=float(args.local_max_growth_residual),
            local_min_ambiguity_margin=float(args.local_min_ambiguity_margin),
            join_min_edge_score=float(args.join_min_edge_score),
            join_min_registered_iou=float(args.join_min_registered_iou),
            join_min_shifted_iou=float(args.join_min_shifted_iou),
            join_min_area_ratio=float(args.join_min_area_ratio),
            join_max_centroid_distance=float(args.join_max_centroid_distance),
            join_max_growth_residual=float(args.join_max_growth_residual),
            join_score_frontier_min_edge_score=float(
                args.join_score_frontier_min_edge_score
            ),
            join_score_frontier_min_registered_iou=float(
                args.join_score_frontier_min_registered_iou
            ),
            join_score_frontier_min_shifted_iou=float(
                args.join_score_frontier_min_shifted_iou
            ),
            join_score_frontier_min_area_ratio=float(
                args.join_score_frontier_min_area_ratio
            ),
            join_score_frontier_max_centroid_distance=float(
                args.join_score_frontier_max_centroid_distance
            ),
            join_score_frontier_max_growth_residual=float(
                args.join_score_frontier_max_growth_residual
            ),
            join_complexity_penalty=float(args.join_complexity_penalty),
            gap_penalty=float(args.gap_penalty),
            component_incoherence_weight=float(args.component_incoherence_weight),
            registered_iou_weight=float(args.registered_iou_weight),
            shifted_iou_weight=float(args.shifted_iou_weight),
            area_ratio_weight=float(args.area_ratio_weight),
            cell_probability_weight=float(args.cell_probability_weight),
            centroid_distance_weight=float(args.centroid_distance_weight),
            threshold_margin_weight=float(args.threshold_margin_weight),
            growth_residual_weight=float(args.growth_residual_weight),
            growth_mahalanobis_weight=float(args.growth_mahalanobis_weight),
            growth_anchor_min_registered_iou=float(
                args.growth_anchor_min_registered_iou
            ),
            growth_anchor_min_shifted_iou=float(args.growth_anchor_min_shifted_iou),
            growth_anchor_min_cell_probability=float(
                args.growth_anchor_min_cell_probability
            ),
        ),
        progress=bool(args.progress),
    )
    write_results(
        [benchmark_result.to_dict() for benchmark_result in result.results],
        args.output,
        cast(OutputFormat, args.format),
    )
    if args.diagnostics_output is not None:
        _write_rows(
            result.diagnostic_rows,
            args.diagnostics_output,
            output_format=cast(Literal["csv", "json"], args.diagnostics_format),
        )
    if args.summary_output is not None:
        _write_rows(
            result.summary_rows,
            args.summary_output,
            output_format=cast(Literal["csv", "json"], args.diagnostics_format),
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
