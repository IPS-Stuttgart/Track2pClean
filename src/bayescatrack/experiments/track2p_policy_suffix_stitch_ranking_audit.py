"""Rank suffix-stitch candidates from non-GT features.

This diagnostic starts from Track2pPolicy ComponentCleanup and asks whether
short missing suffixes, especially the jm039 two-edge suffix, are naturally
discoverable before adding any new stitcher.  Candidate paths are ranked from
registration, shape, margin, cell-probability, and optional activity features;
manual GT is used only to label candidate paths and edges after ranking.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from bayescatrack.association.activity_similarity import activity_similarity_components
from bayescatrack.association.shifted_overlap import _pairwise_shifted_iou_from_support
from bayescatrack.core.bridge import Track2pSession
from bayescatrack.experiments.track2p_benchmark import (
    GROUND_TRUTH_REFERENCE_SOURCE,
    Track2pBenchmarkConfig,
    _load_reference_for_subject,
    _load_subject_sessions,
    _reference_matrix,
    _validate_reference_for_benchmark,
    _validate_reference_roi_indices,
    discover_subject_dirs,
)
from bayescatrack.experiments.track2p_emulation_benchmark import ThresholdMethod
from bayescatrack.experiments.track2p_policy_audit import TrackEdge
from bayescatrack.experiments.track2p_policy_benchmark import (
    TRACK2P_POLICY_DEFAULT_CELL_PROBABILITY_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE,
    track2p_policy_config,
)
from bayescatrack.experiments.track2p_policy_component_audit import (
    ComponentCleanupConfig,
    _evaluated_prediction_rows,
    _mark_applied_splits,
    _normalize_int_track_matrix,
    apply_weakest_bridge_splits,
    component_audit_rows,
)
from bayescatrack.experiments.track2p_policy_component_residual_audit import (
    _cell_probability,
    _no_prune_config,
)
from bayescatrack.experiments.track2p_policy_pruned_benchmark import (
    Track2pPolicyLinkDiagnostic,
    _margin_against_competitor,
    _roi_indices,
    _threshold_assigned_iou,
    _track2p_cross_iou_diagnostic_matrices,
    emulate_track2p_pruned_tracks,
)
from bayescatrack.track2p_registration import register_plane_pair
from scipy.optimize import linear_sum_assignment

TRACK2P_POLICY_SUFFIX_STITCH_RANKING_AUDIT_METHOD = (
    "track2p-policy-suffix-stitch-ranking-audit"
)
CompleteTrack = tuple[int, ...]


def _integral_value(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc
    raise ValueError(f"{name} must be an integer")


def _positive_int_value(value: Any, *, name: str) -> int:
    numeric = _integral_value(value, name=name)
    if numeric <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return numeric


def _positive_int_arg(value: str) -> int:
    try:
        return _positive_int_value(value, name="value")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


@dataclass(frozen=True)
class SuffixStitchRankingAuditResult:
    """Path, edge, and summary rows for suffix-stitch ranking."""

    edge_rows: tuple[dict[str, float | int | str], ...]
    path_rows: tuple[dict[str, float | int | str], ...]
    summary_rows: tuple[dict[str, float | int | str], ...]


@dataclass(frozen=True)
class _PairMatrices:
    source_indices: np.ndarray
    target_indices: np.ndarray
    registered_iou: np.ndarray
    shifted_iou: np.ndarray
    centroid_distance: np.ndarray
    area_ratio: np.ndarray
    activity_similarity: np.ndarray
    threshold: float


@dataclass(frozen=True)
class _EdgeCandidate:
    edge: TrackEdge
    registered_iou: float
    shifted_iou: float
    roi_aware_score: float
    centroid_distance: float
    area_ratio: float
    cell_probability_a: float
    cell_probability_b: float
    row_rank: int
    column_rank: int
    row_margin: float
    column_margin: float
    threshold_margin: float
    activity_similarity: float
    edge_score: float


@dataclass(frozen=True)
class _PathCandidate:
    component_id: int
    fragment_row: tuple[int, ...]
    fragment_span: str
    edges: tuple[_EdgeCandidate, ...]
    path_score: float = float("nan")
    path_rank: int = -1
    is_gt_suffix_path: int = 0


@dataclass
class _FeatureCache:
    sessions: Sequence[Track2pSession]
    transform_type: str
    threshold_method: ThresholdMethod
    iou_distance_threshold: float
    cell_probability_threshold: float
    matrices: dict[int, _PairMatrices]

    def pair(self, session_index: int) -> _PairMatrices:
        cached = self.matrices.get(int(session_index))
        if cached is not None:
            return cached
        reference_session = self.sessions[int(session_index)]
        moving_session = self.sessions[int(session_index) + 1]
        registered = register_plane_pair(
            reference_session.plane_data,
            moving_session.plane_data,
            transform_type=self.transform_type,
        )
        reference_masks = np.asarray(reference_session.plane_data.roi_masks) > 0
        moving_masks = np.asarray(registered.roi_masks) > 0
        registered_iou, distances, area_ratios = _track2p_cross_iou_diagnostic_matrices(
            reference_masks,
            moving_masks,
            distance_threshold=float(self.iou_distance_threshold),
        )
        shifted = _pairwise_shifted_iou_from_support(
            reference_masks, moving_masks, radius=2
        )["shifted_iou"]
        threshold = _assignment_threshold(
            registered_iou, threshold_method=self.threshold_method
        )
        activity = _activity_similarity_matrix(reference_session, moving_session)
        matrices = _PairMatrices(
            source_indices=_roi_indices(reference_session),
            target_indices=_roi_indices(moving_session),
            registered_iou=registered_iou,
            shifted_iou=shifted,
            centroid_distance=distances,
            area_ratio=area_ratios,
            activity_similarity=activity,
            threshold=float(threshold),
        )
        self.matrices[int(session_index)] = matrices
        return matrices


def run_track2p_policy_suffix_stitch_ranking_audit(
    config: Track2pBenchmarkConfig,
    *,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    iou_distance_threshold: float = TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    transform_type: str | None = None,
    cell_probability_threshold: float | None = None,
    cleanup_config: ComponentCleanupConfig | None = None,
    max_suffix_length: int = 2,
    edge_top_k: int = 25,
    path_beam_width: int = 100,
) -> SuffixStitchRankingAuditResult:
    """Rank short suffix-stitch candidates after ComponentCleanup."""

    max_suffix_length = _positive_int_value(
        max_suffix_length, name="max_suffix_length"
    )
    edge_top_k = _positive_int_value(edge_top_k, name="edge_top_k")
    path_beam_width = _positive_int_value(
        path_beam_width, name="path_beam_width"
    )
    policy_config = track2p_policy_config(
        config,
        transform_type=transform_type,
        cell_probability_threshold=cell_probability_threshold,
    )
    subject_dirs = discover_subject_dirs(policy_config.data)
    if not subject_dirs:
        raise ValueError(
            f"No Track2p-style subject directories found under {policy_config.data}"
        )
    cleanup_config = cleanup_config or ComponentCleanupConfig()
    all_edge_rows: list[dict[str, float | int | str]] = []
    all_path_rows: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | int | str]] = []
    for subject_dir in subject_dirs:
        edge_rows, path_rows = _subject_suffix_rows(
            subject_dir,
            config=policy_config,
            cleanup_config=cleanup_config,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
            max_suffix_length=max_suffix_length,
            edge_top_k=edge_top_k,
            path_beam_width=path_beam_width,
        )
        all_edge_rows.extend(edge_rows)
        all_path_rows.extend(path_rows)
        summary_rows.append(_summary_row(subject_dir.name, path_rows))
    summary_rows.append(_summary_row("ALL", all_path_rows))
    return SuffixStitchRankingAuditResult(
        tuple(all_edge_rows), tuple(all_path_rows), tuple(summary_rows)
    )


def _subject_suffix_rows(
    subject_dir: Path,
    *,
    config: Track2pBenchmarkConfig,
    cleanup_config: ComponentCleanupConfig,
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
    max_suffix_length: int,
    edge_top_k: int,
    path_beam_width: int,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    reference = _load_reference_for_subject(
        subject_dir, data_root=config.data, config=config
    )
    _validate_reference_for_benchmark(reference, subject_dir=subject_dir, config=config)
    if reference.source != GROUND_TRUTH_REFERENCE_SOURCE:
        raise ValueError(
            "Track2p-policy suffix-stitch ranking audit requires independent manual GT references"
        )
    sessions = _load_subject_sessions(subject_dir, config)
    _validate_reference_roi_indices(reference, sessions)
    reference_tracks = _reference_matrix(reference, curated_only=config.curated_only)
    prediction = emulate_track2p_pruned_tracks(
        sessions,
        transform_type=config.transform_type,
        threshold_method=threshold_method,
        iou_distance_threshold=float(iou_distance_threshold),
        prune_config=_no_prune_config(),
    )
    policy_full = _normalize_int_track_matrix(prediction.tracks)
    policy_eval, reference_eval, evaluated_track_ids = _evaluated_prediction_rows(
        policy_full, reference_tracks, config=config
    )
    audit_rows = component_audit_rows(
        policy_eval,
        reference_eval,
        sessions=sessions,
        diagnostics=cast(Sequence[Track2pPolicyLinkDiagnostic], prediction.diagnostics),
        subject=subject_dir.name,
        config=cleanup_config,
        track_ids=evaluated_track_ids,
        seed_session=config.seed_session,
    )
    cleaned_full = apply_weakest_bridge_splits(
        policy_full, _mark_applied_splits(audit_rows, apply_splits=True)
    )
    cleaned_eval, reference_eval, _ = _evaluated_prediction_rows(
        cleaned_full, reference_tracks, config=config
    )
    feature_cache = _FeatureCache(
        sessions=sessions,
        transform_type=str(config.transform_type),
        threshold_method=threshold_method,
        iou_distance_threshold=float(iou_distance_threshold),
        cell_probability_threshold=float(config.cell_probability_threshold),
        matrices={},
    )
    paths = _ranked_suffix_paths(
        cleaned_eval,
        reference_eval,
        subject=subject_dir.name,
        feature_cache=feature_cache,
        max_suffix_length=max_suffix_length,
        edge_top_k=edge_top_k,
        path_beam_width=path_beam_width,
    )
    edge_rows = _edge_rows(subject_dir.name, paths)
    path_rows = [
        _path_row(subject_dir.name, path, cleaned_eval, reference_eval)
        for path in paths
    ]
    return edge_rows, path_rows


def _ranked_suffix_paths(
    predicted: np.ndarray,
    reference: np.ndarray,
    *,
    subject: str,
    feature_cache: _FeatureCache,
    max_suffix_length: int,
    edge_top_k: int,
    path_beam_width: int,
) -> tuple[_PathCandidate, ...]:
    del subject
    max_suffix_length = _positive_int_value(
        max_suffix_length, name="max_suffix_length"
    )
    edge_top_k = _positive_int_value(edge_top_k, name="edge_top_k")
    path_beam_width = _positive_int_value(
        path_beam_width, name="path_beam_width"
    )
    paths_by_component: dict[int, list[_PathCandidate]] = defaultdict(list)
    for component_id, row in enumerate(predicted):
        span = _suffix_fragment_span(row)
        if span is None:
            continue
        start_session, tail_session = span
        if tail_session >= row.size - 1:
            continue
        if row.size - tail_session - 1 > int(max_suffix_length):
            continue
        if tail_session - start_session + 1 < 2:
            continue
        max_steps = min(int(max_suffix_length), int(row.size - tail_session - 1))
        paths_by_component[int(component_id)].extend(
            _expand_paths_for_fragment(
                component_id=int(component_id),
                row=tuple(int(value) for value in row),
                fragment_span=f"{start_session}-{tail_session}",
                tail_session=int(tail_session),
                feature_cache=feature_cache,
                max_steps=max_steps,
                edge_top_k=edge_top_k,
                path_beam_width=path_beam_width,
            )
        )

    ranked_paths: list[_PathCandidate] = []
    for component_id, paths in paths_by_component.items():
        labeled = [
            _label_gt_path(path, predicted[int(component_id)], reference)
            for path in paths
        ]
        ranked = _rank_paths(labeled)
        ranked_paths.extend(ranked)
    return tuple(ranked_paths)


def _expand_paths_for_fragment(
    *,
    component_id: int,
    row: tuple[int, ...],
    fragment_span: str,
    tail_session: int,
    feature_cache: _FeatureCache,
    max_steps: int,
    edge_top_k: int,
    path_beam_width: int,
) -> list[_PathCandidate]:
    max_steps = _positive_int_value(max_steps, name="max_steps")
    edge_top_k = _positive_int_value(edge_top_k, name="edge_top_k")
    path_beam_width = _positive_int_value(
        path_beam_width, name="path_beam_width"
    )
    active: list[tuple[int, int, tuple[_EdgeCandidate, ...]]] = [
        (int(tail_session), int(row[int(tail_session)]), ())
    ]
    output: list[_PathCandidate] = []
    for _step in range(int(max_steps)):
        expanded: list[tuple[int, int, tuple[_EdgeCandidate, ...]]] = []
        for session_index, roi, edges in active:
            for edge in _top_edge_candidates(
                feature_cache, session_index, roi, top_k=int(edge_top_k)
            ):
                next_edges = (*edges, edge)
                expanded.append((session_index + 1, edge.edge[3], next_edges))
                output.append(
                    _score_path(
                        _PathCandidate(
                            component_id=component_id,
                            fragment_row=row,
                            fragment_span=fragment_span,
                            edges=next_edges,
                        )
                    )
                )
        expanded.sort(key=lambda item: _mean_edge_score(item[2]), reverse=True)
        active = expanded[:path_beam_width]
    return output


def _top_edge_candidates(
    feature_cache: _FeatureCache, session_index: int, source_roi: int, *, top_k: int
) -> tuple[_EdgeCandidate, ...]:
    top_k = _positive_int_value(top_k, name="top_k")
    if session_index < 0 or session_index + 1 >= len(feature_cache.sessions):
        return ()
    matrices = feature_cache.pair(session_index)
    source_matches = np.flatnonzero(matrices.source_indices == int(source_roi))
    if source_matches.size == 0:
        return ()
    source_local = int(source_matches[0])
    candidates: list[_EdgeCandidate] = []
    for target_local, target_roi in enumerate(matrices.target_indices):
        cell_probability_b = _cell_probability(
            feature_cache.sessions, session_index + 1, int(target_roi)
        )
        if np.isfinite(cell_probability_b) and (
            cell_probability_b < feature_cache.cell_probability_threshold
        ):
            continue
        registered_iou = float(matrices.registered_iou[source_local, target_local])
        shifted_iou = float(matrices.shifted_iou[source_local, target_local])
        if max(registered_iou, shifted_iou) <= 0.0:
            continue
        edge = _edge_candidate(
            feature_cache,
            matrices,
            session_index=session_index,
            source_local=source_local,
            target_local=int(target_local),
        )
        candidates.append(edge)
    candidates.sort(key=lambda edge: edge.edge_score, reverse=True)
    return tuple(candidates[:top_k])


def _edge_candidate(
    feature_cache: _FeatureCache,
    matrices: _PairMatrices,
    *,
    session_index: int,
    source_local: int,
    target_local: int,
) -> _EdgeCandidate:
    roi_a = int(matrices.source_indices[int(source_local)])
    roi_b = int(matrices.target_indices[int(target_local)])
    registered_iou = float(
        matrices.registered_iou[int(source_local), int(target_local)]
    )
    shifted_iou = float(matrices.shifted_iou[int(source_local), int(target_local)])
    area_ratio = float(matrices.area_ratio[int(source_local), int(target_local)])
    centroid_distance = float(
        matrices.centroid_distance[int(source_local), int(target_local)]
    )
    row_margin = _margin_against_competitor(
        matrices.registered_iou[int(source_local), :], selected_index=int(target_local)
    )
    column_margin = _margin_against_competitor(
        matrices.registered_iou[:, int(target_local)], selected_index=int(source_local)
    )
    cell_probability_a = _cell_probability(
        feature_cache.sessions, int(session_index), roi_a
    )
    cell_probability_b = _cell_probability(
        feature_cache.sessions, int(session_index) + 1, roi_b
    )
    activity = float(matrices.activity_similarity[int(source_local), int(target_local)])
    roi_aware_score = float(registered_iou * area_ratio)
    edge_score = _edge_score(
        registered_iou=registered_iou,
        shifted_iou=shifted_iou,
        roi_aware_score=roi_aware_score,
        centroid_distance=centroid_distance,
        area_ratio=area_ratio,
        cell_probability_a=cell_probability_a,
        cell_probability_b=cell_probability_b,
        row_margin=row_margin,
        column_margin=column_margin,
        activity_similarity=activity,
    )
    return _EdgeCandidate(
        edge=(int(session_index), int(session_index) + 1, roi_a, roi_b),
        registered_iou=registered_iou,
        shifted_iou=shifted_iou,
        roi_aware_score=roi_aware_score,
        centroid_distance=centroid_distance,
        area_ratio=area_ratio,
        cell_probability_a=cell_probability_a,
        cell_probability_b=cell_probability_b,
        row_rank=_rank_descending(
            matrices.registered_iou[int(source_local), :],
            selected_index=int(target_local),
        ),
        column_rank=_rank_descending(
            matrices.registered_iou[:, int(target_local)],
            selected_index=int(source_local),
        ),
        row_margin=row_margin,
        column_margin=column_margin,
        threshold_margin=registered_iou - float(matrices.threshold),
        activity_similarity=activity,
        edge_score=edge_score,
    )


def _edge_score(
    *,
    registered_iou: float,
    shifted_iou: float,
    roi_aware_score: float,
    centroid_distance: float,
    area_ratio: float,
    cell_probability_a: float,
    cell_probability_b: float,
    row_margin: float,
    column_margin: float,
    activity_similarity: float,
) -> float:
    min_probability = _nanmin_default((cell_probability_a, cell_probability_b), 0.5)
    activity = (
        0.5 if not np.isfinite(activity_similarity) else float(activity_similarity)
    )
    return float(
        registered_iou
        + 0.50 * shifted_iou
        + 0.20 * roi_aware_score
        + 0.05 * area_ratio
        + 0.05 * min_probability
        + 0.05 * max(0.0, _finite_or(row_margin, 0.0))
        + 0.05 * max(0.0, _finite_or(column_margin, 0.0))
        + 0.05 * activity
        - 0.01 * _finite_or(centroid_distance, 100.0)
    )


def _rank_paths(paths: Sequence[_PathCandidate]) -> tuple[_PathCandidate, ...]:
    sorted_paths = sorted(paths, key=lambda path: path.path_score, reverse=True)
    return tuple(
        _PathCandidate(
            component_id=path.component_id,
            fragment_row=path.fragment_row,
            fragment_span=path.fragment_span,
            edges=path.edges,
            path_score=path.path_score,
            path_rank=rank,
            is_gt_suffix_path=path.is_gt_suffix_path,
        )
        for rank, path in enumerate(sorted_paths, start=1)
    )


def _score_path(path: _PathCandidate) -> _PathCandidate:
    return _PathCandidate(
        component_id=path.component_id,
        fragment_row=path.fragment_row,
        fragment_span=path.fragment_span,
        edges=path.edges,
        path_score=_mean_edge_score(path.edges),
        path_rank=path.path_rank,
        is_gt_suffix_path=path.is_gt_suffix_path,
    )


def _label_gt_path(
    path: _PathCandidate, fragment: np.ndarray, reference: np.ndarray
) -> _PathCandidate:
    is_gt = int(_path_matches_reference_suffix(path, fragment, reference))
    return _PathCandidate(
        component_id=path.component_id,
        fragment_row=path.fragment_row,
        fragment_span=path.fragment_span,
        edges=path.edges,
        path_score=path.path_score,
        path_rank=path.path_rank,
        is_gt_suffix_path=is_gt,
    )


def _path_matches_reference_suffix(
    path: _PathCandidate, fragment: np.ndarray, reference: np.ndarray
) -> bool:
    if not path.edges:
        return False
    observed_sessions = [index for index, value in enumerate(fragment) if value >= 0]
    if not observed_sessions:
        return False
    for reference_row in reference:
        if path.edges[-1].edge[1] < reference_row.size - 1:
            continue
        if not all(
            int(reference_row[session]) == int(fragment[session])
            for session in observed_sessions
            if 0 <= session < reference_row.size
        ):
            continue
        if all(
            int(reference_row[edge.edge[1]]) == int(edge.edge[3])
            for edge in path.edges
            if edge.edge[1] < reference_row.size
        ):
            return True
    return False


def _suffix_fragment_span(row: np.ndarray) -> tuple[int, int] | None:
    observed = np.flatnonzero(np.asarray(row, dtype=int) >= 0)
    if observed.size == 0:
        return None
    start = int(observed[0])
    tail = int(observed[-1])
    expected = np.arange(start, tail + 1, dtype=int)
    if observed.size != expected.size or not np.array_equal(observed, expected):
        return None
    return start, tail


def _edge_rows(
    subject: str, paths: Sequence[_PathCandidate]
) -> list[dict[str, float | int | str]]:
    by_edge: dict[TrackEdge, tuple[_EdgeCandidate, int]] = {}
    for path in paths:
        for edge in path.edges:
            existing = by_edge.get(edge.edge)
            gt = max(int(path.is_gt_suffix_path), int(existing[1]) if existing else 0)
            if existing is None or edge.edge_score > existing[0].edge_score:
                by_edge[edge.edge] = (edge, gt)
            else:
                by_edge[edge.edge] = (existing[0], gt)
    rows: list[dict[str, float | int | str]] = []
    for edge, is_gt in sorted(by_edge.values(), key=lambda item: item[0].edge):
        session_a, session_b, roi_a, roi_b = edge.edge
        rows.append(
            {
                "subject": subject,
                "session_a": int(session_a),
                "session_b": int(session_b),
                "roi_a": int(roi_a),
                "roi_b": int(roi_b),
                "is_gt_suffix_edge": int(is_gt),
                "registered_iou": float(edge.registered_iou),
                "shifted_iou": float(edge.shifted_iou),
                "roi_aware_score": float(edge.roi_aware_score),
                "centroid_distance": float(edge.centroid_distance),
                "area_ratio": float(edge.area_ratio),
                "cell_probability_a": float(edge.cell_probability_a),
                "cell_probability_b": float(edge.cell_probability_b),
                "row_rank": int(edge.row_rank),
                "column_rank": int(edge.column_rank),
                "row_margin": float(edge.row_margin),
                "column_margin": float(edge.column_margin),
                "threshold_margin": float(edge.threshold_margin),
                "activity_similarity": float(edge.activity_similarity),
                "edge_score": float(edge.edge_score),
            }
        )
    return rows


def _path_row(
    subject: str, path: _PathCandidate, predicted: np.ndarray, reference: np.ndarray
) -> dict[str, float | int | str]:
    edge_values = path.edges
    candidate_sessions = tuple(edge.edge[1] for edge in edge_values)
    candidate_rois = tuple(edge.edge[3] for edge in edge_values)
    return {
        "subject": subject,
        "component_id": int(path.component_id),
        "fragment_span": path.fragment_span,
        "candidate_path": _edge_list(tuple(edge.edge for edge in edge_values)),
        "candidate_rois": _int_list(candidate_rois),
        "candidate_sessions": _int_list(candidate_sessions),
        "path_length": int(len(edge_values)),
        "would_reach_final_session": int(
            bool(
                candidate_sessions and max(candidate_sessions) >= predicted.shape[1] - 1
            )
        ),
        "creates_duplicate_source": int(
            any(_creates_duplicate_source(edge.edge, predicted) for edge in edge_values)
        ),
        "creates_duplicate_target": int(
            any(_creates_duplicate_target(edge.edge, predicted) for edge in edge_values)
        ),
        "would_merge_complete_tp": int(
            any(
                _would_merge_complete_tp(
                    edge.edge, predicted, reference, path.component_id
                )
                for edge in edge_values
            )
        ),
        "is_gt_suffix_path": int(path.is_gt_suffix_path),
        "path_rank": int(path.path_rank),
        "path_score": float(path.path_score),
        "min_edge_score": _min_attr(edge_values, "edge_score"),
        "mean_edge_score": _mean_attr(edge_values, "edge_score"),
        "min_cell_probability": min(
            _nanmin_default(
                (edge.cell_probability_a, edge.cell_probability_b), float("nan")
            )
            for edge in edge_values
        ),
        "min_area_ratio": _min_attr(edge_values, "area_ratio"),
        "max_centroid_distance": _max_attr(edge_values, "centroid_distance"),
        "min_registered_iou": _min_attr(edge_values, "registered_iou"),
        "max_registered_iou": _max_attr(edge_values, "registered_iou"),
        "min_shifted_iou": _min_attr(edge_values, "shifted_iou"),
        "min_row_margin": _min_attr(edge_values, "row_margin"),
        "min_column_margin": _min_attr(edge_values, "column_margin"),
        "activity_similarity": _mean_attr(edge_values, "activity_similarity"),
        "motion_consistency": _motion_consistency(edge_values),
        "shape_consistency": _mean_attr(edge_values, "area_ratio"),
    }


def _summary_row(
    subject: str, path_rows: Sequence[Mapping[str, float | int | str]]
) -> dict[str, float | int | str]:
    component_ids = {
        (str(row.get("subject", subject)), int(row["component_id"]))
        for row in path_rows
    }
    gt_rows = [row for row in path_rows if int(row.get("is_gt_suffix_path", 0)) > 0]
    gt_ranks = [int(row["path_rank"]) for row in gt_rows]
    best_gt_score = max(
        (float(row["path_score"]) for row in gt_rows),
        default=float("nan"),
    )
    non_gt_same_gate = 0
    if np.isfinite(best_gt_score):
        non_gt_same_gate = sum(
            int(row.get("is_gt_suffix_path", 0)) == 0
            and float(row.get("path_score", float("-inf"))) >= best_gt_score
            for row in path_rows
        )
    return {
        "subject": subject,
        "suffix_fragment_candidates": int(len(component_ids)),
        "candidate_paths": int(len(path_rows)),
        "number_of_gt_suffix_paths": int(len(gt_rows)),
        "gt_suffix_path_rank": _int_list(tuple(sorted(gt_ranks))),
        "best_gt_suffix_path_rank": min(gt_ranks) if gt_ranks else -1,
        "top1_recovery_rate": _rate(sum(rank <= 1 for rank in gt_ranks), len(gt_ranks)),
        "top3_recovery_rate": _rate(sum(rank <= 3 for rank in gt_ranks), len(gt_ranks)),
        "non_gt_paths_that_would_pass_same_gate": int(non_gt_same_gate),
    }


def _assignment_threshold(
    iou: np.ndarray, *, threshold_method: ThresholdMethod
) -> float:
    if iou.size == 0:
        return float("inf")
    row_ind, col_ind = linear_sum_assignment(1.0 - iou)
    return float(
        _threshold_assigned_iou(iou[row_ind, col_ind], method=threshold_method)
    )


def _activity_similarity_matrix(
    reference_session: Track2pSession, moving_session: Track2pSession
) -> np.ndarray:
    shape = (reference_session.plane_data.n_rois, moving_session.plane_data.n_rois)
    try:
        components = activity_similarity_components(
            reference_session.plane_data, moving_session.plane_data
        )
    except (TypeError, ValueError):
        return np.full(shape, float("nan"), dtype=float)
    value = components.get("activity_similarity")
    if value is None:
        return np.full(shape, float("nan"), dtype=float)
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != shape:
        return np.full(shape, float("nan"), dtype=float)
    return matrix


def _rank_descending(values: np.ndarray, *, selected_index: int) -> int:
    values = np.asarray(values, dtype=float).reshape(-1)
    selected = float(values[int(selected_index)])
    return int(1 + np.sum(values > selected))


def _creates_duplicate_source(edge: TrackEdge, predicted: np.ndarray) -> bool:
    session_a, session_b, roi_a, roi_b = edge
    for row in predicted:
        if session_b >= row.size or session_a >= row.size:
            continue
        if int(row[session_a]) == int(roi_a) and row[session_b] >= 0:
            return int(row[session_b]) != int(roi_b)
    return False


def _creates_duplicate_target(edge: TrackEdge, predicted: np.ndarray) -> bool:
    session_a, session_b, roi_a, roi_b = edge
    for row in predicted:
        if session_b >= row.size or session_a >= row.size:
            continue
        if int(row[session_b]) == int(roi_b) and row[session_a] >= 0:
            return int(row[session_a]) != int(roi_a)
    return False


def _would_merge_complete_tp(
    edge: TrackEdge, predicted: np.ndarray, reference: np.ndarray, component_id: int
) -> bool:
    _session_a, session_b, _roi_a, roi_b = edge
    reference_counts = Counter(tuple(int(value) for value in row) for row in reference)
    for row_id, row in enumerate(predicted):
        if int(row_id) == int(component_id):
            continue
        if session_b >= row.size or int(row[session_b]) != int(roi_b):
            continue
        if (
            np.all(row >= 0)
            and reference_counts.get(tuple(int(value) for value in row), 0) > 0
        ):
            return True
    return False


def _mean_edge_score(edges: Sequence[_EdgeCandidate]) -> float:
    return _mean_attr(edges, "edge_score")


def _mean_attr(edges: Sequence[_EdgeCandidate], name: str) -> float:
    values = [
        float(getattr(edge, name))
        for edge in edges
        if np.isfinite(float(getattr(edge, name)))
    ]
    return float(np.mean(values)) if values else float("nan")


def _min_attr(edges: Sequence[_EdgeCandidate], name: str) -> float:
    values = [
        float(getattr(edge, name))
        for edge in edges
        if np.isfinite(float(getattr(edge, name)))
    ]
    return float(np.min(values)) if values else float("nan")


def _max_attr(edges: Sequence[_EdgeCandidate], name: str) -> float:
    values = [
        float(getattr(edge, name))
        for edge in edges
        if np.isfinite(float(getattr(edge, name)))
    ]
    return float(np.max(values)) if values else float("nan")


def _motion_consistency(edges: Sequence[_EdgeCandidate]) -> float:
    distances = [
        float(edge.centroid_distance)
        for edge in edges
        if np.isfinite(float(edge.centroid_distance))
    ]
    if len(distances) <= 1:
        return 1.0
    return float(1.0 / (1.0 + np.std(np.asarray(distances, dtype=float))))


def _nanmin_default(values: Sequence[float], default: float) -> float:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return float(min(finite)) if finite else float(default)


def _finite_or(value: float, default: float) -> float:
    return float(value) if np.isfinite(float(value)) else float(default)


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _edge_list(edges: Sequence[TrackEdge]) -> str:
    return ";".join(_edge_id(edge) for edge in edges)


def _edge_id(edge: TrackEdge) -> str:
    session_a, session_b, roi_a, roi_b = edge
    return f"{session_a}:{roi_a}->{session_b}:{roi_b}"


def _int_list(values: Sequence[int]) -> str:
    return ",".join(str(int(value)) for value in values)


def write_rows(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    output_format: Literal["csv", "json"] = "csv",
) -> None:
    """Write rows as CSV or JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output_path.write_text(
            json.dumps(list(rows), indent=2) + "\n", encoding="utf-8"
        )
        return
    fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for suffix-stitch ranking audit."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-policy-suffix-stitch-ranking-audit",
        description="Rank short suffix-stitch candidates from non-GT features.",
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
        "--iou-distance-threshold",
        type=float,
        default=TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    )
    parser.add_argument(
        "--cell-probability-threshold",
        type=float,
        default=TRACK2P_POLICY_DEFAULT_CELL_PROBABILITY_THRESHOLD,
    )
    parser.add_argument(
        "--transform-type", default=TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE
    )
    parser.add_argument("--split-risk-threshold", type=float, default=1.50)
    parser.add_argument("--split-penalty", type=float, default=0.25)
    parser.add_argument("--min-side-observations", type=int, default=2)
    parser.add_argument("--max-suffix-length", type=_positive_int_arg, default=2)
    parser.add_argument("--edge-top-k", type=_positive_int_arg, default=25)
    parser.add_argument("--path-beam-width", type=_positive_int_arg, default=100)
    parser.add_argument(
        "--require-complete-track",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--path-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--format", choices=("csv", "json"), default="csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the suffix-stitch ranking audit CLI."""

    args = build_arg_parser().parse_args(argv)
    cleanup_config = ComponentCleanupConfig(
        split_risk_threshold=args.split_risk_threshold,
        split_penalty=args.split_penalty,
        min_side_observations=args.min_side_observations,
        require_complete_track=args.require_complete_track,
    )
    config = Track2pBenchmarkConfig(
        data=args.data,
        method="global-assignment",
        input_format=args.input_format,
        reference=args.reference,
        reference_kind=args.reference_kind,
        plane_name=args.plane_name,
        seed_session=args.seed_session,
        restrict_to_reference_seed_rois=args.restrict_to_reference_seed_rois,
        transform_type=args.transform_type,
        allow_track2p_as_reference_for_smoke_test=args.allow_track2p_as_reference_for_smoke_test,
        include_behavior=args.include_behavior,
        include_non_cells=False,
        cell_probability_threshold=args.cell_probability_threshold,
        exclude_overlapping_pixels=False,
        weighted_masks=False,
        weighted_centroids=False,
    )
    result = run_track2p_policy_suffix_stitch_ranking_audit(
        config,
        threshold_method=cast(ThresholdMethod, args.threshold_method),
        iou_distance_threshold=float(args.iou_distance_threshold),
        transform_type=args.transform_type,
        cell_probability_threshold=float(args.cell_probability_threshold),
        cleanup_config=cleanup_config,
        max_suffix_length=int(args.max_suffix_length),
        edge_top_k=int(args.edge_top_k),
        path_beam_width=int(args.path_beam_width),
    )
    write_rows(result.edge_rows, args.output, output_format=args.format)
    write_rows(result.path_rows, args.path_output, output_format=args.format)
    write_rows(result.summary_rows, args.summary_output, output_format=args.format)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
