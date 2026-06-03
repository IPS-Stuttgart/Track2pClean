"""Growth-field residual audit after CoherenceSuffixTeacherRescue.

This diagnostic starts from the current Track2p-teacher-assisted lead row and
audits the residual official pairwise errors against a label-free growth prior.
The growth/deformation field is fit only from high-confidence agreement edges:
edges that are present in Track2p, Track2pPolicy, and ComponentCleanup or the
combined prediction. Manual-GT labels are used only to define residual errors
and score what-if deltas.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from bayescatrack.association.growth_priors import fit_affine_growth_transform
from bayescatrack.core.bridge import Track2pSession
from bayescatrack.evaluation.complete_track_scores import score_track_matrices
from bayescatrack.experiments import (
    track2p_policy_coherence_suffix_stitch_whatif as suffix,
)
from bayescatrack.experiments.track2p_benchmark import (
    GROUND_TRUTH_REFERENCE_SOURCE,
    Track2pBenchmarkConfig,
    _load_reference_for_subject,
    _load_subject_sessions,
    _predict_subject_tracks,
    _reference_matrix,
    _validate_reference_for_benchmark,
    _validate_reference_roi_indices,
    discover_subject_dirs,
)
from bayescatrack.experiments.track2p_emulation_benchmark import ThresholdMethod
from bayescatrack.experiments.track2p_policy_audit import TrackEdge, track_edge_counter
from bayescatrack.experiments.track2p_policy_benchmark import (
    TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    track2p_policy_config,
)
from bayescatrack.experiments.track2p_policy_component_audit import (
    ComponentCleanupConfig,
    _evaluated_prediction_rows,
    _normalize_int_track_matrix,
)
from bayescatrack.experiments.track2p_policy_component_residual_audit import (
    ResidualFeature,
    _cell_probability,
    _complete_track_counter,
    _feature_subset_for_edges,
    _no_prune_config,
)
from bayescatrack.experiments.track2p_policy_pruned_benchmark import (
    _margin_against_competitor,
    _roi_indices,
    emulate_track2p_pruned_tracks,
)
from bayescatrack.experiments.track2p_policy_suffix_stitch_ranking_audit import (
    _FeatureCache,
    _rank_descending,
)
from bayescatrack.experiments.track2p_policy_teacher_adjacent_rescue import (
    _try_apply_teacher_edge,
    apply_teacher_adjacent_rescue_edges,
)

METHOD = "track2p-policy-growth-field-residual-audit"

ResidualKind = Literal["pairwise_fp", "pairwise_fn"]


@dataclass(frozen=True)
class GrowthFieldResidualAuditResult:
    """Residual edge rows and compact growth-support summaries."""

    edge_rows: tuple[dict[str, Any], ...]
    summary_rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _GrowthModel:
    affine_xy: np.ndarray
    model_type: str
    anchor_count: int
    inlier_count: int
    residual_scale: float
    covariance_inverse: np.ndarray
    expected_area_ratio: float


@dataclass(frozen=True)
class _EdgeGrowthFeatures:
    centroid_a_x: float = float("nan")
    centroid_a_y: float = float("nan")
    centroid_b_x: float = float("nan")
    centroid_b_y: float = float("nan")
    predicted_b_x: float = float("nan")
    predicted_b_y: float = float("nan")
    growth_residual: float = float("nan")
    growth_residual_mahalanobis: float = float("nan")
    radial_direction_cosine: float = float("nan")
    expected_area_ratio: float = float("nan")
    observed_area_ratio: float = float("nan")
    area_growth_residual: float = float("nan")
    local_neighbor_distortion: float = float("nan")
    two_edge_motion_consistency: float = float("nan")
    two_edge_acceleration: float = float("nan")


def run_track2p_policy_growth_field_residual_audit(
    config: Track2pBenchmarkConfig,
    *,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    iou_distance_threshold: float = TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    transform_type: str | None = None,
    cell_probability_threshold: float | None = None,
    cleanup_config: ComponentCleanupConfig | None = None,
    suffix_gate: suffix.CoherenceSuffixStitchGate | None = None,
    edge_top_k: int = 25,
    path_beam_width: int = 100,
    anchor_min_registered_iou: float = 0.50,
    anchor_min_shifted_iou: float = 0.30,
    anchor_min_cell_probability: float = 0.80,
) -> GrowthFieldResidualAuditResult:
    """Audit residual pairwise errors against a fitted growth field."""

    policy_config = track2p_policy_config(
        config,
        transform_type=transform_type,
        cell_probability_threshold=cell_probability_threshold,
    )
    cleanup_config = cleanup_config or ComponentCleanupConfig()
    suffix_gate = suffix_gate or suffix.CoherenceSuffixStitchGate()
    subject_dirs = discover_subject_dirs(policy_config.data)
    if not subject_dirs:
        raise ValueError(
            f"No Track2p-style subject directories found under {policy_config.data}"
        )

    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for subject_dir in subject_dirs:
        subject_rows = _subject_rows(
            subject_dir,
            config=policy_config,
            cleanup_config=cleanup_config,
            suffix_gate=suffix_gate,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
            edge_top_k=int(edge_top_k),
            path_beam_width=int(path_beam_width),
            anchor_min_registered_iou=float(anchor_min_registered_iou),
            anchor_min_shifted_iou=float(anchor_min_shifted_iou),
            anchor_min_cell_probability=float(anchor_min_cell_probability),
        )
        all_rows.extend(subject_rows)
        summary_rows.extend(_summary_rows(subject_dir.name, subject_rows))
    summary_rows.extend(_summary_rows("ALL", all_rows))
    return GrowthFieldResidualAuditResult(tuple(all_rows), tuple(summary_rows))


def _subject_rows(
    subject_dir: Path,
    *,
    config: Track2pBenchmarkConfig,
    cleanup_config: ComponentCleanupConfig,
    suffix_gate: suffix.CoherenceSuffixStitchGate,
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
    edge_top_k: int,
    path_beam_width: int,
    anchor_min_registered_iou: float,
    anchor_min_shifted_iou: float,
    anchor_min_cell_probability: float,
) -> list[dict[str, Any]]:
    reference = _load_reference_for_subject(
        subject_dir, data_root=config.data, config=config
    )
    _validate_reference_for_benchmark(reference, subject_dir=subject_dir, config=config)
    if reference.source != GROUND_TRUTH_REFERENCE_SOURCE:
        raise ValueError(f"{METHOD} requires independent manual-GT references")
    sessions = _load_subject_sessions(subject_dir, config)
    _validate_reference_roi_indices(reference, sessions)
    reference_tracks = _reference_matrix(reference, curated_only=config.curated_only)

    policy_prediction = emulate_track2p_pruned_tracks(
        sessions,
        transform_type=config.transform_type,
        threshold_method=threshold_method,
        iou_distance_threshold=float(iou_distance_threshold),
        prune_config=_no_prune_config(),
    )
    policy_full = _normalize_int_track_matrix(policy_prediction.tracks)
    policy_eval, reference_eval, _policy_ids = _evaluated_prediction_rows(
        policy_full, reference_tracks, config=config
    )
    reference_eval = _as_track_matrix(reference_eval)
    n_sessions = int(reference_eval.shape[1])
    policy_eval = _pad_track_matrix(_as_track_matrix(policy_eval), width=n_sessions)
    cleaned, reference_eval = suffix._component_cleanup_eval(
        sessions,
        reference_tracks,
        subject=subject_dir.name,
        config=config,
        cleanup_config=cleanup_config,
        threshold_method=threshold_method,
        iou_distance_threshold=float(iou_distance_threshold),
    )
    cleaned = _pad_track_matrix(_as_track_matrix(cleaned), width=n_sessions)
    feature_cache = _FeatureCache(
        sessions=sessions,
        transform_type=str(config.transform_type),
        threshold_method=threshold_method,
        iou_distance_threshold=float(iou_distance_threshold),
        cell_probability_threshold=float(config.cell_probability_threshold),
        matrices={},
    )
    paths = suffix._ranked_suffix_paths(
        cleaned,
        reference_eval,
        subject=subject_dir.name,
        feature_cache=feature_cache,
        max_suffix_length=int(suffix_gate.suffix_path_length),
        edge_top_k=int(edge_top_k),
        path_beam_width=int(path_beam_width),
    )
    selected = suffix._select_paths(paths, cleaned, gate=suffix_gate)
    stitched = _pad_track_matrix(
        _as_track_matrix(suffix._apply_suffix_paths(cleaned, selected)),
        width=n_sessions,
    )

    teacher_full, _variant = _predict_subject_tracks(
        subject_dir, replace(config, method="track2p-baseline")
    )
    teacher, _reference_again, _teacher_ids = _evaluated_prediction_rows(
        _normalize_int_track_matrix(teacher_full), reference_tracks, config=config
    )
    teacher = _pad_track_matrix(_as_track_matrix(teacher), width=n_sessions)
    teacher_report = apply_teacher_adjacent_rescue_edges(
        stitched,
        teacher,
        seed_session=config.seed_session,
        allow_completing_rescue=False,
        allow_source_backfill=True,
        allow_fragment_merges=True,
        edge_order="structural",
        teacher_action_filter="all",
        teacher_feature_gate=None,
        min_component_observations=1,
        max_applied_edits=None,
    )
    combined = _pad_track_matrix(
        _as_track_matrix(_normalize_int_track_matrix(teacher_report.tracks)),
        width=n_sessions,
    )
    baseline_scores = dict(score_track_matrices(combined, reference_eval))

    residuals = _residual_edges(combined, reference_eval)
    requested_edges = [edge for edge, _kind, _occurrence in residuals]
    requested_features = _feature_subset_for_edges(
        sessions,
        set(requested_edges),
        transform_type=config.transform_type,
        threshold_method=threshold_method,
        iou_distance_threshold=float(iou_distance_threshold),
    )
    anchor_edges = _anchor_edges(
        sessions,
        feature_cache=feature_cache,
        track2p=teacher,
        policy=policy_eval,
        component_cleanup=cleaned,
        combined=combined,
        min_registered_iou=float(anchor_min_registered_iou),
        min_shifted_iou=float(anchor_min_shifted_iou),
        min_cell_probability=float(anchor_min_cell_probability),
    )
    growth_models = _growth_models_by_pair(sessions, anchor_edges)

    rows: list[dict[str, Any]] = []
    for edge, error_type, occurrence_index in residuals:
        model = growth_models.get((edge[0], edge[1]), _identity_growth_model())
        growth = _edge_growth_features(
            sessions,
            edge,
            model=model,
            anchor_edges=anchor_edges.get((edge[0], edge[1]), ()),
            predicted=combined,
        )
        local = requested_features.get(edge, ResidualFeature())
        candidate, edit_reason = _what_if_candidate(
            combined,
            reference_eval,
            edge,
            error_type=error_type,
            seed_session=config.seed_session,
        )
        candidate_scores = dict(score_track_matrices(candidate, reference_eval))
        delta = suffix._score_delta(baseline_scores, candidate_scores)
        rows.append(
            _edge_row(
                subject=subject_dir.name,
                edge=edge,
                error_type=error_type,
                occurrence_index=occurrence_index,
                local=local,
                growth=growth,
                model=model,
                baseline_scores=baseline_scores,
                candidate_scores=candidate_scores,
                delta=delta,
                edit_reason=edit_reason,
                track2p=teacher,
                policy=policy_eval,
                component_cleanup=cleaned,
                combined=combined,
                threshold_method=threshold_method,
                iou_distance_threshold=float(iou_distance_threshold),
                cell_probability_threshold=float(config.cell_probability_threshold),
                transform_type=config.transform_type,
            )
        )
    return _augment_with_shifted_iou_and_margins(
        rows, sessions, feature_cache=feature_cache
    )


def _residual_edges(
    predicted: np.ndarray, reference: np.ndarray
) -> tuple[tuple[TrackEdge, ResidualKind, int], ...]:
    predicted_counts = track_edge_counter(predicted)
    reference_counts = track_edge_counter(reference)
    rows: list[tuple[TrackEdge, ResidualKind, int]] = []
    for edge in sorted(set(predicted_counts) | set(reference_counts)):
        fp_count = int(predicted_counts.get(edge, 0) - reference_counts.get(edge, 0))
        for occurrence in range(max(0, fp_count)):
            rows.append((edge, "pairwise_fp", occurrence))
        fn_count = int(reference_counts.get(edge, 0) - predicted_counts.get(edge, 0))
        for occurrence in range(max(0, fn_count)):
            rows.append((edge, "pairwise_fn", occurrence))
    return tuple(rows)


def _anchor_edges(
    sessions: Sequence[Track2pSession],
    *,
    feature_cache: _FeatureCache,
    track2p: np.ndarray,
    policy: np.ndarray,
    component_cleanup: np.ndarray,
    combined: np.ndarray,
    min_registered_iou: float,
    min_shifted_iou: float,
    min_cell_probability: float,
) -> dict[tuple[int, int], tuple[TrackEdge, ...]]:
    track2p_edges = set(track_edge_counter(track2p))
    policy_edges = set(track_edge_counter(policy))
    cleanup_or_combined = set(track_edge_counter(component_cleanup)) | set(
        track_edge_counter(combined)
    )
    candidates = track2p_edges & policy_edges & cleanup_or_combined
    by_pair: dict[tuple[int, int], list[TrackEdge]] = defaultdict(list)
    for edge in sorted(candidates):
        session_a, session_b, roi_a, roi_b = edge
        if session_b != session_a + 1:
            continue
        matrices = feature_cache.pair(session_a)
        local_a = _local_index(matrices.source_indices, roi_a)
        local_b = _local_index(matrices.target_indices, roi_b)
        if local_a < 0 or local_b < 0:
            continue
        cell_a = _cell_probability(sessions, session_a, roi_a)
        cell_b = _cell_probability(sessions, session_b, roi_b)
        if min(cell_a, cell_b) < float(min_cell_probability):
            continue
        if float(matrices.registered_iou[local_a, local_b]) < float(min_registered_iou):
            continue
        if float(matrices.shifted_iou[local_a, local_b]) < float(min_shifted_iou):
            continue
        by_pair[(session_a, session_b)].append(edge)

    output: dict[tuple[int, int], tuple[TrackEdge, ...]] = {}
    for pair, edges in by_pair.items():
        source_counts = Counter((edge[0], edge[2]) for edge in edges)
        target_counts = Counter((edge[1], edge[3]) for edge in edges)
        conflict_free = tuple(
            edge
            for edge in edges
            if source_counts[(edge[0], edge[2])] == 1
            and target_counts[(edge[1], edge[3])] == 1
        )
        output[pair] = conflict_free
    return output


def _growth_models_by_pair(
    sessions: Sequence[Track2pSession],
    anchor_edges: Mapping[tuple[int, int], Sequence[TrackEdge]],
) -> dict[tuple[int, int], _GrowthModel]:
    return {
        pair: _fit_growth_model(sessions, edges) for pair, edges in anchor_edges.items()
    }


def _fit_growth_model(
    sessions: Sequence[Track2pSession], edges: Sequence[TrackEdge]
) -> _GrowthModel:
    if not edges:
        return _identity_growth_model()
    sources: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for edge in edges:
        source = _centroid_xy(sessions, edge[0], edge[2])
        target = _centroid_xy(sessions, edge[1], edge[3])
        if source is None or target is None:
            continue
        sources.append(source)
        targets.append(target)
    if not sources:
        return _identity_growth_model()
    source_xy = np.vstack(sources)
    target_xy = np.vstack(targets)
    if source_xy.shape[0] >= 3:
        affine = fit_affine_growth_transform(source_xy, target_xy)
        residual_vectors = _residual_vectors(source_xy, target_xy, affine)
        residual_norms = np.linalg.norm(residual_vectors, axis=1)
        median = float(np.median(residual_norms))
        mad = float(np.median(np.abs(residual_norms - median)))
        keep = residual_norms <= median + max(3.0 * 1.4826 * mad, 2.0)
        if int(np.sum(keep)) >= 3 and not bool(np.all(keep)):
            affine = fit_affine_growth_transform(source_xy[keep], target_xy[keep])
            residual_vectors = _residual_vectors(
                source_xy[keep], target_xy[keep], affine
            )
        model_type = "robust_affine" if int(np.sum(keep)) < len(keep) else "affine"
        inlier_count = int(np.sum(keep))
    else:
        displacement = np.median(target_xy - source_xy, axis=0)
        affine = np.asarray(
            [[1.0, 0.0, float(displacement[0])], [0.0, 1.0, float(displacement[1])]],
            dtype=float,
        )
        residual_vectors = _residual_vectors(source_xy, target_xy, affine)
        model_type = "translation_fallback"
        inlier_count = int(source_xy.shape[0])
    residual_scale = _robust_residual_scale(residual_vectors)
    covariance_inverse = _residual_covariance_inverse(residual_vectors, residual_scale)
    expected_area_ratio = _expected_area_ratio(affine)
    return _GrowthModel(
        affine_xy=affine,
        model_type=model_type,
        anchor_count=int(source_xy.shape[0]),
        inlier_count=inlier_count,
        residual_scale=residual_scale,
        covariance_inverse=covariance_inverse,
        expected_area_ratio=expected_area_ratio,
    )


def _identity_growth_model() -> _GrowthModel:
    return _GrowthModel(
        affine_xy=np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
        model_type="identity_no_anchors",
        anchor_count=0,
        inlier_count=0,
        residual_scale=1.0,
        covariance_inverse=np.eye(2, dtype=float),
        expected_area_ratio=1.0,
    )


def _edge_growth_features(
    sessions: Sequence[Track2pSession],
    edge: TrackEdge,
    *,
    model: _GrowthModel,
    anchor_edges: Sequence[TrackEdge],
    predicted: np.ndarray | None = None,
) -> _EdgeGrowthFeatures:
    source = _centroid_xy(sessions, edge[0], edge[2])
    target = _centroid_xy(sessions, edge[1], edge[3])
    if source is None or target is None:
        return _EdgeGrowthFeatures(expected_area_ratio=model.expected_area_ratio)
    predicted = _apply_affine(source, model.affine_xy)
    residual_vector = target - predicted
    residual = float(np.linalg.norm(residual_vector))
    mahalanobis = float(
        np.sqrt(
            max(
                0.0, float(residual_vector @ model.covariance_inverse @ residual_vector)
            )
        )
    )
    observed_area_ratio = _observed_area_ratio(sessions, edge)
    area_residual = _area_growth_residual(
        observed_area_ratio, model.expected_area_ratio
    )
    motion = _motion_context_features(sessions, edge, predicted, source, target)
    return _EdgeGrowthFeatures(
        centroid_a_x=float(source[0]),
        centroid_a_y=float(source[1]),
        centroid_b_x=float(target[0]),
        centroid_b_y=float(target[1]),
        predicted_b_x=float(predicted[0]),
        predicted_b_y=float(predicted[1]),
        growth_residual=residual,
        growth_residual_mahalanobis=mahalanobis,
        radial_direction_cosine=_radial_direction_cosine(
            sessions, edge, source, target
        ),
        expected_area_ratio=float(model.expected_area_ratio),
        observed_area_ratio=observed_area_ratio,
        area_growth_residual=area_residual,
        local_neighbor_distortion=_local_neighbor_distortion(
            sessions, edge, anchor_edges, expected_area_ratio=model.expected_area_ratio
        ),
        two_edge_motion_consistency=motion["two_edge_motion_consistency"],
        two_edge_acceleration=motion["two_edge_acceleration"],
    )


def _motion_context_features(
    sessions: Sequence[Track2pSession],
    edge: TrackEdge,
    predicted: np.ndarray | None,
    source_xy: np.ndarray,
    target_xy: np.ndarray,
) -> dict[str, float]:
    if predicted is None:
        return {
            "two_edge_motion_consistency": float("nan"),
            "two_edge_acceleration": float("nan"),
        }
    predicted = _as_track_matrix(predicted)
    session_a, session_b, roi_a, roi_b = edge
    if session_a >= predicted.shape[1] or session_b >= predicted.shape[1]:
        return {
            "two_edge_motion_consistency": float("nan"),
            "two_edge_acceleration": float("nan"),
        }
    current = np.asarray(target_xy, dtype=float) - np.asarray(source_xy, dtype=float)
    adjacent_vectors: list[np.ndarray] = []
    source_rows = tuple(np.flatnonzero(predicted[:, session_a] == int(roi_a)))
    if len(source_rows) == 1 and session_a > 0:
        previous_roi = int(predicted[int(source_rows[0]), session_a - 1])
        previous = _centroid_xy(sessions, session_a - 1, previous_roi)
        if previous is not None:
            adjacent_vectors.append(np.asarray(source_xy, dtype=float) - previous)
    target_rows = tuple(np.flatnonzero(predicted[:, session_b] == int(roi_b)))
    if len(target_rows) == 1 and session_b + 1 < predicted.shape[1]:
        next_roi = int(predicted[int(target_rows[0]), session_b + 1])
        next_xy = _centroid_xy(sessions, session_b + 1, next_roi)
        if next_xy is not None:
            adjacent_vectors.append(next_xy - np.asarray(target_xy, dtype=float))
    cosines = [_cosine(current, vector) for vector in adjacent_vectors]
    finite_cosines = [value for value in cosines if np.isfinite(value)]
    accelerations = [
        float(np.linalg.norm(current - vector)) for vector in adjacent_vectors
    ]
    return {
        "two_edge_motion_consistency": (
            float(np.mean(finite_cosines)) if finite_cosines else float("nan")
        ),
        "two_edge_acceleration": (
            float(np.mean(accelerations)) if accelerations else float("nan")
        ),
    }


def _what_if_candidate(
    predicted: np.ndarray,
    reference: np.ndarray,
    edge: TrackEdge,
    *,
    error_type: ResidualKind,
    seed_session: int,
) -> tuple[np.ndarray, str]:
    if error_type == "pairwise_fn":
        candidate, attempt = _try_apply_teacher_edge(
            predicted,
            edge,
            seed_session=int(seed_session),
            allow_completing_rescue=True,
            allow_source_backfill=True,
            allow_fragment_merges=True,
            min_component_observations=1,
        )
        return candidate, str(attempt.get("reason", "not_evaluated"))
    candidate, reason = _split_at_edge(predicted, edge, reference)
    return candidate, reason


def _split_at_edge(
    predicted: np.ndarray, edge: TrackEdge, reference: np.ndarray
) -> tuple[np.ndarray, str]:
    predicted = _as_track_matrix(predicted)
    session_a, session_b, roi_a, roi_b = edge
    if session_a >= predicted.shape[1] or session_b >= predicted.shape[1]:
        return predicted.copy(), "edge_outside_prediction_width"
    if session_b != session_a + 1:
        return np.asarray(predicted, dtype=int).copy(), "not_adjacent"
    rows = [
        int(row_index)
        for row_index in np.flatnonzero(predicted[:, session_a] == roi_a)
        if int(predicted[int(row_index), session_b]) == int(roi_b)
    ]
    if len(rows) != 1:
        return np.asarray(predicted, dtype=int).copy(), "edge_absent_or_ambiguous"
    row_index = rows[0]
    row = np.asarray(predicted[row_index], dtype=int)
    if np.all(row >= 0) and _complete_track_counter(reference).get(
        tuple(int(value) for value in row), 0
    ):
        return np.asarray(predicted, dtype=int).copy(), "would_break_complete_tp"
    left = row.copy()
    right = row.copy()
    left[session_b:] = -1
    right[:session_b] = -1
    fragments = [fragment for fragment in (left, right) if np.any(fragment >= 0)]
    output = np.delete(np.asarray(predicted, dtype=int), row_index, axis=0)
    if fragments:
        output = np.vstack([output, *fragments])
    return output, "split_edge"


def _as_track_matrix(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=int)
    if matrix.ndim == 0:
        return matrix.reshape(1, 1)
    if matrix.ndim == 1:
        return matrix.reshape(1, -1)
    return matrix


def _pad_track_matrix(matrix: np.ndarray, *, width: int) -> np.ndarray:
    matrix = _as_track_matrix(matrix)
    if matrix.shape[1] == int(width):
        return matrix
    if matrix.shape[1] > int(width):
        return matrix[:, : int(width)]
    padded = np.full((matrix.shape[0], int(width)), -1, dtype=int)
    padded[:, : matrix.shape[1]] = matrix
    return padded


def _edge_row(
    *,
    subject: str,
    edge: TrackEdge,
    error_type: ResidualKind,
    occurrence_index: int,
    local: ResidualFeature,
    growth: _EdgeGrowthFeatures,
    model: _GrowthModel,
    baseline_scores: Mapping[str, Any],
    candidate_scores: Mapping[str, Any],
    delta: Mapping[str, int],
    edit_reason: str,
    track2p: np.ndarray,
    policy: np.ndarray,
    component_cleanup: np.ndarray,
    combined: np.ndarray,
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
    cell_probability_threshold: float,
    transform_type: str,
) -> dict[str, Any]:
    session_a, session_b, roi_a, roi_b = edge
    return {
        "subject": subject,
        "session_a": int(session_a),
        "session_b": int(session_b),
        "roi_a": int(roi_a),
        "roi_b": int(roi_b),
        "error_type": error_type,
        "is_residual_fp": int(error_type == "pairwise_fp"),
        "is_residual_fn": int(error_type == "pairwise_fn"),
        "occurrence_index": int(occurrence_index),
        "track2p_supported": int(
            track_edge_counter(track2p).get(edge, 0) > occurrence_index
        ),
        "policy_supported": int(
            track_edge_counter(policy).get(edge, 0) > occurrence_index
        ),
        "component_cleanup_supported": int(
            track_edge_counter(component_cleanup).get(edge, 0) > occurrence_index
        ),
        "coherence_suffix_teacher_supported": int(
            track_edge_counter(combined).get(edge, 0) > occurrence_index
        ),
        "centroid_a_x": growth.centroid_a_x,
        "centroid_a_y": growth.centroid_a_y,
        "centroid_b_x": growth.centroid_b_x,
        "centroid_b_y": growth.centroid_b_y,
        "predicted_b_x": growth.predicted_b_x,
        "predicted_b_y": growth.predicted_b_y,
        "growth_residual": growth.growth_residual,
        "growth_residual_mahalanobis": growth.growth_residual_mahalanobis,
        "radial_direction_cosine": growth.radial_direction_cosine,
        "expected_area_ratio": growth.expected_area_ratio,
        "observed_area_ratio": growth.observed_area_ratio,
        "area_growth_residual": growth.area_growth_residual,
        "local_neighbor_distortion": growth.local_neighbor_distortion,
        "two_edge_motion_consistency": growth.two_edge_motion_consistency,
        "two_edge_acceleration": growth.two_edge_acceleration,
        "growth_model_type": model.model_type,
        "growth_anchor_count": int(model.anchor_count),
        "growth_inlier_count": int(model.inlier_count),
        "growth_residual_scale": float(model.residual_scale),
        "registered_iou": float(local.registered_iou),
        "shifted_iou": float("nan"),
        "centroid_distance": float(local.centroid_distance),
        "area_ratio": float(local.area_ratio),
        "cell_probability_a": float(local.cell_probability_a),
        "cell_probability_b": float(local.cell_probability_b),
        "row_rank": int(local.row_rank),
        "column_rank": int(local.column_rank),
        "row_margin": float(local.row_margin),
        "column_margin": float(local.column_margin),
        "threshold_margin": float(local.threshold_margin),
        "assigned_by_hungarian": int(local.assigned_by_hungarian),
        "edit_reason": edit_reason,
        "would_fix_complete_fn": int(
            delta["complete_track_false_negatives"] < 0
            and delta["complete_track_true_positives"] > 0
        ),
        "would_create_complete_fp": int(delta["complete_track_false_positives"] > 0),
        "pairwise_tp_delta": int(delta["pairwise_true_positives"]),
        "pairwise_fp_delta": int(delta["pairwise_false_positives"]),
        "pairwise_fn_delta": int(delta["pairwise_false_negatives"]),
        "complete_tp_delta": int(delta["complete_track_true_positives"]),
        "complete_fp_delta": int(delta["complete_track_false_positives"]),
        "complete_fn_delta": int(delta["complete_track_false_negatives"]),
        "new_pairwise_f1_micro": float(candidate_scores["pairwise_f1"]),
        "new_complete_track_f1_micro": float(candidate_scores["complete_track_f1"]),
        "baseline_pairwise_f1_micro": float(baseline_scores["pairwise_f1"]),
        "baseline_complete_track_f1_micro": float(baseline_scores["complete_track_f1"]),
        "threshold_method": str(threshold_method),
        "iou_distance_threshold": float(iou_distance_threshold),
        "cell_probability_threshold": float(cell_probability_threshold),
        "transform_type": str(transform_type),
    }


def _summary_rows(
    subject: str, rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for error_type in ("pairwise_fp", "pairwise_fn", "ALL"):
        subset = [
            row
            for row in rows
            if error_type == "ALL" or str(row.get("error_type")) == error_type
        ]
        output.append(
            {
                "subject": subject,
                "error_type": error_type,
                "n_edges": int(len(subset)),
                "median_growth_residual": _median(subset, "growth_residual"),
                "median_growth_residual_mahalanobis": _median(
                    subset, "growth_residual_mahalanobis"
                ),
                "median_radial_direction_cosine": _median(
                    subset, "radial_direction_cosine"
                ),
                "median_area_growth_residual": _median(subset, "area_growth_residual"),
                "median_local_neighbor_distortion": _median(
                    subset, "local_neighbor_distortion"
                ),
                "would_fix_complete_fn": int(
                    sum(int(row.get("would_fix_complete_fn", 0)) for row in subset)
                ),
                "would_create_complete_fp": int(
                    sum(int(row.get("would_create_complete_fp", 0)) for row in subset)
                ),
                "max_new_pairwise_f1_micro": _max(subset, "new_pairwise_f1_micro"),
                "max_new_complete_track_f1_micro": _max(
                    subset, "new_complete_track_f1_micro"
                ),
            }
        )
    return output


def _centroid_xy(
    sessions: Sequence[Track2pSession], session_index: int, suite2p_roi: int
) -> np.ndarray | None:
    if session_index < 0 or session_index >= len(sessions):
        return None
    roi_indices = _roi_indices(sessions[session_index])
    matches = np.flatnonzero(roi_indices == int(suite2p_roi))
    if matches.size == 0:
        return None
    centroids = np.asarray(
        sessions[session_index].plane_data.centroids(order="xy"), dtype=float
    ).T
    return np.asarray(centroids[int(matches[0])], dtype=float)


def _observed_area_ratio(sessions: Sequence[Track2pSession], edge: TrackEdge) -> float:
    session_a, session_b, roi_a, roi_b = edge
    area_a = _roi_area(sessions, session_a, roi_a)
    area_b = _roi_area(sessions, session_b, roi_b)
    if not np.isfinite(area_a) or not np.isfinite(area_b) or area_a <= 0.0:
        return float("nan")
    return float(area_b / area_a)


def _roi_area(
    sessions: Sequence[Track2pSession], session_index: int, suite2p_roi: int
) -> float:
    if session_index < 0 or session_index >= len(sessions):
        return float("nan")
    roi_indices = _roi_indices(sessions[session_index])
    matches = np.flatnonzero(roi_indices == int(suite2p_roi))
    if matches.size == 0:
        return float("nan")
    areas = np.asarray(sessions[session_index].plane_data.roi_areas(), dtype=float)
    return float(areas[int(matches[0])])


def _apply_affine(point_xy: np.ndarray, affine_xy: np.ndarray) -> np.ndarray:
    return np.asarray(point_xy, dtype=float) @ affine_xy[:, :2].T + affine_xy[:, 2]


def _residual_vectors(
    source_xy: np.ndarray, target_xy: np.ndarray, affine_xy: np.ndarray
) -> np.ndarray:
    predicted = source_xy @ affine_xy[:, :2].T + affine_xy[:, 2][None, :]
    return np.asarray(target_xy, dtype=float) - predicted


def _robust_residual_scale(residual_vectors: np.ndarray) -> float:
    norms = np.linalg.norm(np.asarray(residual_vectors, dtype=float), axis=1)
    if norms.size == 0:
        return 1.0
    median = float(np.median(norms))
    mad = float(np.median(np.abs(norms - median)))
    return max(float(1.4826 * mad), median, 1.0)


def _residual_covariance_inverse(
    residual_vectors: np.ndarray, residual_scale: float
) -> np.ndarray:
    residuals = np.asarray(residual_vectors, dtype=float)
    if residuals.shape[0] >= 2:
        covariance = np.cov(residuals.T)
    else:
        covariance = np.eye(2, dtype=float) * float(residual_scale) ** 2
    covariance = np.asarray(covariance, dtype=float).reshape(2, 2)
    covariance += np.eye(2, dtype=float) * 1.0e-6
    return np.linalg.pinv(covariance)


def _expected_area_ratio(affine_xy: np.ndarray) -> float:
    linear = np.asarray(affine_xy, dtype=float)[:, :2]
    determinant = float(abs(np.linalg.det(linear)))
    if not np.isfinite(determinant) or determinant <= 0.0:
        return 1.0
    return determinant


def _area_growth_residual(observed: float, expected: float) -> float:
    if (
        not np.isfinite(observed)
        or not np.isfinite(expected)
        or observed <= 0.0
        or expected <= 0.0
    ):
        return float("nan")
    return float(abs(np.log(observed / expected)))


def _radial_direction_cosine(
    sessions: Sequence[Track2pSession],
    edge: TrackEdge,
    source: np.ndarray,
    target: np.ndarray,
) -> float:
    anchor = _lower_left_anchor(sessions[edge[0]])
    radial = np.asarray(source, dtype=float) - anchor
    displacement = np.asarray(target, dtype=float) - np.asarray(source, dtype=float)
    denominator = float(np.linalg.norm(radial) * np.linalg.norm(displacement))
    if denominator <= 1.0e-12:
        return float("nan")
    return float(np.clip(np.dot(radial, displacement) / denominator, -1.0, 1.0))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1.0e-12:
        return float("nan")
    return float(np.clip(float(np.dot(left, right)) / denominator, -1.0, 1.0))


def _lower_left_anchor(session: Track2pSession) -> np.ndarray:
    masks = np.asarray(session.plane_data.roi_masks)
    height = masks.shape[-2] if masks.ndim >= 2 else 1
    return np.asarray([0.0, float(max(0, height - 1))], dtype=float)


def _local_neighbor_distortion(
    sessions: Sequence[Track2pSession],
    edge: TrackEdge,
    anchor_edges: Sequence[TrackEdge],
    *,
    expected_area_ratio: float,
) -> float:
    source = _centroid_xy(sessions, edge[0], edge[2])
    target = _centroid_xy(sessions, edge[1], edge[3])
    if source is None or target is None or len(anchor_edges) < 2:
        return float("nan")
    values: list[float] = []
    scale = float(np.sqrt(max(expected_area_ratio, 1.0e-12)))
    for anchor in anchor_edges:
        anchor_source = _centroid_xy(sessions, anchor[0], anchor[2])
        anchor_target = _centroid_xy(sessions, anchor[1], anchor[3])
        if anchor_source is None or anchor_target is None:
            continue
        source_distance = float(np.linalg.norm(source - anchor_source))
        target_distance = float(np.linalg.norm(target - anchor_target))
        if source_distance <= 1.0e-9 or target_distance <= 0.0:
            continue
        values.append(abs(float(np.log((target_distance / source_distance) / scale))))
    if not values:
        return float("nan")
    nearest = sorted(values)[:8]
    return float(np.median(np.asarray(nearest, dtype=float)))


def _local_index(values: np.ndarray, roi: int) -> int:
    matches = np.flatnonzero(np.asarray(values, dtype=int) == int(roi))
    if matches.size == 0:
        return -1
    return int(matches[0])


def _median(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [
        float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))
    ]
    if not values:
        return float("nan")
    return float(np.median(np.asarray(values, dtype=float)))


def _max(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [
        float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))
    ]
    if not values:
        return float("nan")
    return float(np.max(np.asarray(values, dtype=float)))


def _augment_with_shifted_iou_and_margins(
    rows: list[dict[str, Any]],
    sessions: Sequence[Track2pSession],
    *,
    feature_cache: _FeatureCache,
) -> list[dict[str, Any]]:
    for row in rows:
        session_a = int(row["session_a"])
        roi_a = int(row["roi_a"])
        roi_b = int(row["roi_b"])
        matrices = feature_cache.pair(session_a)
        local_a = _local_index(matrices.source_indices, roi_a)
        local_b = _local_index(matrices.target_indices, roi_b)
        if local_a < 0 or local_b < 0:
            continue
        row["shifted_iou"] = float(matrices.shifted_iou[local_a, local_b])
        row["registered_iou"] = float(matrices.registered_iou[local_a, local_b])
        row["centroid_distance"] = float(matrices.centroid_distance[local_a, local_b])
        row["area_ratio"] = float(matrices.area_ratio[local_a, local_b])
        row["row_rank"] = int(
            _rank_descending(matrices.registered_iou[local_a], selected_index=local_b)
        )
        row["column_rank"] = int(
            _rank_descending(
                matrices.registered_iou[:, local_b], selected_index=local_a
            )
        )
        row["row_margin"] = float(
            _margin_against_competitor(
                matrices.registered_iou[local_a], selected_index=local_b
            )
        )
        row["column_margin"] = float(
            _margin_against_competitor(
                matrices.registered_iou[:, local_b], selected_index=local_a
            )
        )
        row["threshold_margin"] = float(
            matrices.registered_iou[local_a, local_b] - matrices.threshold
        )
        row["cell_probability_a"] = _cell_probability(sessions, session_a, roi_a)
        row["cell_probability_b"] = _cell_probability(
            sessions, int(row["session_b"]), roi_b
        )
    return rows


def write_rows(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    output_format: Literal["csv", "json"] = "csv",
) -> None:
    """Write audit rows as CSV or JSON."""

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
    """Build the growth-field residual-audit parser."""

    parser = suffix.build_arg_parser()
    parser.prog = (
        "python -m bayescatrack.experiments.track2p_policy_growth_field_residual_audit"
    )
    parser.description = "Audit residual CoherenceSuffixTeacherRescue official errors against a label-free lower-left growth/deformation prior."
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--anchor-min-registered-iou", type=float, default=0.50)
    parser.add_argument("--anchor-min-shifted-iou", type=float, default=0.30)
    parser.add_argument("--anchor-min-cell-probability", type=float, default=0.80)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the growth-field residual audit."""

    args = build_arg_parser().parse_args(argv)
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
    cleanup_config = ComponentCleanupConfig(
        split_risk_threshold=args.split_risk_threshold,
        split_penalty=args.split_penalty,
        min_side_observations=args.min_side_observations,
        require_complete_track=args.require_complete_track,
    )
    suffix_gate = suffix.CoherenceSuffixStitchGate(
        suffix_path_length=int(args.suffix_path_length),
        min_cell_probability=float(args.min_cell_probability),
        min_area_ratio=float(args.min_area_ratio),
        max_centroid_distance=float(args.max_centroid_distance),
        min_shifted_iou=float(args.min_shifted_iou),
        min_motion_consistency=float(args.min_motion_consistency),
        min_shape_consistency=float(args.min_shape_consistency),
        max_stitches_per_subject=int(args.max_stitches_per_subject),
    )
    result = run_track2p_policy_growth_field_residual_audit(
        config,
        threshold_method=cast(ThresholdMethod, args.threshold_method),
        iou_distance_threshold=float(args.iou_distance_threshold),
        transform_type=args.transform_type,
        cell_probability_threshold=float(args.cell_probability_threshold),
        cleanup_config=cleanup_config,
        suffix_gate=suffix_gate,
        edge_top_k=int(args.edge_top_k),
        path_beam_width=int(args.path_beam_width),
        anchor_min_registered_iou=float(args.anchor_min_registered_iou),
        anchor_min_shifted_iou=float(args.anchor_min_shifted_iou),
        anchor_min_cell_probability=float(args.anchor_min_cell_probability),
    )
    rows = list(result.edge_rows)
    write_rows(
        rows,
        args.output,
        output_format=cast(Literal["csv", "json"], args.format),
    )
    if args.summary_output is not None:
        write_rows(
            result.summary_rows,
            args.summary_output,
            output_format=cast(Literal["csv", "json"], args.format),
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
