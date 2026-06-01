"""Residual official-error audit after Track2p-policy component cleanup.

The component-cleanup row is now the strongest Track2p-policy family baseline.
Further tuning should therefore start from the errors that remain under the
official scorer, not from extra gap-rescue candidates that do not affect scored
adjacent links or complete rows.  This module reruns component cleanup and
exports every remaining duplicate-aware pairwise and complete-track error with
local registration features, support flags, component context, and a coarse
mechanism bucket.
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
from bayescatrack.core.bridge import Track2pSession
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
from bayescatrack.experiments.track2p_emulation_benchmark import (
    ThresholdMethod,
    emulate_track2p_tracks,
)
from bayescatrack.experiments.track2p_policy_audit import (
    TrackEdge,
    track_edge_counter,
)
from bayescatrack.experiments.track2p_policy_benchmark import (
    TRACK2P_POLICY_DEFAULT_CELL_PROBABILITY_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_MAX_GAP,
    TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE,
    track2p_policy_config,
)
from bayescatrack.experiments.track2p_policy_component_audit import (
    ComponentCleanupConfig,
    _complete_track_status,
    _evaluated_prediction_rows,
    _mark_applied_splits,
    _normalize_int_track_matrix,
    apply_weakest_bridge_splits,
    component_audit_rows,
)
from bayescatrack.experiments.track2p_policy_pruned_benchmark import (
    Track2pPolicyPruneConfig,
    _margin_against_competitor,
    _roi_indices,
    _threshold_assigned_iou,
    _track2p_cross_iou_diagnostic_matrices,
    emulate_track2p_pruned_tracks,
)
from bayescatrack.track2p_registration import register_plane_pair
from scipy.optimize import linear_sum_assignment

TRACK2P_POLICY_COMPONENT_RESIDUAL_AUDIT_METHOD = (
    "track2p-policy-component-residual-audit"
)
ResidualErrorType = Literal["pairwise_fp", "pairwise_fn", "complete_fp", "complete_fn"]
CompleteTrack = tuple[int, ...]


@dataclass(frozen=True)
class ResidualFeature:
    """Registration/local-assignment features for one requested edge."""

    registered_iou: float = float("nan")
    centroid_distance: float = float("nan")
    area_ratio: float = float("nan")
    cell_probability_a: float = float("nan")
    cell_probability_b: float = float("nan")
    row_rank: int = -1
    column_rank: int = -1
    row_margin: float = float("nan")
    column_margin: float = float("nan")
    threshold: float = float("nan")
    threshold_margin: float = float("nan")
    assigned_by_hungarian: int = 0


@dataclass(frozen=True)
class ResidualAuditResult:
    """Residual error rows plus compact per-subject summary rows."""

    error_rows: tuple[dict[str, float | int | str], ...]
    summary_rows: tuple[dict[str, float | int | str], ...]


def run_track2p_policy_component_residual_audit(
    config: Track2pBenchmarkConfig,
    *,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    iou_distance_threshold: float = TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    transform_type: str | None = None,
    cell_probability_threshold: float | None = None,
    max_gap: int = TRACK2P_POLICY_DEFAULT_MAX_GAP,
    cleanup_config: ComponentCleanupConfig | None = None,
    feature_mode: Literal["policy-diagnostics", "registered-subset", "none"] = (
        "policy-diagnostics"
    ),
) -> ResidualAuditResult:
    """Return all official residual errors after component cleanup."""

    policy_config = track2p_policy_config(
        config,
        transform_type=transform_type,
        cell_probability_threshold=cell_probability_threshold,
        max_gap=max_gap,
    )
    subject_dirs = discover_subject_dirs(policy_config.data)
    if not subject_dirs:
        raise ValueError(
            f"No Track2p-style subject directories found under {policy_config.data}"
        )

    cleanup_config = cleanup_config or ComponentCleanupConfig()
    all_rows: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | int | str]] = []
    for subject_dir in subject_dirs:
        reference = _load_reference_for_subject(
            subject_dir, data_root=policy_config.data, config=policy_config
        )
        _validate_reference_for_benchmark(
            reference, subject_dir=subject_dir, config=policy_config
        )
        if reference.source != GROUND_TRUTH_REFERENCE_SOURCE:
            raise ValueError(
                "Track2p-policy component residual audit requires independent "
                "manual GT references"
            )
        sessions = _load_subject_sessions(subject_dir, policy_config)
        _validate_reference_roi_indices(reference, sessions)
        reference_tracks = _reference_matrix(
            reference, curated_only=policy_config.curated_only
        )

        policy_prediction = emulate_track2p_pruned_tracks(
            sessions,
            transform_type=policy_config.transform_type,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
            prune_config=_no_prune_config(),
        )
        policy_full = _normalize_int_track_matrix(policy_prediction.tracks)
        policy_eval, reference_eval, evaluated_track_ids = _evaluated_prediction_rows(
            policy_full, reference_tracks, config=policy_config
        )
        component_rows = component_audit_rows(
            policy_eval,
            reference_eval,
            sessions=sessions,
            diagnostics=policy_prediction.diagnostics,
            subject=subject_dir.name,
            config=cleanup_config,
            track_ids=evaluated_track_ids,
            seed_session=policy_config.seed_session,
        )
        cleaned_full = apply_weakest_bridge_splits(
            policy_full, _mark_applied_splits(component_rows, apply_splits=True)
        )
        cleaned_eval, reference_eval, _ = _evaluated_prediction_rows(
            cleaned_full, reference_tracks, config=policy_config
        )
        track2p_eval = _track2p_baseline_eval(
            subject_dir,
            reference_tracks,
            config=policy_config,
        )
        gap_full = _normalize_int_track_matrix(
            emulate_track2p_tracks(
                sessions,
                transform_type=policy_config.transform_type,
                threshold_method=threshold_method,
                iou_distance_threshold=float(iou_distance_threshold),
                max_gap=int(policy_config.max_gap),
            )
        )
        gap_eval, _, _ = _evaluated_prediction_rows(
            gap_full, reference_tracks, config=policy_config
        )

        feature_index = _residual_feature_index(
            sessions,
            policy_prediction.diagnostics,
            cleaned_eval,
            reference_eval,
            transform_type=policy_config.transform_type,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
            feature_mode=feature_mode,
        )
        subject_rows = residual_error_rows(
            cleaned_eval,
            reference_eval,
            subject=subject_dir.name,
            sessions=sessions,
            track2p_tracks=track2p_eval,
            policy_tracks=policy_eval,
            gap_tracks=gap_eval,
            before_cleanup_tracks=policy_eval,
            feature_index=feature_index,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
            cell_probability_threshold=float(policy_config.cell_probability_threshold),
            transform_type=policy_config.transform_type,
            seed_session=policy_config.seed_session,
        )
        all_rows.extend(subject_rows)
        summary_rows.append(
            _summary_row(
                subject_dir.name,
                subject_rows,
                threshold_method=threshold_method,
                iou_distance_threshold=float(iou_distance_threshold),
            )
        )

    summary_rows.append(
        _summary_row(
            "ALL",
            all_rows,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
        )
    )
    return ResidualAuditResult(tuple(all_rows), tuple(summary_rows))


def residual_error_rows(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    subject: str = "",
    sessions: Sequence[Track2pSession] = (),
    track2p_tracks: Any | None = None,
    policy_tracks: Any | None = None,
    gap_tracks: Any | None = None,
    before_cleanup_tracks: Any | None = None,
    feature_index: Mapping[TrackEdge, ResidualFeature] | None = None,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    iou_distance_threshold: float = TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    cell_probability_threshold: float = TRACK2P_POLICY_DEFAULT_CELL_PROBABILITY_THRESHOLD,
    transform_type: str = TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE,
    seed_session: int = 0,
) -> list[dict[str, float | int | str]]:
    """Return duplicate-aware pairwise and complete residual error rows."""

    predicted = _normalize_int_track_matrix(predicted_track_matrix)
    reference = _normalize_int_track_matrix(reference_track_matrix)
    track2p = _optional_track_matrix(track2p_tracks, like=predicted)
    policy = _optional_track_matrix(policy_tracks, like=predicted)
    gap = _optional_track_matrix(gap_tracks, like=predicted)
    before_cleanup = _optional_track_matrix(before_cleanup_tracks, like=predicted)
    features = dict(feature_index or {})

    rows: list[dict[str, float | int | str]] = []
    rows.extend(
        _pairwise_residual_rows(
            predicted,
            reference,
            subject=subject,
            sessions=sessions,
            track2p_tracks=track2p,
            policy_tracks=policy,
            gap_tracks=gap,
            before_cleanup_tracks=before_cleanup,
            feature_index=features,
            threshold_method=threshold_method,
            iou_distance_threshold=iou_distance_threshold,
            cell_probability_threshold=cell_probability_threshold,
            transform_type=transform_type,
            seed_session=seed_session,
        )
    )
    rows.extend(
        _complete_residual_rows(
            predicted,
            reference,
            subject=subject,
            sessions=sessions,
            track2p_tracks=track2p,
            policy_tracks=policy,
            gap_tracks=gap,
            before_cleanup_tracks=before_cleanup,
            feature_index=features,
            threshold_method=threshold_method,
            iou_distance_threshold=iou_distance_threshold,
            cell_probability_threshold=cell_probability_threshold,
            transform_type=transform_type,
            seed_session=seed_session,
        )
    )
    return rows


def _pairwise_residual_rows(
    predicted: np.ndarray,
    reference: np.ndarray,
    *,
    subject: str,
    sessions: Sequence[Track2pSession],
    track2p_tracks: np.ndarray,
    policy_tracks: np.ndarray,
    gap_tracks: np.ndarray,
    before_cleanup_tracks: np.ndarray,
    feature_index: Mapping[TrackEdge, ResidualFeature],
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
    cell_probability_threshold: float,
    transform_type: str,
    seed_session: int,
) -> list[dict[str, float | int | str]]:
    predicted_counts = track_edge_counter(predicted)
    reference_counts = track_edge_counter(reference)
    track2p_counts = track_edge_counter(track2p_tracks)
    policy_counts = track_edge_counter(policy_tracks)
    gap_counts = track_edge_counter(gap_tracks)
    before_cleanup_counts = track_edge_counter(before_cleanup_tracks)
    contexts = _component_contexts(predicted, reference, seed_session=seed_session)
    component_by_edge = _component_context_by_edge(predicted, contexts)

    rows: list[dict[str, float | int | str]] = []
    for edge in sorted(
        set(predicted_counts) | set(reference_counts),
        key=lambda item: (item[0] == seed_session, item),
    ):
        predicted_count = int(predicted_counts.get(edge, 0))
        reference_count = int(reference_counts.get(edge, 0))
        matched = min(predicted_count, reference_count)
        false_positive = max(0, predicted_count - matched)
        false_negative = max(0, reference_count - matched)
        for occurrence in range(false_positive):
            rows.append(
                _edge_error_row(
                    edge,
                    "pairwise_fp",
                    occurrence_index=occurrence,
                    subject=subject,
                    sessions=sessions,
                    reference=reference,
                    predicted=predicted,
                    track2p_supported=track2p_counts.get(edge, 0) > 0,
                    policy_supported=policy_counts.get(edge, 0) > 0,
                    gap_supported=gap_counts.get(edge, 0) > 0,
                    component_cleanup_affected=(
                        before_cleanup_counts.get(edge, 0)
                        != predicted_counts.get(edge, 0)
                    ),
                    feature=feature_index.get(edge),
                    context=component_by_edge.get(edge),
                    threshold_method=threshold_method,
                    iou_distance_threshold=iou_distance_threshold,
                    cell_probability_threshold=cell_probability_threshold,
                    transform_type=transform_type,
                    reason_bucket=_pairwise_fp_reason(
                        edge,
                        predicted_counts=predicted_counts,
                        reference_counts=reference_counts,
                        context=component_by_edge.get(edge),
                    ),
                )
            )
        for occurrence in range(false_negative):
            context = _nearest_component_for_reference_edge(
                edge, predicted, contexts, seed_session=seed_session
            )
            rows.append(
                _edge_error_row(
                    edge,
                    "pairwise_fn",
                    occurrence_index=occurrence,
                    subject=subject,
                    sessions=sessions,
                    reference=reference,
                    predicted=predicted,
                    track2p_supported=track2p_counts.get(edge, 0) > 0,
                    policy_supported=policy_counts.get(edge, 0) > 0,
                    gap_supported=gap_counts.get(edge, 0) > 0,
                    component_cleanup_affected=(
                        before_cleanup_counts.get(edge, 0)
                        != predicted_counts.get(edge, 0)
                    ),
                    feature=feature_index.get(edge),
                    context=context,
                    threshold_method=threshold_method,
                    iou_distance_threshold=iou_distance_threshold,
                    cell_probability_threshold=cell_probability_threshold,
                    transform_type=transform_type,
                    reason_bucket=_pairwise_fn_reason(
                        edge,
                        predicted=predicted,
                        reference=reference,
                        policy_supported=policy_counts.get(edge, 0) > 0,
                        gap_supported=gap_counts.get(edge, 0) > 0,
                        component_cleanup_affected=(
                            before_cleanup_counts.get(edge, 0)
                            != predicted_counts.get(edge, 0)
                        ),
                        seed_session=seed_session,
                    ),
                )
            )
    return rows


def _complete_residual_rows(
    predicted: np.ndarray,
    reference: np.ndarray,
    *,
    subject: str,
    sessions: Sequence[Track2pSession],
    track2p_tracks: np.ndarray,
    policy_tracks: np.ndarray,
    gap_tracks: np.ndarray,
    before_cleanup_tracks: np.ndarray,
    feature_index: Mapping[TrackEdge, ResidualFeature],
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
    cell_probability_threshold: float,
    transform_type: str,
    seed_session: int,
) -> list[dict[str, float | int | str]]:
    predicted_counts = _complete_track_counter(predicted)
    reference_counts = _complete_track_counter(reference)
    track2p_counts = _complete_track_counter(track2p_tracks)
    policy_counts = _complete_track_counter(policy_tracks)
    gap_counts = _complete_track_counter(gap_tracks)
    before_cleanup_counts = _complete_track_counter(before_cleanup_tracks)
    contexts = _component_contexts(predicted, reference, seed_session=seed_session)
    rows: list[dict[str, float | int | str]] = []
    for track in sorted(set(predicted_counts) | set(reference_counts)):
        predicted_count = int(predicted_counts.get(track, 0))
        reference_count = int(reference_counts.get(track, 0))
        matched = min(predicted_count, reference_count)
        false_positive = max(0, predicted_count - matched)
        false_negative = max(0, reference_count - matched)
        for occurrence in range(false_positive):
            component = _component_context_for_complete_track(
                track, predicted, contexts
            )
            rows.append(
                _complete_error_row(
                    track,
                    "complete_fp",
                    occurrence_index=occurrence,
                    subject=subject,
                    sessions=sessions,
                    reference=reference,
                    predicted=predicted,
                    track2p_supported=track2p_counts.get(track, 0) > 0,
                    policy_supported=policy_counts.get(track, 0) > 0,
                    gap_supported=gap_counts.get(track, 0) > 0,
                    component_cleanup_affected=(
                        before_cleanup_counts.get(track, 0)
                        != predicted_counts.get(track, 0)
                    ),
                    feature_index=feature_index,
                    context=component,
                    threshold_method=threshold_method,
                    iou_distance_threshold=iou_distance_threshold,
                    cell_probability_threshold=cell_probability_threshold,
                    transform_type=transform_type,
                    reason_bucket=_complete_fp_reason(track, reference),
                )
            )
        for occurrence in range(false_negative):
            component = _nearest_component_for_reference_track(
                track, predicted, contexts, seed_session=seed_session
            )
            rows.append(
                _complete_error_row(
                    track,
                    "complete_fn",
                    occurrence_index=occurrence,
                    subject=subject,
                    sessions=sessions,
                    reference=reference,
                    predicted=predicted,
                    track2p_supported=track2p_counts.get(track, 0) > 0,
                    policy_supported=policy_counts.get(track, 0) > 0,
                    gap_supported=gap_counts.get(track, 0) > 0,
                    component_cleanup_affected=(
                        before_cleanup_counts.get(track, 0)
                        != predicted_counts.get(track, 0)
                    ),
                    feature_index=feature_index,
                    context=component,
                    threshold_method=threshold_method,
                    iou_distance_threshold=iou_distance_threshold,
                    cell_probability_threshold=cell_probability_threshold,
                    transform_type=transform_type,
                    reason_bucket=_complete_fn_reason(
                        track,
                        predicted=predicted,
                        policy_supported=policy_counts.get(track, 0) > 0,
                        gap_supported=gap_counts.get(track, 0) > 0,
                    ),
                )
            )
    return rows


def _edge_error_row(
    edge: TrackEdge,
    error_type: ResidualErrorType,
    *,
    occurrence_index: int,
    subject: str,
    sessions: Sequence[Track2pSession],
    reference: np.ndarray,
    predicted: np.ndarray,
    track2p_supported: bool,
    policy_supported: bool,
    gap_supported: bool,
    component_cleanup_affected: bool,
    feature: ResidualFeature | None,
    context: Mapping[str, int | str] | None,
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
    cell_probability_threshold: float,
    transform_type: str,
    reason_bucket: str,
) -> dict[str, float | int | str]:
    session_a, session_b, roi_a, roi_b = edge
    feature = feature or ResidualFeature()
    context = context or _empty_component_context()
    nearest_gt = _nearest_reference_track_for_edge(edge, reference)
    return {
        "subject": subject,
        "error_type": error_type,
        "track_id_or_edge": _edge_id(edge),
        "occurrence_index": int(occurrence_index),
        "session_a": int(session_a),
        "session_b": int(session_b),
        "session_a_name": _session_name(sessions, session_a),
        "session_b_name": _session_name(sessions, session_b),
        "sessions_involved": f"{session_a}->{session_b}",
        "roi_a": int(roi_a),
        "roi_b": int(roi_b),
        "roi_ids_involved": f"{roi_a}->{roi_b}",
        "is_track2p_supported": int(track2p_supported),
        "is_policy_supported": int(policy_supported),
        "is_gap_rescue_supported": int(gap_supported),
        "is_component_cleanup_affected": int(component_cleanup_affected),
        "registered_iou": float(feature.registered_iou),
        "centroid_distance": float(feature.centroid_distance),
        "area_ratio": float(feature.area_ratio),
        "row_rank": int(feature.row_rank),
        "column_rank": int(feature.column_rank),
        "row_margin": float(feature.row_margin),
        "column_margin": float(feature.column_margin),
        "threshold": float(feature.threshold),
        "threshold_margin": float(feature.threshold_margin),
        "assigned_by_hungarian": int(feature.assigned_by_hungarian),
        "cell_probability_a": _cell_probability(sessions, session_a, roi_a),
        "cell_probability_b": _cell_probability(sessions, session_b, roi_b),
        "component_id": int(context["component_id"]),
        "component_size": int(context["component_size"]),
        "complete_track_status": str(context["complete_track_status"]),
        "nearest_gt_track_id": int(nearest_gt),
        "nearest_predicted_track_id": _nearest_predicted_track_for_edge(
            edge, predicted
        ),
        "reason_bucket": reason_bucket,
        "threshold_method": str(threshold_method),
        "iou_distance_threshold": float(iou_distance_threshold),
        "cell_probability_threshold": float(cell_probability_threshold),
        "transform_type": str(transform_type),
    }


def _complete_error_row(
    track: CompleteTrack,
    error_type: ResidualErrorType,
    *,
    occurrence_index: int,
    subject: str,
    sessions: Sequence[Track2pSession],
    reference: np.ndarray,
    predicted: np.ndarray,
    track2p_supported: bool,
    policy_supported: bool,
    gap_supported: bool,
    component_cleanup_affected: bool,
    feature_index: Mapping[TrackEdge, ResidualFeature],
    context: Mapping[str, int | str] | None,
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
    cell_probability_threshold: float,
    transform_type: str,
    reason_bucket: str,
) -> dict[str, float | int | str]:
    representative_edge = _representative_track_edge(track, feature_index)
    feature = feature_index.get(representative_edge, ResidualFeature())
    context = context or _empty_component_context()
    nearest_gt = _nearest_reference_track_for_track(track, reference)
    return {
        "subject": subject,
        "error_type": error_type,
        "track_id_or_edge": _track_id(track),
        "occurrence_index": int(occurrence_index),
        "session_a": int(representative_edge[0]) if representative_edge else -1,
        "session_b": int(representative_edge[1]) if representative_edge else -1,
        "session_a_name": (
            _session_name(sessions, representative_edge[0])
            if representative_edge
            else ""
        ),
        "session_b_name": (
            _session_name(sessions, representative_edge[1])
            if representative_edge
            else ""
        ),
        "sessions_involved": ",".join(str(index) for index in range(len(track))),
        "roi_a": int(representative_edge[2]) if representative_edge else -1,
        "roi_b": int(representative_edge[3]) if representative_edge else -1,
        "roi_ids_involved": ",".join(str(value) for value in track),
        "is_track2p_supported": int(track2p_supported),
        "is_policy_supported": int(policy_supported),
        "is_gap_rescue_supported": int(gap_supported),
        "is_component_cleanup_affected": int(component_cleanup_affected),
        "registered_iou": float(feature.registered_iou),
        "centroid_distance": float(feature.centroid_distance),
        "area_ratio": float(feature.area_ratio),
        "row_rank": int(feature.row_rank),
        "column_rank": int(feature.column_rank),
        "row_margin": float(feature.row_margin),
        "column_margin": float(feature.column_margin),
        "threshold": float(feature.threshold),
        "threshold_margin": float(feature.threshold_margin),
        "assigned_by_hungarian": int(feature.assigned_by_hungarian),
        "cell_probability_a": (
            _cell_probability(sessions, representative_edge[0], representative_edge[2])
            if representative_edge
            else float("nan")
        ),
        "cell_probability_b": (
            _cell_probability(sessions, representative_edge[1], representative_edge[3])
            if representative_edge
            else float("nan")
        ),
        "component_id": int(context["component_id"]),
        "component_size": int(context["component_size"]),
        "complete_track_status": str(context["complete_track_status"]),
        "nearest_gt_track_id": int(nearest_gt),
        "nearest_predicted_track_id": _nearest_predicted_track_for_track(
            track, predicted
        ),
        "reason_bucket": reason_bucket,
        "threshold_method": str(threshold_method),
        "iou_distance_threshold": float(iou_distance_threshold),
        "cell_probability_threshold": float(cell_probability_threshold),
        "transform_type": str(transform_type),
    }


def _feature_subset_for_edges(
    sessions: Sequence[Track2pSession],
    edges: Sequence[TrackEdge] | set[TrackEdge],
    *,
    transform_type: str,
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
) -> dict[TrackEdge, ResidualFeature]:
    requested_by_pair: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
    for session_a, session_b, roi_a, roi_b in edges:
        if session_b != session_a + 1:
            continue
        requested_by_pair[(session_a, session_b)].add((roi_a, roi_b))
    output: dict[TrackEdge, ResidualFeature] = {}
    for (session_a, session_b), requested in requested_by_pair.items():
        output.update(
            _pair_feature_subset(
                sessions[session_a],
                sessions[session_b],
                session_a=session_a,
                requested_edges=requested,
                transform_type=transform_type,
                threshold_method=threshold_method,
                iou_distance_threshold=iou_distance_threshold,
            )
        )
    return output


def _residual_feature_index(
    sessions: Sequence[Track2pSession],
    diagnostics: Sequence[Any],
    predicted: np.ndarray,
    reference: np.ndarray,
    *,
    transform_type: str,
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
    feature_mode: Literal["policy-diagnostics", "registered-subset", "none"],
) -> dict[TrackEdge, ResidualFeature]:
    if feature_mode == "none":
        return {}
    if feature_mode == "policy-diagnostics":
        return _feature_index_from_policy_diagnostics(sessions, diagnostics)
    if feature_mode == "registered-subset":
        return _feature_subset_for_edges(
            sessions,
            _residual_pairwise_edge_set(predicted, reference),
            transform_type=transform_type,
            threshold_method=threshold_method,
            iou_distance_threshold=iou_distance_threshold,
        )
    raise ValueError(
        "feature_mode must be 'policy-diagnostics', 'registered-subset', or 'none'"
    )


def _feature_index_from_policy_diagnostics(
    sessions: Sequence[Track2pSession], diagnostics: Sequence[Any]
) -> dict[TrackEdge, ResidualFeature]:
    roi_indices_by_session = [_roi_indices(session) for session in sessions]
    output: dict[TrackEdge, ResidualFeature] = {}
    for diagnostic in diagnostics:
        session_index = int(diagnostic.session_index)
        if session_index < 0 or session_index + 1 >= len(roi_indices_by_session):
            continue
        source_indices = roi_indices_by_session[session_index]
        target_indices = roi_indices_by_session[session_index + 1]
        local_a = int(diagnostic.local_roi_a)
        local_b = int(diagnostic.local_roi_b)
        if local_a >= len(source_indices) or local_b >= len(target_indices):
            continue
        output[
            (
                session_index,
                session_index + 1,
                int(source_indices[local_a]),
                int(target_indices[local_b]),
            )
        ] = ResidualFeature(
            registered_iou=float(diagnostic.assigned_iou),
            centroid_distance=float(diagnostic.centroid_distance),
            area_ratio=float(diagnostic.area_ratio),
            cell_probability_a=_cell_probability(
                sessions, session_index, int(source_indices[local_a])
            ),
            cell_probability_b=_cell_probability(
                sessions, session_index + 1, int(target_indices[local_b])
            ),
            row_rank=1,
            column_rank=1,
            row_margin=float(diagnostic.row_margin),
            column_margin=float(diagnostic.column_margin),
            threshold=float(diagnostic.threshold),
            threshold_margin=float(diagnostic.threshold_margin),
            assigned_by_hungarian=1,
        )
    return output


def _pair_feature_subset(
    reference_session: Track2pSession,
    moving_session: Track2pSession,
    *,
    session_a: int,
    requested_edges: set[tuple[int, int]],
    transform_type: str,
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
) -> dict[TrackEdge, ResidualFeature]:
    source_indices = _roi_indices(reference_session)
    target_indices = _roi_indices(moving_session)
    source_local = {int(roi): index for index, roi in enumerate(source_indices)}
    target_local = {int(roi): index for index, roi in enumerate(target_indices)}
    registered = register_plane_pair(
        reference_session.plane_data,
        moving_session.plane_data,
        transform_type=transform_type,
    )
    iou, distances, area_ratios = _track2p_cross_iou_diagnostic_matrices(
        np.asarray(reference_session.plane_data.roi_masks) > 0,
        np.asarray(registered.roi_masks) > 0,
        distance_threshold=float(iou_distance_threshold),
    )
    if iou.size == 0:
        return {}
    row_ind, col_ind = linear_sum_assignment(1.0 - iou)
    assigned_pairs = {(int(row), int(column)) for row, column in zip(row_ind, col_ind)}
    threshold = _threshold_assigned_iou(iou[row_ind, col_ind], method=threshold_method)
    output: dict[TrackEdge, ResidualFeature] = {}
    for roi_a, roi_b in requested_edges:
        local_a = source_local.get(int(roi_a))
        local_b = target_local.get(int(roi_b))
        if local_a is None or local_b is None:
            continue
        value = float(iou[local_a, local_b])
        output[(session_a, session_a + 1, int(roi_a), int(roi_b))] = ResidualFeature(
            registered_iou=value,
            centroid_distance=float(distances[local_a, local_b]),
            area_ratio=float(area_ratios[local_a, local_b]),
            cell_probability_a=_local_cell_probability(reference_session, local_a),
            cell_probability_b=_local_cell_probability(moving_session, local_b),
            row_rank=_rank_descending(iou[local_a, :], selected_index=local_b),
            column_rank=_rank_descending(iou[:, local_b], selected_index=local_a),
            row_margin=_margin_against_competitor(
                iou[local_a, :], selected_index=local_b
            ),
            column_margin=_margin_against_competitor(
                iou[:, local_b], selected_index=local_a
            ),
            threshold=float(threshold),
            threshold_margin=value - float(threshold),
            assigned_by_hungarian=int((local_a, local_b) in assigned_pairs),
        )
    return output


def _track2p_baseline_eval(
    subject_dir: Path,
    reference_tracks: np.ndarray,
    *,
    config: Track2pBenchmarkConfig,
) -> np.ndarray:
    baseline_config = replace(config, method="track2p-baseline")
    predicted, _variant = _predict_subject_tracks(subject_dir, baseline_config)
    predicted_eval, _reference_eval, _ = _evaluated_prediction_rows(
        predicted, reference_tracks, config=config
    )
    return predicted_eval


def _residual_pairwise_edge_set(
    predicted: np.ndarray, reference: np.ndarray
) -> set[TrackEdge]:
    predicted_counts = track_edge_counter(predicted)
    reference_counts = track_edge_counter(reference)
    return {
        edge
        for edge in set(predicted_counts) | set(reference_counts)
        if predicted_counts.get(edge, 0) != reference_counts.get(edge, 0)
    }


def _component_contexts(
    predicted: np.ndarray, reference: np.ndarray, *, seed_session: int
) -> dict[int, dict[str, int | str]]:
    reference_complete_counts = _complete_track_counter(reference)
    contexts: dict[int, dict[str, int | str]] = {}
    for track_id, row in enumerate(predicted):
        contexts[int(track_id)] = {
            "component_id": int(track_id),
            "component_size": int(np.sum(row >= 0)),
            "complete_track_status": _complete_track_status(
                row, reference_complete_counts
            ),
            "seed_roi": (
                int(row[seed_session])
                if 0 <= seed_session < row.size and row[seed_session] >= 0
                else -1
            ),
        }
    return contexts


def _component_context_by_edge(
    predicted: np.ndarray, contexts: Mapping[int, Mapping[str, int | str]]
) -> dict[TrackEdge, Mapping[str, int | str]]:
    output: dict[TrackEdge, Mapping[str, int | str]] = {}
    for track_id, row in enumerate(predicted):
        for edge in _consecutive_edges(row):
            output.setdefault(edge, contexts[int(track_id)])
    return output


def _nearest_component_for_reference_edge(
    edge: TrackEdge,
    predicted: np.ndarray,
    contexts: Mapping[int, Mapping[str, int | str]],
    *,
    seed_session: int,
) -> Mapping[str, int | str] | None:
    _session_a, _session_b, roi_a, roi_b = edge
    for track_id, row in enumerate(predicted):
        if roi_a in row or roi_b in row:
            return contexts.get(int(track_id))
    if 0 <= seed_session < predicted.shape[1]:
        for track_id, row in enumerate(predicted):
            if row[seed_session] == roi_a:
                return contexts.get(int(track_id))
    return None


def _component_context_for_complete_track(
    track: CompleteTrack,
    predicted: np.ndarray,
    contexts: Mapping[int, Mapping[str, int | str]],
) -> Mapping[str, int | str] | None:
    for track_id, row in enumerate(predicted):
        if tuple(int(value) for value in row) == tuple(track):
            return contexts.get(int(track_id))
    return None


def _nearest_component_for_reference_track(
    track: CompleteTrack,
    predicted: np.ndarray,
    contexts: Mapping[int, Mapping[str, int | str]],
    *,
    seed_session: int,
) -> Mapping[str, int | str] | None:
    best_track_id = _nearest_predicted_track_for_track(track, predicted)
    if best_track_id >= 0:
        return contexts.get(best_track_id)
    if 0 <= seed_session < len(track):
        seed_roi = int(track[seed_session])
        for track_id, row in enumerate(predicted):
            if 0 <= seed_session < row.size and row[seed_session] == seed_roi:
                return contexts.get(int(track_id))
    return None


def _pairwise_fp_reason(
    edge: TrackEdge,
    *,
    predicted_counts: Mapping[TrackEdge, int],
    reference_counts: Mapping[TrackEdge, int],
    context: Mapping[str, int | str] | None,
) -> str:
    session_a, session_b, roi_a, roi_b = edge
    if predicted_counts.get(edge, 0) > 1:
        return "duplicate target/source conflict"
    for ref_a, ref_b, ref_source, ref_target in reference_counts:
        if ref_a != session_a or ref_b != session_b:
            continue
        if ref_source == roi_a and ref_target != roi_b:
            return "wrong target selected"
        if ref_source != roi_a and ref_target == roi_b:
            return "duplicate target/source conflict"
    if context and context.get("complete_track_status") == "false_positive":
        return "extra continuation / over-merged component"
    return "ambiguous/manual-GT sparse case"


def _pairwise_fn_reason(
    edge: TrackEdge,
    *,
    predicted: np.ndarray,
    reference: np.ndarray,
    policy_supported: bool,
    gap_supported: bool,
    component_cleanup_affected: bool,
    seed_session: int,
) -> str:
    session_a, session_b, roi_a, roi_b = edge
    if component_cleanup_affected and policy_supported:
        return "component cleanup removed supported edge"
    if gap_supported:
        return "gap evidence not converted into adjacent scored edge"
    for row in predicted:
        if row[session_a] == roi_a and row[session_b] >= 0 and row[session_b] != roi_b:
            return "wrong target selected"
        if row[session_b] == roi_b and row[session_a] >= 0 and row[session_a] != roi_a:
            return "duplicate target/source conflict"

    # The source ROI of a residual edge can belong to any session.  A previous
    # audit bucket checked that ROI against the seed-session column directly,
    # which can falsely classify ordinary downstream missed edges as
    # ``missing seed-session ROI`` when ``session_a != seed_session``.  Recover
    # the seed ROI from the reference track containing this edge and test that
    # seed ROI instead.
    seed_roi = _reference_seed_roi_for_edge(edge, reference, seed_session=seed_session)
    if (
        seed_roi >= 0
        and 0 <= seed_session < predicted.shape[1]
        and not np.any(predicted[:, seed_session] == seed_roi)
    ):
        return "missing seed-session ROI"
    return "missed valid adjacent edge"


def _reference_seed_roi_for_edge(
    edge: TrackEdge, reference: np.ndarray, *, seed_session: int
) -> int:
    if seed_session < 0 or seed_session >= reference.shape[1]:
        return -1
    for row in reference:
        if edge in set(_consecutive_edges(row)):
            return int(row[seed_session])
    return -1


def _complete_fp_reason(track: CompleteTrack, reference: np.ndarray) -> str:
    predicted_edges = set(_consecutive_edges(np.asarray(track, dtype=int)))
    reference_edges = track_edge_counter(reference)
    if any(edge not in reference_edges for edge in predicted_edges):
        return "extra continuation / over-merged component"
    return "ambiguous/manual-GT sparse case"


def _complete_fn_reason(
    track: CompleteTrack,
    *,
    predicted: np.ndarray,
    policy_supported: bool,
    gap_supported: bool,
) -> str:
    if policy_supported:
        return "component cleanup removed supported edge"
    if gap_supported:
        return "gap evidence not converted into adjacent scored edge"
    nearest = _nearest_predicted_track_for_track(track, predicted)
    if nearest >= 0:
        return "fragmented GT track"
    return "missing seed-session ROI"


def _representative_track_edge(
    track: CompleteTrack, feature_index: Mapping[TrackEdge, ResidualFeature]
) -> TrackEdge | None:
    edges = tuple(_consecutive_edges(np.asarray(track, dtype=int)))
    if not edges:
        return None
    with_features = tuple(edge for edge in edges if edge in feature_index)
    if not with_features:
        return edges[0]
    return min(
        with_features,
        key=lambda edge: _nan_to_inf(feature_index[edge].threshold_margin),
    )


def _nearest_reference_track_for_edge(edge: TrackEdge, reference: np.ndarray) -> int:
    for track_id, row in enumerate(reference):
        if edge in set(_consecutive_edges(row)):
            return int(track_id)
    return _nearest_track_by_roi_overlap(edge[2:], reference)


def _nearest_predicted_track_for_edge(edge: TrackEdge, predicted: np.ndarray) -> int:
    return _nearest_track_by_roi_overlap(edge[2:], predicted)


def _nearest_reference_track_for_track(
    track: CompleteTrack, reference: np.ndarray
) -> int:
    return _nearest_track_by_roi_overlap(track, reference)


def _nearest_predicted_track_for_track(
    track: CompleteTrack, predicted: np.ndarray
) -> int:
    return _nearest_track_by_roi_overlap(track, predicted)


def _nearest_track_by_roi_overlap(values: Sequence[int], matrix: np.ndarray) -> int:
    best_track_id = -1
    best_overlap = 0
    value_set = {int(value) for value in values if int(value) >= 0}
    for track_id, row in enumerate(matrix):
        overlap = len(value_set & {int(value) for value in row if value >= 0})
        if overlap > best_overlap:
            best_overlap = overlap
            best_track_id = int(track_id)
    return best_track_id


def _consecutive_edges(row: np.ndarray) -> tuple[TrackEdge, ...]:
    edges: list[TrackEdge] = []
    for session_index in range(max(0, row.size - 1)):
        source = row[session_index]
        target = row[session_index + 1]
        if source >= 0 and target >= 0:
            edges.append((session_index, session_index + 1, int(source), int(target)))
    return tuple(edges)


def _complete_track_counter(track_matrix: np.ndarray) -> Counter[CompleteTrack]:
    counter: Counter[CompleteTrack] = Counter()
    for row in _normalize_int_track_matrix(track_matrix):
        if np.all(row >= 0):
            counter[tuple(int(value) for value in row)] += 1
    return counter


def _optional_track_matrix(value: Any | None, *, like: np.ndarray) -> np.ndarray:
    if value is None:
        return np.full((0, like.shape[1]), -1, dtype=int)
    return _normalize_int_track_matrix(value)


def _empty_component_context() -> dict[str, int | str]:
    return {
        "component_id": -1,
        "component_size": 0,
        "complete_track_status": "absent",
        "seed_roi": -1,
    }


def _cell_probability(
    sessions: Sequence[Track2pSession], session_index: int, suite2p_roi: int
) -> float:
    if session_index < 0 or session_index >= len(sessions):
        return float("nan")
    probabilities = sessions[session_index].plane_data.cell_probabilities
    if probabilities is None:
        return float("nan")
    roi_indices = _roi_indices(sessions[session_index])
    matches = np.flatnonzero(roi_indices == int(suite2p_roi))
    if matches.size == 0:
        return float("nan")
    return float(np.asarray(probabilities, dtype=float)[int(matches[0])])


def _local_cell_probability(session: Track2pSession, local_roi_index: int) -> float:
    probabilities = session.plane_data.cell_probabilities
    if probabilities is None:
        return float("nan")
    values = np.asarray(probabilities, dtype=float)
    index = int(local_roi_index)
    if index < 0 or index >= values.size:
        return float("nan")
    return float(values[index])


def _rank_descending(values: np.ndarray, *, selected_index: int) -> int:
    values = np.asarray(values, dtype=float).reshape(-1)
    selected = float(values[int(selected_index)])
    return int(1 + np.sum(values > selected))


def _session_name(sessions: Sequence[Track2pSession], session_index: int) -> str:
    if 0 <= session_index < len(sessions):
        return str(sessions[session_index].session_name)
    return str(session_index)


def _edge_id(edge: TrackEdge) -> str:
    session_a, session_b, roi_a, roi_b = edge
    return f"{session_a}:{roi_a}->{session_b}:{roi_b}"


def _track_id(track: CompleteTrack) -> str:
    return ",".join(str(value) for value in track)


def _nan_to_inf(value: float) -> float:
    return float("inf") if not np.isfinite(value) else float(value)


def _summary_row(
    subject: str,
    rows: Sequence[Mapping[str, float | int | str]],
    *,
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
) -> dict[str, float | int | str]:
    counts = Counter(str(row["error_type"]) for row in rows)
    return {
        "subject": subject,
        "pairwise_false_positives": int(counts["pairwise_fp"]),
        "pairwise_false_negatives": int(counts["pairwise_fn"]),
        "complete_track_false_positives": int(counts["complete_fp"]),
        "complete_track_false_negatives": int(counts["complete_fn"]),
        "total_residual_errors": int(len(rows)),
        "track2p_supported_errors": int(
            sum(int(row.get("is_track2p_supported", 0)) for row in rows)
        ),
        "policy_supported_errors": int(
            sum(int(row.get("is_policy_supported", 0)) for row in rows)
        ),
        "gap_rescue_supported_errors": int(
            sum(int(row.get("is_gap_rescue_supported", 0)) for row in rows)
        ),
        "component_cleanup_affected_errors": int(
            sum(int(row.get("is_component_cleanup_affected", 0)) for row in rows)
        ),
        "threshold_method": str(threshold_method),
        "iou_distance_threshold": float(iou_distance_threshold),
    }


def write_residual_rows(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    output_format: Literal["csv", "json"] = "csv",
) -> None:
    """Write residual audit rows as CSV or JSON."""

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
    """Build the command-line parser for component-cleanup residual audit."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-policy-component-residual-audit",
        description=(
            "Audit every official pairwise and complete-track error remaining "
            "after Track2p-policy component cleanup."
        ),
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
    parser.add_argument("--max-gap", type=int, default=TRACK2P_POLICY_DEFAULT_MAX_GAP)
    parser.add_argument("--split-risk-threshold", type=float, default=1.50)
    parser.add_argument("--split-penalty", type=float, default=0.25)
    parser.add_argument("--min-side-observations", type=int, default=2)
    parser.add_argument(
        "--feature-mode",
        choices=("policy-diagnostics", "registered-subset", "none"),
        default="policy-diagnostics",
        help=(
            "Use fast already-computed Track2p-policy link diagnostics by "
            "default. registered-subset recomputes registration features for "
            "all residual pairwise edges and can be slow."
        ),
    )
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
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--format", choices=("csv", "json"), default="csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Track2p-policy component residual audit CLI."""

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
        max_gap=args.max_gap,
        allow_track2p_as_reference_for_smoke_test=args.allow_track2p_as_reference_for_smoke_test,
        include_behavior=args.include_behavior,
        include_non_cells=False,
        cell_probability_threshold=args.cell_probability_threshold,
        exclude_overlapping_pixels=False,
        weighted_masks=False,
        weighted_centroids=False,
    )
    result = run_track2p_policy_component_residual_audit(
        config,
        threshold_method=cast(ThresholdMethod, args.threshold_method),
        iou_distance_threshold=float(args.iou_distance_threshold),
        transform_type=args.transform_type,
        cell_probability_threshold=float(args.cell_probability_threshold),
        max_gap=int(args.max_gap),
        cleanup_config=cleanup_config,
        feature_mode=cast(
            Literal["policy-diagnostics", "registered-subset", "none"],
            args.feature_mode,
        ),
    )
    write_residual_rows(result.error_rows, args.output, output_format=args.format)
    if args.summary_output is not None:
        write_residual_rows(
            result.summary_rows, args.summary_output, output_format=args.format
        )
    return 0


def _no_prune_config() -> Track2pPolicyPruneConfig:
    return Track2pPolicyPruneConfig(
        threshold_margin=0.0,
        competition_margin=0.0,
        min_area_ratio=0.0,
        centroid_distance=float("inf"),
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
