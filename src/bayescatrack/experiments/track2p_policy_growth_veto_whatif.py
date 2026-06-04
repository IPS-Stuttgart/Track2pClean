"""Growth-veto what-if audit over accepted CoherenceSuffixTeacherRescue edges.

The residual growth-field audit found one plausible high-residual false
continuation.  A deployable veto needs a stricter safety check: evaluate the
same label-free growth signal on every accepted edge, not only on residual
errors.  This module starts from CoherenceSuffixTeacherRescue, computes
growth/deformation features for all accepted adjacent edges, and simulates the
global-score effect of removing each edge by splitting its component.

Manual-GT labels and score deltas are audit-only columns.  The growth field,
support flags, component context, and hypothetical split decision are computed
without reference labels.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
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
    _no_prune_config,
)
from bayescatrack.experiments.track2p_policy_growth_field_residual_audit import (
    _as_track_matrix,
    _augment_with_shifted_iou_and_margins,
    _cell_probability,
    _edge_growth_features,
    _growth_models_by_pair,
    _identity_growth_model,
    _pad_track_matrix,
    write_rows,
)
from bayescatrack.experiments.track2p_policy_pruned_benchmark import (
    _roi_indices,
    emulate_track2p_pruned_tracks,
)
from bayescatrack.experiments.track2p_policy_suffix_stitch_ranking_audit import (
    _FeatureCache,
)
from bayescatrack.experiments.track2p_policy_teacher_adjacent_rescue import (
    apply_teacher_adjacent_rescue_edges,
)

METHOD = "track2p-policy-growth-veto-whatif"
_SCORE_KEYS = (
    "pairwise_true_positives",
    "pairwise_false_positives",
    "pairwise_false_negatives",
    "complete_track_true_positives",
    "complete_track_false_positives",
    "complete_track_false_negatives",
)
_RISK_THRESHOLDS = (0.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0)


@dataclass(frozen=True)
class GrowthVetoWhatIfResult:
    """Accepted-edge rows and threshold safety summaries."""

    edge_rows: tuple[dict[str, Any], ...]
    summary_rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _SubjectState:
    subject: str
    sessions: Sequence[Track2pSession]
    policy: np.ndarray
    component_cleanup: np.ndarray
    coherence_suffix: np.ndarray
    teacher: np.ndarray
    combined: np.ndarray
    reference: np.ndarray
    feature_cache: _FeatureCache
    anchor_edges: Mapping[tuple[int, int], Sequence[TrackEdge]]
    growth_models: Mapping[tuple[int, int], Any]
    baseline_scores: Mapping[str, Any]


@dataclass(frozen=True)
class _SplitResult:
    tracks: np.ndarray
    row_index: int
    reason: str
    would_split_component: int
    complete_component_size: int
    is_terminal_edge: int
    is_last_session_edge: int


def run_track2p_policy_growth_veto_whatif(
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
    progress: bool = False,
) -> GrowthVetoWhatIfResult:
    """Return one growth-veto what-if row per accepted adjacent edge."""

    policy_config = track2p_policy_config(
        config,
        transform_type=transform_type,
        cell_probability_threshold=cell_probability_threshold,
    )
    cleanup_config = cleanup_config or ComponentCleanupConfig()
    suffix_gate = suffix_gate or suffix.CoherenceSuffixStitchGate()
    subject_dirs = discover_subject_dirs(policy_config.data)
    if not subject_dirs:
        raise ValueError(f"No Track2p-style subject directories found under {policy_config.data}")

    states: list[_SubjectState] = []
    for index, subject_dir in enumerate(subject_dirs, start=1):
        _log_progress(
            progress,
            f"{METHOD}: subject {index}/{len(subject_dirs)} {subject_dir.name}: build prediction",
        )
        states.append(
            _subject_state(
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
                progress=progress,
            )
        )
        _log_progress(
            progress,
            f"{METHOD}: subject {index}/{len(subject_dirs)} {subject_dir.name}: prediction ready",
        )
    global_baseline_scores = _global_scores(state.baseline_scores for state in states)

    edge_rows: list[dict[str, Any]] = []
    for index, state in enumerate(states, start=1):
        _log_progress(
            progress,
            f"{METHOD}: subject {index}/{len(states)} {state.subject}: score accepted edges",
        )
        edge_rows.extend(
            _accepted_edge_rows(
                state,
                global_baseline_scores=global_baseline_scores,
                threshold_method=threshold_method,
                iou_distance_threshold=float(iou_distance_threshold),
                cell_probability_threshold=float(policy_config.cell_probability_threshold),
                transform_type=policy_config.transform_type,
            )
        )
        _log_progress(
            progress,
            f"{METHOD}: subject {index}/{len(states)} {state.subject}: edge rows done",
        )
    summary_rows = _summary_rows(edge_rows, global_baseline_scores)
    return GrowthVetoWhatIfResult(tuple(edge_rows), tuple(summary_rows))


def _subject_state(
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
    progress: bool,
) -> _SubjectState:
    reference = _load_reference_for_subject(subject_dir, data_root=config.data, config=config)
    _validate_reference_for_benchmark(reference, subject_dir=subject_dir, config=config)
    if reference.source != GROUND_TRUTH_REFERENCE_SOURCE:
        raise ValueError(f"{METHOD} requires independent manual-GT references")
    sessions = _load_subject_sessions(subject_dir, config)
    _validate_reference_roi_indices(reference, sessions)
    reference_tracks = _reference_matrix(reference, curated_only=config.curated_only)
    _log_progress(progress, f"{METHOD}: {subject_dir.name}: sessions/reference loaded")

    policy_prediction = emulate_track2p_pruned_tracks(
        sessions,
        transform_type=config.transform_type,
        threshold_method=threshold_method,
        iou_distance_threshold=float(iou_distance_threshold),
        prune_config=_no_prune_config(),
    )
    policy_full = _normalize_int_track_matrix(policy_prediction.tracks)
    policy_eval, reference_eval, _policy_ids = _evaluated_prediction_rows(policy_full, reference_tracks, config=config)
    reference_eval = _as_track_matrix(reference_eval)
    n_sessions = int(reference_eval.shape[1])
    policy_eval = _pad_track_matrix(_as_track_matrix(policy_eval), width=n_sessions)
    _log_progress(progress, f"{METHOD}: {subject_dir.name}: policy reconstructed")

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
    _log_progress(progress, f"{METHOD}: {subject_dir.name}: component cleanup ready")
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
    _log_progress(
        progress,
        f"{METHOD}: {subject_dir.name}: suffix candidates={len(paths)} selected={len(selected)}",
    )
    stitched = _pad_track_matrix(
        _as_track_matrix(suffix._apply_suffix_paths(cleaned, selected)),
        width=n_sessions,
    )

    teacher_full, _variant = _predict_subject_tracks(subject_dir, replace(config, method="track2p-baseline"))
    teacher_eval, _reference_again, _teacher_ids = _evaluated_prediction_rows(_normalize_int_track_matrix(teacher_full), reference_tracks, config=config)
    teacher_eval = _pad_track_matrix(_as_track_matrix(teacher_eval), width=n_sessions)
    _log_progress(progress, f"{METHOD}: {subject_dir.name}: teacher baseline ready")
    teacher_report = apply_teacher_adjacent_rescue_edges(
        stitched,
        teacher_eval,
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
    _log_progress(
        progress,
        f"{METHOD}: {subject_dir.name}: teacher edits={sum(int(row.get('applied', 0)) for row in teacher_report.rows)}",
    )
    anchor_edges = _anchor_edges_from_policy_diagnostics(
        sessions,
        diagnostics=policy_prediction.diagnostics,
        track2p=teacher_eval,
        component_cleanup=cleaned,
        combined=combined,
        min_registered_iou=float(anchor_min_registered_iou),
        min_cell_probability=float(anchor_min_cell_probability),
    )
    _log_progress(progress, f"{METHOD}: {subject_dir.name}: growth anchors ready")
    return _SubjectState(
        subject=subject_dir.name,
        sessions=sessions,
        policy=policy_eval,
        component_cleanup=cleaned,
        coherence_suffix=stitched,
        teacher=teacher_eval,
        combined=combined,
        reference=reference_eval,
        feature_cache=feature_cache,
        anchor_edges=anchor_edges,
        growth_models=_growth_models_by_pair(sessions, anchor_edges),
        baseline_scores=dict(score_track_matrices(combined, reference_eval)),
    )


def _log_progress(enabled: bool, message: str) -> None:
    if enabled:
        print(message, file=sys.stderr, flush=True)


def _anchor_edges_from_policy_diagnostics(
    sessions: Sequence[Track2pSession],
    *,
    diagnostics: Sequence[Any],
    track2p: np.ndarray,
    component_cleanup: np.ndarray,
    combined: np.ndarray,
    min_registered_iou: float,
    min_cell_probability: float,
) -> dict[tuple[int, int], tuple[TrackEdge, ...]]:
    track2p_edges = set(track_edge_counter(track2p))
    cleanup_or_combined = set(track_edge_counter(component_cleanup)) | set(track_edge_counter(combined))
    roi_indices_by_session = [_roi_indices(session) for session in sessions]
    by_pair: dict[tuple[int, int], list[TrackEdge]] = defaultdict(list)
    for diagnostic in diagnostics:
        session_a = int(diagnostic.session_index)
        session_b = session_a + 1
        if session_a < 0 or session_b >= len(roi_indices_by_session):
            continue
        source_indices = roi_indices_by_session[session_a]
        target_indices = roi_indices_by_session[session_b]
        local_a = int(diagnostic.local_roi_a)
        local_b = int(diagnostic.local_roi_b)
        if local_a >= len(source_indices) or local_b >= len(target_indices):
            continue
        roi_a = int(source_indices[local_a])
        roi_b = int(target_indices[local_b])
        edge = (session_a, session_b, roi_a, roi_b)
        if edge not in track2p_edges or edge not in cleanup_or_combined:
            continue
        if float(diagnostic.assigned_iou) < float(min_registered_iou):
            continue
        cell_a = _cell_probability(sessions, session_a, roi_a)
        cell_b = _cell_probability(sessions, session_b, roi_b)
        if min(cell_a, cell_b) < float(min_cell_probability):
            continue
        by_pair[(session_a, session_b)].append(edge)

    output: dict[tuple[int, int], tuple[TrackEdge, ...]] = {}
    for pair, edges in by_pair.items():
        source_counts = Counter((edge[0], edge[2]) for edge in edges)
        target_counts = Counter((edge[1], edge[3]) for edge in edges)
        output[pair] = tuple(edge for edge in edges if source_counts[(edge[0], edge[2])] == 1 and target_counts[(edge[1], edge[3])] == 1)
    return output


def _accepted_edge_rows(
    state: _SubjectState,
    *,
    global_baseline_scores: Mapping[str, Any],
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
    cell_probability_threshold: float,
    transform_type: str,
) -> list[dict[str, Any]]:
    combined_counts = track_edge_counter(state.combined)
    reference_counts = track_edge_counter(state.reference)
    policy_counts = track_edge_counter(state.policy)
    cleanup_counts = track_edge_counter(state.component_cleanup)
    suffix_counts = track_edge_counter(state.coherence_suffix)
    teacher_counts = track_edge_counter(state.teacher)

    rows: list[dict[str, Any]] = []
    for edge in sorted(combined_counts):
        if edge[1] != edge[0] + 1:
            continue
        model = state.growth_models.get((edge[0], edge[1]), _identity_growth_model())
        for occurrence_index in range(int(combined_counts[edge])):
            growth = _edge_growth_features(
                state.sessions,
                edge,
                model=model,
                anchor_edges=state.anchor_edges.get((edge[0], edge[1]), ()),
                predicted=state.combined,
            )
            split = _remove_edge_occurrence(state.combined, edge, occurrence_index=occurrence_index)
            candidate_scores = dict(score_track_matrices(split.tracks, state.reference))
            delta = _score_delta(state.baseline_scores, candidate_scores)
            global_candidate_scores = _apply_subject_delta(global_baseline_scores, delta)
            edge_source = _edge_source(
                edge,
                occurrence_index=occurrence_index,
                policy_counts=policy_counts,
                cleanup_counts=cleanup_counts,
                suffix_counts=suffix_counts,
            )
            rows.append(
                {
                    "subject": state.subject,
                    "session_a": int(edge[0]),
                    "session_b": int(edge[1]),
                    "roi_a": int(edge[2]),
                    "roi_b": int(edge[3]),
                    "occurrence_index": int(occurrence_index),
                    "edge_source": edge_source,
                    "is_terminal_edge": int(split.is_terminal_edge),
                    "is_last_session_edge": int(split.is_last_session_edge),
                    "track2p_supported": int(teacher_counts.get(edge, 0) > occurrence_index),
                    "policy_supported": int(policy_counts.get(edge, 0) > occurrence_index),
                    "teacher_supported": int(teacher_counts.get(edge, 0) > occurrence_index),
                    "component_cleanup_supported": int(cleanup_counts.get(edge, 0) > occurrence_index),
                    "coherence_suffix_supported": int(suffix_counts.get(edge, 0) > occurrence_index),
                    "growth_residual": growth.growth_residual,
                    "growth_residual_mahalanobis": growth.growth_residual_mahalanobis,
                    "growth_model_type": model.model_type,
                    "growth_anchor_count": int(model.anchor_count),
                    "growth_inlier_count": int(model.inlier_count),
                    "complete_component_size": int(split.complete_component_size),
                    "component_risk": _component_risk(growth),
                    "would_split_component": int(split.would_split_component),
                    "edge_status_against_gt": ("true_positive" if reference_counts.get(edge, 0) > occurrence_index else "false_positive"),
                    "pairwise_tp_delta_if_removed": int(delta["pairwise_true_positives"]),
                    "pairwise_fp_delta_if_removed": int(delta["pairwise_false_positives"]),
                    "pairwise_fn_delta_if_removed": int(delta["pairwise_false_negatives"]),
                    "complete_tp_delta_if_removed": int(delta["complete_track_true_positives"]),
                    "complete_fp_delta_if_removed": int(delta["complete_track_false_positives"]),
                    "complete_fn_delta_if_removed": int(delta["complete_track_false_negatives"]),
                    "new_pairwise_f1_micro": float(global_candidate_scores["pairwise_f1"]),
                    "new_complete_track_f1_micro": float(global_candidate_scores["complete_track_f1"]),
                    "baseline_pairwise_f1_micro": float(global_baseline_scores["pairwise_f1"]),
                    "baseline_complete_track_f1_micro": float(global_baseline_scores["complete_track_f1"]),
                    "remove_reason": split.reason,
                    "threshold_method": str(threshold_method),
                    "iou_distance_threshold": float(iou_distance_threshold),
                    "cell_probability_threshold": float(cell_probability_threshold),
                    "transform_type": str(transform_type),
                }
            )
    return _augment_with_shifted_iou_and_margins(rows, state.sessions, feature_cache=state.feature_cache)


def _remove_edge_occurrence(predicted: np.ndarray, edge: TrackEdge, *, occurrence_index: int) -> _SplitResult:
    predicted = _as_track_matrix(predicted)
    session_a, session_b, roi_a, roi_b = edge
    if session_a >= predicted.shape[1] or session_b >= predicted.shape[1]:
        return _SplitResult(
            tracks=predicted.copy(),
            row_index=-1,
            reason="edge_outside_prediction_width",
            would_split_component=0,
            complete_component_size=0,
            is_terminal_edge=0,
            is_last_session_edge=0,
        )
    rows = [int(row_index) for row_index in np.flatnonzero(predicted[:, session_a] == roi_a) if int(predicted[int(row_index), session_b]) == int(roi_b)]
    if int(occurrence_index) >= len(rows):
        return _SplitResult(
            tracks=predicted.copy(),
            row_index=-1,
            reason="edge_absent_or_ambiguous",
            would_split_component=0,
            complete_component_size=0,
            is_terminal_edge=0,
            is_last_session_edge=int(session_b == predicted.shape[1] - 1),
        )
    row_index = rows[int(occurrence_index)]
    row = np.asarray(predicted[row_index], dtype=int)
    left = row.copy()
    right = row.copy()
    left[session_b:] = -1
    right[:session_b] = -1
    fragments = [fragment for fragment in (left, right) if np.any(fragment >= 0)]
    output = np.delete(predicted, row_index, axis=0)
    if fragments:
        output = np.vstack([output, *fragments])
    return _SplitResult(
        tracks=output,
        row_index=row_index,
        reason="split_edge",
        would_split_component=1,
        complete_component_size=int(np.sum(row >= 0)),
        is_terminal_edge=int(_is_terminal_edge(row, session_a, session_b)),
        is_last_session_edge=int(session_b == predicted.shape[1] - 1),
    )


def _is_terminal_edge(row: np.ndarray, session_a: int, session_b: int) -> bool:
    previous_present = int(session_a) > 0 and int(row[int(session_a) - 1]) >= 0
    next_present = int(session_b) + 1 < row.size and int(row[int(session_b) + 1]) >= 0
    return bool(not previous_present or not next_present)


def _edge_source(
    edge: TrackEdge,
    *,
    occurrence_index: int,
    policy_counts: Counter[TrackEdge],
    cleanup_counts: Counter[TrackEdge],
    suffix_counts: Counter[TrackEdge],
) -> str:
    if suffix_counts.get(edge, 0) <= int(occurrence_index):
        return "teacher"
    if cleanup_counts.get(edge, 0) <= int(occurrence_index):
        return "suffix"
    if policy_counts.get(edge, 0) <= int(occurrence_index):
        return "component"
    return "policy"


def _component_risk(growth: Any) -> float:
    value = float(growth.growth_residual_mahalanobis)
    return value if np.isfinite(value) else float("nan")


def _score_delta(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, int]:
    return {key: int(candidate[key]) - int(baseline[key]) for key in _SCORE_KEYS}


def _global_scores(score_rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    scores = {key: 0 for key in _SCORE_KEYS}
    for row in score_rows:
        for key in _SCORE_KEYS:
            scores[key] += int(row[key])
    scores["pairwise_f1"] = _f1(
        scores["pairwise_true_positives"],
        scores["pairwise_false_positives"],
        scores["pairwise_false_negatives"],
    )
    scores["complete_track_f1"] = _f1(
        scores["complete_track_true_positives"],
        scores["complete_track_false_positives"],
        scores["complete_track_false_negatives"],
    )
    return scores


def _apply_subject_delta(global_baseline_scores: Mapping[str, Any], delta: Mapping[str, int]) -> dict[str, Any]:
    scores = {key: int(global_baseline_scores[key]) + int(delta[key]) for key in _SCORE_KEYS}
    scores["pairwise_f1"] = _f1(
        scores["pairwise_true_positives"],
        scores["pairwise_false_positives"],
        scores["pairwise_false_negatives"],
    )
    scores["complete_track_f1"] = _f1(
        scores["complete_track_true_positives"],
        scores["complete_track_false_positives"],
        scores["complete_track_false_negatives"],
    )
    return scores


def _f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * int(tp) + int(fp) + int(fn)
    if denominator == 0:
        return 1.0
    return float(2 * int(tp) / denominator)


def _summary_rows(
    edge_rows: Sequence[Mapping[str, Any]],
    global_baseline_scores: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    subjects = sorted({str(row["subject"]) for row in edge_rows}) + ["ALL"]
    for subject in subjects:
        subject_rows = [row for row in edge_rows if subject == "ALL" or str(row["subject"]) == subject]
        for threshold in _RISK_THRESHOLDS:
            selected = [
                row for row in subject_rows if np.isfinite(float(row.get("growth_residual_mahalanobis", np.nan))) and float(row["growth_residual_mahalanobis"]) >= float(threshold)
            ]
            rows.append(
                {
                    "subject": subject,
                    "growth_residual_mahalanobis_threshold": float(threshold),
                    "accepted_edges": int(len(subject_rows)),
                    "selected_edges": int(len(selected)),
                    "selected_true_positive_edges": int(sum(str(row.get("edge_status_against_gt")) == "true_positive" for row in selected)),
                    "selected_false_positive_edges": int(sum(str(row.get("edge_status_against_gt")) == "false_positive" for row in selected)),
                    "selected_terminal_edges": int(sum(int(row.get("is_terminal_edge", 0)) for row in selected)),
                    "selected_last_session_edges": int(sum(int(row.get("is_last_session_edge", 0)) for row in selected)),
                    "selected_pairwise_tp_delta_sum": int(sum(int(row.get("pairwise_tp_delta_if_removed", 0)) for row in selected)),
                    "selected_pairwise_fp_delta_sum": int(sum(int(row.get("pairwise_fp_delta_if_removed", 0)) for row in selected)),
                    "selected_pairwise_fn_delta_sum": int(sum(int(row.get("pairwise_fn_delta_if_removed", 0)) for row in selected)),
                    "selected_complete_tp_delta_sum": int(sum(int(row.get("complete_tp_delta_if_removed", 0)) for row in selected)),
                    "selected_complete_fp_delta_sum": int(sum(int(row.get("complete_fp_delta_if_removed", 0)) for row in selected)),
                    "selected_complete_fn_delta_sum": int(sum(int(row.get("complete_fn_delta_if_removed", 0)) for row in selected)),
                    "baseline_pairwise_f1_micro": float(global_baseline_scores["pairwise_f1"]),
                    "baseline_complete_track_f1_micro": float(global_baseline_scores["complete_track_f1"]),
                    "best_single_veto_pairwise_f1_micro": _max_float(selected, "new_pairwise_f1_micro"),
                    "best_single_veto_complete_track_f1_micro": _max_float(selected, "new_complete_track_f1_micro"),
                }
            )
    return tuple(rows)


def _max_float(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
    if not values:
        return float("nan")
    return float(max(values))


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the growth-veto what-if parser."""

    parser = suffix.build_arg_parser()
    parser.prog = "python -m bayescatrack.experiments.track2p_policy_growth_veto_whatif"
    parser.description = "Audit growth-veto one-edge removal what-ifs over every accepted CoherenceSuffixTeacherRescue edge."
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--anchor-min-registered-iou", type=float, default=0.50)
    parser.add_argument("--anchor-min-shifted-iou", type=float, default=0.30)
    parser.add_argument("--anchor-min-cell-probability", type=float, default=0.80)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print per-subject growth-veto audit progress to stderr.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the growth-veto what-if audit."""

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
        allow_track2p_as_reference_for_smoke_test=(args.allow_track2p_as_reference_for_smoke_test),
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
    result = run_track2p_policy_growth_veto_whatif(
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
        progress=bool(args.progress),
    )
    write_rows(
        result.edge_rows,
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
