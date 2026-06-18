"""One-edit Pareto audit for the promoted coherence suffix-stitch row.

This diagnostic starts from the final
``Track2pPolicy + ComponentCleanup + CoherenceSuffixStitch`` prediction and
scores exact single edits against the official duplicate-aware objectives:

* remove one residual pairwise false-positive edge by splitting its component;
* add one residual pairwise false-negative edge when it is structurally clean;
* swap a wrong predicted adjacent edge for a missing GT-adjacent edge when the
  FP and FN are conflict-coupled.

The rows are label-aware what-ifs. They are not a tracker method; they identify
which remaining residuals could improve both pairwise and complete-track scores
without hiding complete-track regressions.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from bayescatrack.core.bridge import Track2pSession
from bayescatrack.evaluation.complete_track_scores import score_track_matrices
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
from bayescatrack.experiments.track2p_policy_audit import TrackEdge, track_edge_counter
from bayescatrack.experiments.track2p_policy_benchmark import (
    TRACK2P_POLICY_DEFAULT_CELL_PROBABILITY_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE,
    track2p_policy_config,
)
from bayescatrack.experiments.track2p_policy_coherence_suffix_stitch_whatif import (
    CoherenceSuffixStitchGate,
    _apply_suffix_paths,
    _FeatureCache,
    _positive_int_arg,
    _positive_int_value,
    _select_paths,
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
    _track2p_baseline_eval,
)
from bayescatrack.experiments.track2p_policy_pruned_benchmark import (
    _margin_against_competitor,
    emulate_track2p_pruned_tracks,
)
from bayescatrack.experiments.track2p_policy_suffix_stitch_ranking_audit import (
    _rank_descending,
    _ranked_suffix_paths,
)
from pyrecest.utils.track_edit_whatif import (
    TrackEdit,
    score_track_edit_delta,
)

TRACK2P_POLICY_COHERENCE_PARETO_WHATIF_METHOD = "track2p-policy-coherence-pareto-whatif"

_COUNT_KEYS = (
    "pairwise_true_positives",
    "pairwise_false_positives",
    "pairwise_false_negatives",
    "complete_track_true_positives",
    "complete_track_false_positives",
    "complete_track_false_negatives",
)
_OUTPUT_FIELDS = (
    "edit_type",
    "subject",
    "session_a",
    "session_b",
    "roi_a",
    "roi_b",
    "support_bucket",
    "track2p_supported",
    "policy_supported",
    "coherence_supported",
    "component_cleanup_required",
    "registered_iou",
    "shifted_iou",
    "centroid_distance",
    "area_ratio",
    "cell_probability_a",
    "cell_probability_b",
    "row_rank",
    "column_rank",
    "row_margin",
    "column_margin",
    "would_break_complete_tp",
    "would_create_complete_fp",
    "pairwise_tp_delta",
    "pairwise_fp_delta",
    "pairwise_fn_delta",
    "complete_tp_delta",
    "complete_fp_delta",
    "complete_fn_delta",
    "new_pairwise_f1_micro",
    "new_complete_track_f1_micro",
    "pareto_improves_track2p",
    "structural_risk",
    "structural_risk_reason",
    "rescue_applied",
    "rescue_action",
    "swap_removed_session_a",
    "swap_removed_session_b",
    "swap_removed_roi_a",
    "swap_removed_roi_b",
)


@dataclass(frozen=True)
class CoherenceParetoWhatIfResult:
    """One-edit rows and compact summary rows."""

    rows: tuple[dict[str, float | int | str], ...]
    summary_rows: tuple[dict[str, float | int | str], ...]


@dataclass(frozen=True)
class _SubjectState:
    subject: str
    sessions: Sequence[Track2pSession]
    predicted: np.ndarray
    cleaned: np.ndarray
    reference: np.ndarray
    track2p: np.ndarray
    policy: np.ndarray
    base_scores: Mapping[str, float | int]
    track2p_scores: Mapping[str, float | int]
    feature_cache: _FeatureCache


@dataclass(frozen=True)
class _Simulation:
    candidate: np.ndarray
    applied: bool
    action: str
    reason: str
    duplicate_source: bool = False
    duplicate_target: bool = False


def run_track2p_policy_coherence_pareto_whatif(
    config: Track2pBenchmarkConfig,
    *,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    iou_distance_threshold: float = TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    transform_type: str | None = None,
    cell_probability_threshold: float | None = None,
    cleanup_config: ComponentCleanupConfig | None = None,
    gate: CoherenceSuffixStitchGate | None = None,
    edge_top_k: int = 25,
    path_beam_width: int = 100,
) -> CoherenceParetoWhatIfResult:
    """Score exact one-edit Pareto candidates after coherence suffix stitching."""

    edge_top_k = _positive_int_value(edge_top_k, name="edge_top_k")
    path_beam_width = _positive_int_value(path_beam_width, name="path_beam_width")
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
    gate = gate or CoherenceSuffixStitchGate()
    states = tuple(
        _subject_state(
            subject_dir,
            config=policy_config,
            cleanup_config=cleanup_config,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
            gate=gate,
            edge_top_k=edge_top_k,
            path_beam_width=path_beam_width,
        )
        for subject_dir in subject_dirs
    )
    global_base = _sum_scores(state.base_scores for state in states)
    global_track2p = _sum_scores(state.track2p_scores for state in states)
    rows: list[dict[str, float | int | str]] = []
    for state in states:
        rows.extend(
            _subject_whatif_rows(
                state,
                global_base=global_base,
                global_track2p=global_track2p,
            )
        )
    rows.sort(key=_pareto_sort_key)
    summary = _summary_rows(
        rows, global_base=global_base, global_track2p=global_track2p
    )
    return CoherenceParetoWhatIfResult(tuple(rows), tuple(summary))


def _subject_state(
    subject_dir: Path,
    *,
    config: Track2pBenchmarkConfig,
    cleanup_config: ComponentCleanupConfig,
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
    gate: CoherenceSuffixStitchGate,
    edge_top_k: int,
    path_beam_width: int,
) -> _SubjectState:
    reference = _load_reference_for_subject(
        subject_dir, data_root=config.data, config=config
    )
    _validate_reference_for_benchmark(reference, subject_dir=subject_dir, config=config)
    if reference.source != GROUND_TRUTH_REFERENCE_SOURCE:
        raise ValueError(
            "Track2p-policy coherence Pareto what-if requires independent manual GT references"
        )
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
    policy_eval, reference_eval, evaluated_track_ids = _evaluated_prediction_rows(
        policy_full, reference_tracks, config=config
    )
    audit_rows = component_audit_rows(
        policy_eval,
        reference_eval,
        sessions=sessions,
        diagnostics=policy_prediction.diagnostics,
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
        max_suffix_length=int(gate.suffix_path_length),
        edge_top_k=int(edge_top_k),
        path_beam_width=int(path_beam_width),
    )
    selected = _select_paths(paths, cleaned_eval, gate=gate)
    stitched_eval = _apply_suffix_paths(cleaned_eval, selected)
    track2p_eval = _track2p_baseline_eval(subject_dir, reference_tracks, config=config)
    return _SubjectState(
        subject=subject_dir.name,
        sessions=sessions,
        predicted=stitched_eval,
        cleaned=cleaned_eval,
        reference=reference_eval,
        track2p=track2p_eval,
        policy=policy_eval,
        base_scores=dict(score_track_matrices(stitched_eval, reference_eval)),
        track2p_scores=dict(score_track_matrices(track2p_eval, reference_eval)),
        feature_cache=feature_cache,
    )


def _subject_whatif_rows(
    state: _SubjectState,
    *,
    global_base: Mapping[str, int],
    global_track2p: Mapping[str, int],
) -> list[dict[str, float | int | str]]:
    predicted_counts = track_edge_counter(state.predicted)
    reference_counts = track_edge_counter(state.reference)
    rows: list[dict[str, float | int | str]] = []

    for edge, count in _residual_edges(
        predicted_counts, reference_counts, error_type="pairwise_fp"
    ):
        for occurrence_index in range(count):
            simulation = _simulate_remove_edge(state.predicted, edge, occurrence_index)
            rows.append(
                _candidate_row(
                    state,
                    edit_type="remove_pairwise_fp",
                    edge=edge,
                    simulation=simulation,
                    global_base=global_base,
                    global_track2p=global_track2p,
                )
            )

    fn_edges: list[TrackEdge] = []
    for edge, count in _residual_edges(
        predicted_counts, reference_counts, error_type="pairwise_fn"
    ):
        fn_edges.extend([edge] * count)
        for occurrence_index in range(count):
            simulation = _simulate_add_edge(state.predicted, edge)
            rows.append(
                _candidate_row(
                    state,
                    edit_type="add_pairwise_fn",
                    edge=edge,
                    simulation=simulation,
                    global_base=global_base,
                    global_track2p=global_track2p,
                    occurrence_index=occurrence_index,
                )
            )

    seen_swaps: set[tuple[TrackEdge, TrackEdge]] = set()
    fp_edge_set = {
        edge
        for edge, _count in _residual_edges(
            predicted_counts, reference_counts, error_type="pairwise_fp"
        )
    }
    for fn_edge in fn_edges:
        for fp_edge in _conflict_coupled_fp_edges(
            fn_edge, state.predicted, fp_edge_set
        ):
            key = (fn_edge, fp_edge)
            if key in seen_swaps:
                continue
            seen_swaps.add(key)
            simulation = _simulate_swap_edge(
                state.predicted, missing_edge=fn_edge, wrong_edge=fp_edge
            )
            rows.append(
                _candidate_row(
                    state,
                    edit_type="swap_conflict_coupled_edge",
                    edge=fn_edge,
                    simulation=simulation,
                    global_base=global_base,
                    global_track2p=global_track2p,
                    swap_removed_edge=fp_edge,
                )
            )
    return rows


def _residual_edges(
    predicted_counts: Mapping[TrackEdge, int],
    reference_counts: Mapping[TrackEdge, int],
    *,
    error_type: Literal["pairwise_fp", "pairwise_fn"],
) -> tuple[tuple[TrackEdge, int], ...]:
    output: list[tuple[TrackEdge, int]] = []
    for edge in sorted(set(predicted_counts) | set(reference_counts)):
        predicted_count = int(predicted_counts.get(edge, 0))
        reference_count = int(reference_counts.get(edge, 0))
        if error_type == "pairwise_fp":
            count = max(0, predicted_count - min(predicted_count, reference_count))
        else:
            count = max(0, reference_count - min(predicted_count, reference_count))
        if count > 0:
            output.append((edge, count))
    return tuple(output)


def _simulate_remove_edge(
    predicted: np.ndarray, edge: TrackEdge, occurrence_index: int = 0
) -> _Simulation:
    output = np.asarray(predicted, dtype=int).copy()
    session_a, session_b, roi_a, roi_b = edge
    matching_rows = tuple(
        int(row_index)
        for row_index in np.flatnonzero(
            (output[:, session_a] == roi_a) & (output[:, session_b] == roi_b)
        )
    )
    if occurrence_index >= len(matching_rows):
        return _Simulation(output, False, "reject", "edge_not_found")

    row_index = matching_rows[int(occurrence_index)]
    left = output[row_index].copy()
    right = output[row_index].copy()
    left[session_b:] = -1
    right[:session_b] = -1
    pieces = [row for index, row in enumerate(output) if index != row_index]
    if np.any(left >= 0):
        pieces.append(left)
    if np.any(right >= 0):
        pieces.append(right)
    candidate = np.vstack(pieces).astype(int, copy=False) if pieces else output[:0]
    return _Simulation(candidate, True, "split_component_at_edge", "accepted")


def _simulate_add_edge(predicted: np.ndarray, edge: TrackEdge) -> _Simulation:
    output = np.asarray(predicted, dtype=int).copy()
    session_a, session_b, roi_a, roi_b = edge
    source_rows = tuple(np.flatnonzero(output[:, session_a] == roi_a))
    target_rows = tuple(np.flatnonzero(output[:, session_b] == roi_b))
    duplicate_source = any(
        output[row_index, session_b] >= 0 and output[row_index, session_b] != roi_b
        for row_index in source_rows
    )
    duplicate_target = any(
        output[row_index, session_a] >= 0 and output[row_index, session_a] != roi_a
        for row_index in target_rows
    )
    if duplicate_source or duplicate_target:
        return _Simulation(
            output,
            False,
            "reject",
            "duplicate_source_or_target",
            duplicate_source=bool(duplicate_source),
            duplicate_target=bool(duplicate_target),
        )
    if len(source_rows) == 1 and len(target_rows) == 0:
        row_index = int(source_rows[0])
        output[row_index, session_b] = int(roi_b)
        return _Simulation(output, True, "insert_target", "accepted")
    if len(source_rows) == 0 and len(target_rows) == 1:
        row_index = int(target_rows[0])
        output[row_index, session_a] = int(roi_a)
        return _Simulation(output, True, "insert_source", "accepted")
    if len(source_rows) == 1 and len(target_rows) == 1:
        source_row = int(source_rows[0])
        target_row = int(target_rows[0])
        if source_row == target_row:
            return _Simulation(output, True, "already_same_component", "accepted")
        merged = _merge_rows_if_compatible(output[source_row], output[target_row])
        if merged is not None:
            keep = [index for index in range(output.shape[0]) if index != target_row]
            output[source_row] = merged
            return _Simulation(
                output[np.asarray(keep, dtype=int)],
                True,
                "merge_components",
                "accepted",
            )
    if len(source_rows) == 0 and len(target_rows) == 0:
        new_row = np.full((1, output.shape[1]), -1, dtype=int)
        new_row[0, session_a] = int(roi_a)
        new_row[0, session_b] = int(roi_b)
        return _Simulation(
            np.vstack([output, new_row]), True, "add_partial_component", "accepted"
        )
    return _Simulation(output, False, "reject", "ambiguous_multiple_components")


def _simulate_swap_edge(
    predicted: np.ndarray, *, missing_edge: TrackEdge, wrong_edge: TrackEdge
) -> _Simulation:
    output = np.asarray(predicted, dtype=int).copy()
    session_a, session_b, roi_a, roi_b = missing_edge
    wrong_a, wrong_b, wrong_roi_a, wrong_roi_b = wrong_edge
    if (session_a, session_b) != (wrong_a, wrong_b):
        return _Simulation(output, False, "reject", "session_pair_mismatch")
    matching_rows = tuple(
        int(row_index)
        for row_index in np.flatnonzero(
            (output[:, session_a] == wrong_roi_a)
            & (output[:, session_b] == wrong_roi_b)
        )
    )
    if not matching_rows:
        return _Simulation(output, False, "reject", "wrong_edge_not_found")
    row_index = matching_rows[0]
    duplicate_source = bool(
        roi_a != wrong_roi_a
        and np.any(
            (output[:, session_a] == roi_a) & (np.arange(output.shape[0]) != row_index)
        )
    )
    duplicate_target = bool(
        roi_b != wrong_roi_b
        and np.any(
            (output[:, session_b] == roi_b) & (np.arange(output.shape[0]) != row_index)
        )
    )
    output[row_index, session_a] = int(roi_a)
    output[row_index, session_b] = int(roi_b)
    return _Simulation(
        output,
        True,
        "swap_adjacent_edge",
        "accepted",
        duplicate_source=duplicate_source,
        duplicate_target=duplicate_target,
    )


def _merge_rows_if_compatible(left: np.ndarray, right: np.ndarray) -> np.ndarray | None:
    conflict = (left >= 0) & (right >= 0) & (left != right)
    if np.any(conflict):
        return None
    return np.where(left >= 0, left, right)


def _conflict_coupled_fp_edges(
    fn_edge: TrackEdge, predicted: np.ndarray, fp_edges: set[TrackEdge]
) -> tuple[TrackEdge, ...]:
    session_a, session_b, roi_a, roi_b = fn_edge
    output: list[TrackEdge] = []
    for row in predicted:
        if session_a >= row.size or session_b >= row.size:
            continue
        pred_a = int(row[session_a])
        pred_b = int(row[session_b])
        if pred_a < 0 or pred_b < 0:
            continue
        edge = (session_a, session_b, pred_a, pred_b)
        if edge not in fp_edges:
            continue
        if pred_a == roi_a and pred_b != roi_b:
            output.append(edge)
        elif pred_a != roi_a and pred_b == roi_b:
            output.append(edge)
    return tuple(dict.fromkeys(output))


def _candidate_row(
    state: _SubjectState,
    *,
    edit_type: str,
    edge: TrackEdge,
    simulation: _Simulation,
    global_base: Mapping[str, int],
    global_track2p: Mapping[str, int],
    occurrence_index: int = 0,
    swap_removed_edge: TrackEdge | None = None,
) -> dict[str, float | int | str]:
    edit = _track_edit_for_candidate(
        edit_type=edit_type,
        edge=edge,
        occurrence_index=occurrence_index,
        swap_removed_edge=swap_removed_edge,
    )
    edit_delta = score_track_edit_delta(
        state.predicted,
        state.reference,
        edit,
        count_duplicates=True,
    )
    deltas = _edit_delta_counts(edit_delta)
    new_global = _apply_deltas(global_base, deltas)
    feature = _edge_feature_row(state, edge)
    would_break_complete_tp = bool(edit_delta.breaks_complete_track)
    would_create_complete_fp = bool(edit_delta.complete_fp_delta > 0)
    support = _support_flags(state, edge)
    structural_risk, risk_reason = _structural_risk(
        simulation,
        would_break_complete_tp=would_break_complete_tp,
        would_create_complete_fp=would_create_complete_fp,
    )
    return {
        "edit_type": edit_type,
        "subject": state.subject,
        "session_a": int(edge[0]),
        "session_b": int(edge[1]),
        "roi_a": int(edge[2]),
        "roi_b": int(edge[3]),
        "occurrence_index": int(occurrence_index),
        "support_bucket": _support_bucket(support),
        "track2p_supported": int(support["track2p_supported"]),
        "policy_supported": int(support["policy_supported"]),
        "coherence_supported": int(support["coherence_supported"]),
        "component_cleanup_required": int(support["component_cleanup_required"]),
        **feature,
        "would_break_complete_tp": int(would_break_complete_tp),
        "would_create_complete_fp": int(would_create_complete_fp),
        "pairwise_tp_delta": int(deltas["pairwise_true_positives"]),
        "pairwise_fp_delta": int(deltas["pairwise_false_positives"]),
        "pairwise_fn_delta": int(deltas["pairwise_false_negatives"]),
        "complete_tp_delta": int(deltas["complete_track_true_positives"]),
        "complete_fp_delta": int(deltas["complete_track_false_positives"]),
        "complete_fn_delta": int(deltas["complete_track_false_negatives"]),
        "new_pairwise_f1_micro": _f1(
            new_global["pairwise_true_positives"],
            new_global["pairwise_false_positives"],
            new_global["pairwise_false_negatives"],
        ),
        "new_complete_track_f1_micro": _f1(
            new_global["complete_track_true_positives"],
            new_global["complete_track_false_positives"],
            new_global["complete_track_false_negatives"],
        ),
        "pareto_improves_track2p": int(
            _pareto_improves_track2p(new_global, global_track2p)
        ),
        "structural_risk": int(structural_risk),
        "structural_risk_reason": risk_reason,
        "rescue_applied": int(simulation.applied),
        "rescue_action": simulation.action,
        "rescue_reject_reason": simulation.reason if not simulation.applied else "",
        "swap_removed_session_a": (
            int(swap_removed_edge[0]) if swap_removed_edge else -1
        ),
        "swap_removed_session_b": (
            int(swap_removed_edge[1]) if swap_removed_edge else -1
        ),
        "swap_removed_roi_a": int(swap_removed_edge[2]) if swap_removed_edge else -1,
        "swap_removed_roi_b": int(swap_removed_edge[3]) if swap_removed_edge else -1,
    }


def _track_edit_for_candidate(
    *,
    edit_type: str,
    edge: TrackEdge,
    occurrence_index: int = 0,
    swap_removed_edge: TrackEdge | None = None,
) -> TrackEdit:
    session_a, session_b, roi_a, roi_b = edge
    if edit_type == "remove_pairwise_fp":
        return TrackEdit(
            kind="remove_link",
            session_a=session_a,
            session_b=session_b,
            source_observation=roi_a,
            target_observation=roi_b,
            metadata={"occurrence_index": int(occurrence_index)},
        )
    if edit_type == "add_pairwise_fn":
        return TrackEdit(
            kind="add_link",
            session_a=session_a,
            session_b=session_b,
            source_observation=roi_a,
            target_observation=roi_b,
        )
    if edit_type == "swap_conflict_coupled_edge" and swap_removed_edge is not None:
        wrong_session_a, wrong_session_b, wrong_roi_a, wrong_roi_b = swap_removed_edge
        return TrackEdit(
            kind="swap_link",
            session_a=session_a,
            session_b=session_b,
            source_observation=roi_a,
            target_observation=roi_b,
            metadata={
                "remove_session_a": wrong_session_a,
                "remove_session_b": wrong_session_b,
                "remove_source_observation": wrong_roi_a,
                "remove_target_observation": wrong_roi_b,
            },
        )
    raise ValueError(f"Unsupported coherence Pareto edit type: {edit_type!r}")


def _edit_delta_counts(edit_delta: Any) -> dict[str, int]:
    return {
        "pairwise_true_positives": int(edit_delta.pairwise_tp_delta),
        "pairwise_false_positives": int(edit_delta.pairwise_fp_delta),
        "pairwise_false_negatives": int(edit_delta.pairwise_fn_delta),
        "complete_track_true_positives": int(edit_delta.complete_tp_delta),
        "complete_track_false_positives": int(edit_delta.complete_fp_delta),
        "complete_track_false_negatives": int(edit_delta.complete_fn_delta),
    }


def _edge_feature_row(state: _SubjectState, edge: TrackEdge) -> dict[str, float | int]:
    session_a, session_b, roi_a, roi_b = edge
    base = {
        "registered_iou": float("nan"),
        "shifted_iou": float("nan"),
        "centroid_distance": float("nan"),
        "area_ratio": float("nan"),
        "cell_probability_a": _cell_probability(state.sessions, session_a, roi_a),
        "cell_probability_b": _cell_probability(state.sessions, session_b, roi_b),
        "row_rank": -1,
        "column_rank": -1,
        "row_margin": float("nan"),
        "column_margin": float("nan"),
    }
    if session_b != session_a + 1:
        return base
    pair = state.feature_cache.pair(session_a)
    source_lookup = {int(roi): index for index, roi in enumerate(pair.source_indices)}
    target_lookup = {int(roi): index for index, roi in enumerate(pair.target_indices)}
    source_index = source_lookup.get(int(roi_a))
    target_index = target_lookup.get(int(roi_b))
    if source_index is None or target_index is None:
        return base
    return {
        "registered_iou": float(pair.registered_iou[source_index, target_index]),
        "shifted_iou": float(pair.shifted_iou[source_index, target_index]),
        "centroid_distance": float(pair.centroid_distance[source_index, target_index]),
        "area_ratio": float(pair.area_ratio[source_index, target_index]),
        "cell_probability_a": base["cell_probability_a"],
        "cell_probability_b": base["cell_probability_b"],
        "row_rank": int(
            _rank_descending(
                pair.registered_iou[source_index, :], selected_index=target_index
            )
        ),
        "column_rank": int(
            _rank_descending(
                pair.registered_iou[:, target_index], selected_index=source_index
            )
        ),
        "row_margin": float(
            _margin_against_competitor(
                pair.registered_iou[source_index, :], selected_index=target_index
            )
        ),
        "column_margin": float(
            _margin_against_competitor(
                pair.registered_iou[:, target_index], selected_index=source_index
            )
        ),
    }


def _support_flags(state: _SubjectState, edge: TrackEdge) -> dict[str, bool]:
    track2p_count = track_edge_counter(state.track2p).get(edge, 0)
    policy_count = track_edge_counter(state.policy).get(edge, 0)
    cleaned_count = track_edge_counter(state.cleaned).get(edge, 0)
    coherence_count = track_edge_counter(state.predicted).get(edge, 0)
    return {
        "track2p_supported": bool(track2p_count > 0),
        "policy_supported": bool(policy_count > 0),
        "coherence_supported": bool(coherence_count > 0),
        "component_cleanup_required": bool(policy_count != cleaned_count),
    }


def _support_bucket(flags: Mapping[str, bool]) -> str:
    enabled = [
        name.removesuffix("_supported")
        for name in ("track2p_supported", "policy_supported", "coherence_supported")
        if flags.get(name, False)
    ]
    if flags.get("component_cleanup_required", False):
        enabled.append("component-cleanup")
    return "+".join(enabled) if enabled else "unsupported"


def _structural_risk(
    simulation: _Simulation,
    *,
    would_break_complete_tp: bool,
    would_create_complete_fp: bool,
) -> tuple[int, str]:
    reasons: list[str] = []
    risk = 0
    if not simulation.applied:
        risk += 3
        reasons.append(f"not-applied:{simulation.reason}")
    if simulation.duplicate_source:
        risk += 2
        reasons.append("duplicate-source")
    if simulation.duplicate_target:
        risk += 2
        reasons.append("duplicate-target")
    if would_break_complete_tp:
        risk += 3
        reasons.append("breaks-complete-tp")
    if would_create_complete_fp:
        risk += 1
        reasons.append("creates-complete-fp")
    return risk, ";".join(reasons) if reasons else "none"


def _sum_scores(scores: Sequence[Mapping[str, float | int]]) -> dict[str, int]:
    score_rows = tuple(scores)
    return {
        key: int(sum(int(score[key]) for score in score_rows)) for key in _COUNT_KEYS
    }


def _apply_deltas(
    baseline: Mapping[str, int], deltas: Mapping[str, int]
) -> dict[str, int]:
    return {key: max(0, int(baseline[key]) + int(deltas[key])) for key in _COUNT_KEYS}


def _f1(tp: Any, fp: Any, fn: Any) -> float:
    denominator = 2 * int(tp) + int(fp) + int(fn)
    if denominator == 0:
        return 1.0
    return float(2 * int(tp) / denominator)


def _pareto_improves_track2p(
    candidate_counts: Mapping[str, int], track2p_counts: Mapping[str, int]
) -> bool:
    candidate_pairwise = _f1(
        candidate_counts["pairwise_true_positives"],
        candidate_counts["pairwise_false_positives"],
        candidate_counts["pairwise_false_negatives"],
    )
    candidate_complete = _f1(
        candidate_counts["complete_track_true_positives"],
        candidate_counts["complete_track_false_positives"],
        candidate_counts["complete_track_false_negatives"],
    )
    track2p_pairwise = _f1(
        track2p_counts["pairwise_true_positives"],
        track2p_counts["pairwise_false_positives"],
        track2p_counts["pairwise_false_negatives"],
    )
    track2p_complete = _f1(
        track2p_counts["complete_track_true_positives"],
        track2p_counts["complete_track_false_positives"],
        track2p_counts["complete_track_false_negatives"],
    )
    return bool(
        candidate_pairwise >= track2p_pairwise
        and candidate_complete >= track2p_complete
        and (
            candidate_pairwise > track2p_pairwise
            or candidate_complete > track2p_complete
        )
    )


def _pareto_sort_key(
    row: Mapping[str, float | int | str],
) -> tuple[float, float, int, str, str]:
    return (
        -float(row["new_pairwise_f1_micro"]),
        -float(row["new_complete_track_f1_micro"]),
        int(row.get("structural_risk", 0)),
        str(row.get("edit_type", "")),
        str(row.get("subject", "")),
    )


def _summary_rows(
    rows: Sequence[Mapping[str, float | int | str]],
    *,
    global_base: Mapping[str, int],
    global_track2p: Mapping[str, int],
) -> tuple[dict[str, float | int | str], ...]:
    baseline_pairwise = _f1(
        global_base["pairwise_true_positives"],
        global_base["pairwise_false_positives"],
        global_base["pairwise_false_negatives"],
    )
    baseline_complete = _f1(
        global_base["complete_track_true_positives"],
        global_base["complete_track_false_positives"],
        global_base["complete_track_false_negatives"],
    )
    track2p_pairwise = _f1(
        global_track2p["pairwise_true_positives"],
        global_track2p["pairwise_false_positives"],
        global_track2p["pairwise_false_negatives"],
    )
    track2p_complete = _f1(
        global_track2p["complete_track_true_positives"],
        global_track2p["complete_track_false_positives"],
        global_track2p["complete_track_false_negatives"],
    )
    output: list[dict[str, float | int | str]] = [
        {
            "group": "ALL",
            "candidate_count": int(len(rows)),
            "baseline_pairwise_f1_micro": baseline_pairwise,
            "baseline_complete_track_f1_micro": baseline_complete,
            "track2p_pairwise_f1_micro": track2p_pairwise,
            "track2p_complete_track_f1_micro": track2p_complete,
            "best_new_pairwise_f1_micro": max(
                (float(row["new_pairwise_f1_micro"]) for row in rows),
                default=baseline_pairwise,
            ),
            "best_new_complete_track_f1_micro": max(
                (float(row["new_complete_track_f1_micro"]) for row in rows),
                default=baseline_complete,
            ),
            "pareto_improves_track2p_count": int(
                sum(int(row.get("pareto_improves_track2p", 0)) for row in rows)
            ),
        }
    ]
    grouped: dict[tuple[str, str], list[Mapping[str, float | int | str]]] = defaultdict(
        list
    )
    for row in rows:
        grouped[(str(row["edit_type"]), str(row["support_bucket"]))].append(row)
    for (edit_type, support_bucket), group_rows in sorted(grouped.items()):
        best = min(group_rows, key=_pareto_sort_key)
        output.append(
            {
                "group": f"{edit_type}|{support_bucket}",
                "edit_type": edit_type,
                "support_bucket": support_bucket,
                "candidate_count": int(len(group_rows)),
                "applied_count": int(
                    sum(int(row.get("rescue_applied", 0)) for row in group_rows)
                ),
                "structural_risk_min": int(
                    min(int(row.get("structural_risk", 0)) for row in group_rows)
                ),
                "best_subject": str(best["subject"]),
                "best_edge": f"{best['session_a']}:{best['roi_a']}->{best['session_b']}:{best['roi_b']}",
                "best_pairwise_tp_delta": int(best["pairwise_tp_delta"]),
                "best_pairwise_fp_delta": int(best["pairwise_fp_delta"]),
                "best_pairwise_fn_delta": int(best["pairwise_fn_delta"]),
                "best_complete_tp_delta": int(best["complete_tp_delta"]),
                "best_complete_fp_delta": int(best["complete_fp_delta"]),
                "best_complete_fn_delta": int(best["complete_fn_delta"]),
                "best_new_pairwise_f1_micro": float(best["new_pairwise_f1_micro"]),
                "best_new_complete_track_f1_micro": float(
                    best["new_complete_track_f1_micro"]
                ),
                "pareto_improves_track2p_count": int(
                    sum(
                        int(row.get("pareto_improves_track2p", 0)) for row in group_rows
                    )
                ),
                "baseline_pairwise_f1_micro": baseline_pairwise,
                "baseline_complete_track_f1_micro": baseline_complete,
                "track2p_pairwise_f1_micro": track2p_pairwise,
                "track2p_complete_track_f1_micro": track2p_complete,
            }
        )
    return tuple(output)


def write_rows(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    output_format: Literal["csv", "json"] = "csv",
    preferred_fields: Sequence[str] = (),
) -> None:
    """Write rows while preserving the paper-facing audit column order."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output_path.write_text(
            json.dumps(list(rows), indent=2) + "\n", encoding="utf-8"
        )
        return
    remaining = sorted({key for row in rows for key in row} - set(preferred_fields))
    fieldnames = [
        field for field in preferred_fields if any(field in row for row in rows)
    ]
    fieldnames.extend(remaining)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the coherence Pareto one-edit audit."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-policy-coherence-pareto-whatif",
        description="Run exact one-edit Pareto what-ifs after CoherenceSuffixStitch.",
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
    parser.add_argument("--suffix-path-length", type=_positive_int_arg, default=2)
    parser.add_argument("--min-cell-probability", type=float, default=0.80)
    parser.add_argument("--min-area-ratio", type=float, default=0.80)
    parser.add_argument("--max-centroid-distance", type=float, default=6.0)
    parser.add_argument("--min-shifted-iou", type=float, default=0.30)
    parser.add_argument("--min-motion-consistency", type=float, default=0.50)
    parser.add_argument("--min-shape-consistency", type=float, default=0.82)
    parser.add_argument("--max-stitches-per-subject", type=_positive_int_arg, default=1)
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
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--format", choices=("csv", "json"), default="csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the coherence Pareto one-edit audit CLI."""

    args = build_arg_parser().parse_args(argv)
    cleanup_config = ComponentCleanupConfig(
        split_risk_threshold=args.split_risk_threshold,
        split_penalty=args.split_penalty,
        min_side_observations=args.min_side_observations,
        require_complete_track=args.require_complete_track,
    )
    gate = CoherenceSuffixStitchGate(
        suffix_path_length=int(args.suffix_path_length),
        min_cell_probability=float(args.min_cell_probability),
        min_area_ratio=float(args.min_area_ratio),
        max_centroid_distance=float(args.max_centroid_distance),
        min_shifted_iou=float(args.min_shifted_iou),
        min_motion_consistency=float(args.min_motion_consistency),
        min_shape_consistency=float(args.min_shape_consistency),
        max_stitches_per_subject=int(args.max_stitches_per_subject),
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
    result = run_track2p_policy_coherence_pareto_whatif(
        config,
        threshold_method=cast(ThresholdMethod, args.threshold_method),
        iou_distance_threshold=float(args.iou_distance_threshold),
        transform_type=args.transform_type,
        cell_probability_threshold=float(args.cell_probability_threshold),
        cleanup_config=cleanup_config,
        gate=gate,
        edge_top_k=int(args.edge_top_k),
        path_beam_width=int(args.path_beam_width),
    )
    write_rows(
        result.rows,
        args.output,
        output_format=args.format,
        preferred_fields=_OUTPUT_FIELDS,
    )
    if args.summary_output is not None:
        write_rows(result.summary_rows, args.summary_output, output_format=args.format)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
