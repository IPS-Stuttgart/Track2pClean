"""Audit Track2p-supported false negatives after component cleanup.

The component-cleanup residual audit showed that gap-rescue support is
concentrated in false positives.  This diagnostic instead looks at the only
plausible pairwise-F1 improvement path left: manual-GT adjacent links that are
missing after Track2pPolicy component cleanup but are present in Track2p's own
output.  Each row simulates adding one such adjacent edge and reports conflict
flags plus official pairwise/complete-track score deltas.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
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
from bayescatrack.experiments.track2p_emulation_benchmark import (
    ThresholdMethod,
    emulate_track2p_tracks,
)
from bayescatrack.experiments.track2p_policy_audit import TrackEdge, track_edge_counter
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
from bayescatrack.experiments.track2p_policy_component_residual_audit import (
    ResidualFeature,
    _cell_probability,
    _feature_subset_for_edges,
    _no_prune_config,
    _track2p_baseline_eval,
)
from bayescatrack.experiments.track2p_policy_pruned_benchmark import (
    emulate_track2p_pruned_tracks,
)

TRACK2P_POLICY_TEACHER_FN_AUDIT_METHOD = "track2p-policy-teacher-fn-audit"


@dataclass(frozen=True)
class TeacherFnAuditResult:
    """Teacher-supported FN audit rows plus compact summary rows."""

    edge_rows: tuple[dict[str, float | int | str], ...]
    summary_rows: tuple[dict[str, float | int | str], ...]


@dataclass(frozen=True)
class _RescueSimulation:
    candidate: np.ndarray
    applied: bool
    action: str
    reason: str
    would_create_duplicate_source: bool
    would_create_duplicate_target: bool
    would_merge_components: bool
    would_break_complete_tp: bool


def run_track2p_policy_teacher_fn_audit(
    config: Track2pBenchmarkConfig,
    *,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    iou_distance_threshold: float = TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    transform_type: str | None = None,
    cell_probability_threshold: float | None = None,
    max_gap: int = TRACK2P_POLICY_DEFAULT_MAX_GAP,
    cleanup_config: ComponentCleanupConfig | None = None,
    feature_mode: Literal["none", "registered-subset"] = "none",
) -> TeacherFnAuditResult:
    """Return Track2p-supported residual pairwise FNs after component cleanup."""

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
                "Track2p-policy teacher-FN audit requires independent manual GT "
                "references"
            )
        sessions = _load_subject_sessions(subject_dir, policy_config)
        _validate_reference_roi_indices(reference, sessions)
        reference_tracks = _reference_matrix(
            reference, curated_only=policy_config.curated_only
        )
        predicted_eval, reference_eval, policy_eval = _component_cleanup_eval(
            sessions,
            reference_tracks,
            config=policy_config,
            cleanup_config=cleanup_config,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
        )
        track2p_eval = _track2p_baseline_eval(
            subject_dir, reference_tracks, config=policy_config
        )
        gap_eval, _, _ = _evaluated_prediction_rows(
            _normalize_int_track_matrix(
                emulate_track2p_tracks(
                    sessions,
                    transform_type=policy_config.transform_type,
                    threshold_method=threshold_method,
                    iou_distance_threshold=float(iou_distance_threshold),
                    max_gap=int(policy_config.max_gap),
                )
            ),
            reference_tracks,
            config=policy_config,
        )
        candidate_edges = _track2p_supported_fn_edges(
            predicted_eval,
            reference_eval,
            track2p_eval,
            policy_eval,
        )
        feature_index = (
            _feature_subset_for_edges(
                sessions,
                set(candidate_edges),
                transform_type=policy_config.transform_type,
                threshold_method=threshold_method,
                iou_distance_threshold=float(iou_distance_threshold),
            )
            if feature_mode == "registered-subset"
            else {}
        )
        subject_rows = _teacher_fn_rows(
            subject=subject_dir.name,
            sessions=sessions,
            predicted=predicted_eval,
            reference=reference_eval,
            track2p=track2p_eval,
            policy=policy_eval,
            gap=gap_eval,
            candidate_edges=candidate_edges,
            feature_index=feature_index,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
            transform_type=policy_config.transform_type,
            cell_probability_threshold=float(policy_config.cell_probability_threshold),
        )
        all_rows.extend(subject_rows)
        summary_rows.append(_summary_row(subject_dir.name, subject_rows))

    summary_rows.append(_summary_row("ALL", all_rows))
    return TeacherFnAuditResult(tuple(all_rows), tuple(summary_rows))


def _component_cleanup_eval(
    sessions: Sequence[Track2pSession],
    reference_tracks: np.ndarray,
    *,
    config: Track2pBenchmarkConfig,
    cleanup_config: ComponentCleanupConfig,
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        diagnostics=prediction.diagnostics,
        subject="",
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
    return cleaned_eval, reference_eval, policy_eval


def _track2p_supported_fn_edges(
    predicted: np.ndarray,
    reference: np.ndarray,
    track2p: np.ndarray,
    policy: np.ndarray,
) -> tuple[TrackEdge, ...]:
    predicted_counts = track_edge_counter(predicted)
    reference_counts = track_edge_counter(reference)
    track2p_counts = track_edge_counter(track2p)
    policy_counts = track_edge_counter(policy)
    edges: list[TrackEdge] = []
    for edge in sorted(reference_counts):
        missing = int(reference_counts[edge] - predicted_counts.get(edge, 0))
        if missing <= 0:
            continue
        if track2p_counts.get(edge, 0) <= 0:
            continue
        if policy_counts.get(edge, 0) > 0:
            continue
        edges.extend([edge] * missing)
    return tuple(edges)


def _teacher_fn_rows(
    *,
    subject: str,
    sessions: Sequence[Track2pSession],
    predicted: np.ndarray,
    reference: np.ndarray,
    track2p: np.ndarray,
    policy: np.ndarray,
    gap: np.ndarray,
    candidate_edges: Sequence[TrackEdge],
    feature_index: Mapping[TrackEdge, ResidualFeature],
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
    transform_type: str,
    cell_probability_threshold: float,
) -> list[dict[str, float | int | str]]:
    baseline_scores = score_track_matrices(predicted, reference)
    reference_complete = _complete_track_counter(reference)
    rows: list[dict[str, float | int | str]] = []
    for occurrence_index, edge in enumerate(candidate_edges):
        simulation = _simulate_adjacent_rescue(
            predicted,
            edge,
            reference_complete=reference_complete,
        )
        candidate_scores = score_track_matrices(simulation.candidate, reference)
        session_a, session_b, roi_a, roi_b = edge
        feature = feature_index.get(edge, ResidualFeature())
        rows.append(
            {
                "subject": subject,
                "session_a": int(session_a),
                "session_b": int(session_b),
                "session_a_name": _session_name(sessions, session_a),
                "session_b_name": _session_name(sessions, session_b),
                "roi_a": int(roi_a),
                "roi_b": int(roi_b),
                "occurrence_index": int(occurrence_index),
                "track2p_supported": int(track_edge_counter(track2p).get(edge, 0) > 0),
                "policy_supported": int(track_edge_counter(policy).get(edge, 0) > 0),
                "gap_rescue_supported": int(track_edge_counter(gap).get(edge, 0) > 0),
                "component_cleanup_supported": int(
                    track_edge_counter(predicted).get(edge, 0) > 0
                ),
                "registered_iou": float(feature.registered_iou),
                "assigned_iou": float(feature.registered_iou),
                "centroid_distance": float(feature.centroid_distance),
                "area_ratio": float(feature.area_ratio),
                "cell_probability_a": _cell_probability(sessions, session_a, roi_a),
                "cell_probability_b": _cell_probability(sessions, session_b, roi_b),
                "row_rank": int(feature.row_rank),
                "column_rank": int(feature.column_rank),
                "row_margin": float(feature.row_margin),
                "column_margin": float(feature.column_margin),
                "threshold": float(feature.threshold),
                "threshold_margin": float(feature.threshold_margin),
                "would_create_duplicate_source": int(
                    simulation.would_create_duplicate_source
                ),
                "would_create_duplicate_target": int(
                    simulation.would_create_duplicate_target
                ),
                "would_merge_components": int(simulation.would_merge_components),
                "would_break_complete_tp": int(simulation.would_break_complete_tp),
                "rescue_applied": int(simulation.applied),
                "rescue_action": simulation.action,
                "rescue_reject_reason": simulation.reason,
                **_score_delta_columns(
                    baseline_scores, candidate_scores, prefix="what_if_pairwise"
                ),
                **_score_delta_columns(
                    baseline_scores,
                    candidate_scores,
                    prefix="what_if_complete_track",
                    score_prefix="complete_track",
                ),
                "threshold_method": str(threshold_method),
                "iou_distance_threshold": float(iou_distance_threshold),
                "cell_probability_threshold": float(cell_probability_threshold),
                "transform_type": str(transform_type),
            }
        )
    return rows


def _simulate_adjacent_rescue(
    predicted: np.ndarray,
    edge: TrackEdge,
    *,
    reference_complete: Counter[tuple[int, ...]],
) -> _RescueSimulation:
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
    would_break_complete_tp = any(
        _complete_track_status(output[row_index], reference_complete)
        == "true_positive"
        for row_index in set(source_rows) | set(target_rows)
    )
    if duplicate_source or duplicate_target:
        return _RescueSimulation(
            candidate=output,
            applied=False,
            action="reject",
            reason="duplicate_source_or_target",
            would_create_duplicate_source=duplicate_source,
            would_create_duplicate_target=duplicate_target,
            would_merge_components=False,
            would_break_complete_tp=would_break_complete_tp,
        )

    if len(source_rows) == 1 and len(target_rows) == 0:
        row_index = int(source_rows[0])
        if output[row_index, session_b] < 0:
            output[row_index, session_b] = roi_b
            return _accepted(output, "insert_target", would_break_complete_tp)
    if len(source_rows) == 0 and len(target_rows) == 1:
        row_index = int(target_rows[0])
        if output[row_index, session_a] < 0:
            output[row_index, session_a] = roi_a
            return _accepted(output, "insert_source", would_break_complete_tp)
    if len(source_rows) == 1 and len(target_rows) == 1:
        source_row = int(source_rows[0])
        target_row = int(target_rows[0])
        if source_row == target_row:
            return _accepted(output, "already_same_component", would_break_complete_tp)
        merged = _merge_rows_if_compatible(output[source_row], output[target_row])
        if merged is not None:
            keep = [idx for idx in range(output.shape[0]) if idx != target_row]
            output[source_row] = merged
            output = output[np.asarray(keep, dtype=int)]
            return _RescueSimulation(
                candidate=output,
                applied=True,
                action="merge_components",
                reason="accepted",
                would_create_duplicate_source=False,
                would_create_duplicate_target=False,
                would_merge_components=True,
                would_break_complete_tp=would_break_complete_tp,
            )

    if len(source_rows) == 0 and len(target_rows) == 0:
        new_row = np.full((1, output.shape[1]), -1, dtype=int)
        new_row[0, session_a] = roi_a
        new_row[0, session_b] = roi_b
        output = np.vstack([output, new_row])
        return _accepted(output, "add_partial_component", would_break_complete_tp)

    return _RescueSimulation(
        candidate=output,
        applied=False,
        action="reject",
        reason="ambiguous_multiple_components",
        would_create_duplicate_source=False,
        would_create_duplicate_target=False,
        would_merge_components=False,
        would_break_complete_tp=would_break_complete_tp,
    )


def _accepted(
    output: np.ndarray, action: str, would_break_complete_tp: bool
) -> _RescueSimulation:
    return _RescueSimulation(
        candidate=output,
        applied=True,
        action=action,
        reason="accepted",
        would_create_duplicate_source=False,
        would_create_duplicate_target=False,
        would_merge_components=False,
        would_break_complete_tp=would_break_complete_tp,
    )


def _merge_rows_if_compatible(left: np.ndarray, right: np.ndarray) -> np.ndarray | None:
    conflict = (left >= 0) & (right >= 0) & (left != right)
    if np.any(conflict):
        return None
    return np.where(left >= 0, left, right)


def _score_delta_columns(
    baseline: Mapping[str, float | int],
    candidate: Mapping[str, float | int],
    *,
    prefix: str,
    score_prefix: str = "pairwise",
) -> dict[str, float | int | str]:
    tp_delta = int(candidate[f"{score_prefix}_true_positives"]) - int(
        baseline[f"{score_prefix}_true_positives"]
    )
    fp_delta = int(candidate[f"{score_prefix}_false_positives"]) - int(
        baseline[f"{score_prefix}_false_positives"]
    )
    fn_delta = int(candidate[f"{score_prefix}_false_negatives"]) - int(
        baseline[f"{score_prefix}_false_negatives"]
    )
    f1_delta = float(candidate[f"{score_prefix}_f1"]) - float(
        baseline[f"{score_prefix}_f1"]
    )
    return {
        f"{prefix}_tp_delta": tp_delta,
        f"{prefix}_fp_delta": fp_delta,
        f"{prefix}_fn_delta": fn_delta,
        f"{prefix}_f1_delta": f1_delta,
        f"{prefix}_delta": (
            f"TP {tp_delta:+d}, FP {fp_delta:+d}, FN {fn_delta:+d}, "
            f"F1 {f1_delta:+.6f}"
        ),
    }


def _complete_track_counter(track_matrix: np.ndarray) -> Counter[tuple[int, ...]]:
    counter: Counter[tuple[int, ...]] = Counter()
    for row in _normalize_int_track_matrix(track_matrix):
        if np.all(row >= 0):
            counter[tuple(int(value) for value in row)] += 1
    return counter


def _session_name(sessions: Sequence[Track2pSession], session_index: int) -> str:
    if 0 <= session_index < len(sessions):
        return str(sessions[session_index].session_name)
    return str(session_index)


def _summary_row(
    subject: str, rows: Sequence[Mapping[str, float | int | str]]
) -> dict[str, float | int | str]:
    clean = [
        row
        for row in rows
        if int(row.get("rescue_applied", 0))
        and int(row.get("would_create_duplicate_source", 0)) == 0
        and int(row.get("would_create_duplicate_target", 0)) == 0
        and int(row.get("would_break_complete_tp", 0)) == 0
        and int(row.get("what_if_pairwise_tp_delta", 0)) > 0
        and int(row.get("what_if_pairwise_fp_delta", 0)) == 0
    ]
    return {
        "subject": subject,
        "track2p_supported_pairwise_fns": int(len(rows)),
        "conflict_free_metric_positive": int(len(clean)),
        "would_merge_components": int(
            sum(int(row.get("would_merge_components", 0)) for row in rows)
        ),
        "would_break_complete_tp": int(
            sum(int(row.get("would_break_complete_tp", 0)) for row in rows)
        ),
        "pairwise_tp_delta_if_all_clean": int(
            sum(int(row.get("what_if_pairwise_tp_delta", 0)) for row in clean)
        ),
        "pairwise_fp_delta_if_all_clean": int(
            sum(int(row.get("what_if_pairwise_fp_delta", 0)) for row in clean)
        ),
        "pairwise_fn_delta_if_all_clean": int(
            sum(int(row.get("what_if_pairwise_fn_delta", 0)) for row in clean)
        ),
        "complete_tp_delta_if_all_clean": int(
            sum(int(row.get("what_if_complete_track_tp_delta", 0)) for row in clean)
        ),
        "complete_fp_delta_if_all_clean": int(
            sum(int(row.get("what_if_complete_track_fp_delta", 0)) for row in clean)
        ),
        "complete_fn_delta_if_all_clean": int(
            sum(int(row.get("what_if_complete_track_fn_delta", 0)) for row in clean)
        ),
    }


def write_teacher_fn_rows(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    output_format: Literal["csv", "json"] = "csv",
) -> None:
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
    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-policy-teacher-fn-audit",
        description=(
            "Audit residual ComponentCleanup pairwise false negatives that are "
            "present in Track2p output."
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
        choices=("none", "registered-subset"),
        default="none",
        help=(
            "Do not recompute registration features by default. "
            "registered-subset fills local features for the Track2p-supported "
            "FN candidates, but can be slow."
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
    result = run_track2p_policy_teacher_fn_audit(
        config,
        threshold_method=cast(ThresholdMethod, args.threshold_method),
        iou_distance_threshold=float(args.iou_distance_threshold),
        transform_type=args.transform_type,
        cell_probability_threshold=float(args.cell_probability_threshold),
        max_gap=int(args.max_gap),
        cleanup_config=cleanup_config,
        feature_mode=cast(Literal["none", "registered-subset"], args.feature_mode),
    )
    write_teacher_fn_rows(result.edge_rows, args.output, output_format=args.format)
    if args.summary_output is not None:
        write_teacher_fn_rows(
            result.summary_rows, args.summary_output, output_format=args.format
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
