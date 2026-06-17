"""Audit Track2p-teacher adjacent edges after coherence suffix stitching.

The audit starts from the frozen Track2pPolicy ComponentCleanup row, derives the
same Track2p-teacher adjacent-rescue candidates used by the teacher row, overlays
those candidates on the CoherenceSuffixStitch prediction, and scores each edge
as a one-edit what-if.  Manual-GT labels are used only for audit labels and score
deltas, not for selecting candidate edges.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
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
from bayescatrack.experiments.track2p_policy_benchmark import track2p_policy_config
from bayescatrack.experiments.track2p_policy_component_audit import (
    ComponentCleanupConfig,
    _evaluated_prediction_rows,
    _normalize_int_track_matrix,
)
from bayescatrack.experiments.track2p_policy_teacher_adjacent_rescue import (
    TeacherAdjacentRescueReport,
    _try_apply_teacher_edge,
    apply_teacher_adjacent_rescue_edges,
)

METHOD = "track2p-policy-coherence-teacher-overlay-audit"

FIELDNAMES = (
    "subject",
    "session_a",
    "session_b",
    "roi_a",
    "roi_b",
    "already_in_component_cleanup",
    "already_in_coherence_suffix_stitch",
    "track2p_supported",
    "edge_status_against_gt",
    "creates_duplicate_source",
    "creates_duplicate_target",
    "would_break_complete_tp",
    "would_create_complete_fp",
    "pairwise_tp_delta_if_added",
    "pairwise_fp_delta_if_added",
    "pairwise_fn_delta_if_added",
    "complete_tp_delta_if_added",
    "complete_fp_delta_if_added",
    "complete_fn_delta_if_added",
)


@dataclass(frozen=True)
class CoherenceTeacherOverlayAuditResult:
    """Rows for the teacher-overlay audit."""

    rows: tuple[dict[str, Any], ...]


def run_track2p_policy_coherence_teacher_overlay_audit(
    config: Track2pBenchmarkConfig,
    *,
    threshold_method: ThresholdMethod = "min",
    iou_distance_threshold: float = 12.0,
    transform_type: str | None = None,
    cell_probability_threshold: float | None = None,
    cleanup_config: ComponentCleanupConfig | None = None,
    suffix_gate: suffix.CoherenceSuffixStitchGate | None = None,
    edge_top_k: int = 25,
    path_beam_width: int = 100,
) -> CoherenceTeacherOverlayAuditResult:
    """Return one row per ComponentCleanup teacher-rescue edge."""

    policy_config = track2p_policy_config(
        config,
        transform_type=transform_type,
        cell_probability_threshold=cell_probability_threshold,
    )
    cleanup_config = cleanup_config or ComponentCleanupConfig()
    suffix_gate = suffix_gate or suffix.CoherenceSuffixStitchGate()

    rows: list[dict[str, Any]] = []
    for subject_dir in discover_subject_dirs(policy_config.data):
        rows.extend(
            _subject_rows(
                subject_dir,
                config=policy_config,
                cleanup_config=cleanup_config,
                suffix_gate=suffix_gate,
                threshold_method=threshold_method,
                iou_distance_threshold=float(iou_distance_threshold),
                edge_top_k=int(edge_top_k),
                path_beam_width=int(path_beam_width),
            )
        )
    return CoherenceTeacherOverlayAuditResult(tuple(rows))


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

    cleaned, reference_eval = suffix._component_cleanup_eval(
        sessions,
        reference_tracks,
        subject=subject_dir.name,
        config=config,
        cleanup_config=cleanup_config,
        threshold_method=threshold_method,
        iou_distance_threshold=float(iou_distance_threshold),
    )
    feature_cache = suffix._FeatureCache(
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
    stitched = suffix._apply_suffix_paths(cleaned, selected)

    teacher_full, _variant = _predict_subject_tracks(
        subject_dir, replace(config, method="track2p-baseline")
    )
    teacher, _reference_again, _teacher_ids = _evaluated_prediction_rows(
        _normalize_int_track_matrix(teacher_full), reference_tracks, config=config
    )
    teacher_report = apply_teacher_adjacent_rescue_edges(
        cleaned,
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
    return _audit_teacher_edges(
        subject=subject_dir.name,
        cleaned=cleaned,
        stitched=stitched,
        reference=reference_eval,
        teacher=teacher,
        teacher_report=teacher_report,
        seed_session=config.seed_session,
    )


def _audit_teacher_edges(
    *,
    subject: str,
    cleaned: np.ndarray,
    stitched: np.ndarray,
    reference: np.ndarray,
    teacher: np.ndarray,
    teacher_report: TeacherAdjacentRescueReport,
    seed_session: int,
) -> list[dict[str, Any]]:
    baseline_scores = dict(score_track_matrices(stitched, reference))
    cleaned_counts = track_edge_counter(cleaned)
    stitched_counts = track_edge_counter(stitched)
    teacher_counts = track_edge_counter(teacher)
    reference_counts = track_edge_counter(reference)
    rows: list[dict[str, Any]] = []
    for teacher_row in teacher_report.rows:
        edge = _edge_from_row(teacher_row)
        occurrence_index = int(teacher_row.get("occurrence_index", 0))
        candidate, _attempt = _try_apply_teacher_edge(
            stitched,
            edge,
            seed_session=int(seed_session),
            allow_completing_rescue=False,
            allow_source_backfill=True,
            allow_fragment_merges=True,
            min_component_observations=1,
        )
        candidate_scores = dict(score_track_matrices(candidate, reference))
        delta = suffix._score_delta(baseline_scores, candidate_scores)
        conflict_flags = _conflict_flags(stitched, edge)
        rows.append(
            {
                "subject": subject,
                "session_a": int(edge[0]),
                "session_b": int(edge[1]),
                "roi_a": int(edge[2]),
                "roi_b": int(edge[3]),
                "already_in_component_cleanup": int(
                    cleaned_counts.get(edge, 0) > occurrence_index
                ),
                "already_in_coherence_suffix_stitch": int(
                    stitched_counts.get(edge, 0) > occurrence_index
                ),
                "track2p_supported": int(
                    teacher_counts.get(edge, 0) > occurrence_index
                ),
                "edge_status_against_gt": (
                    "true_positive"
                    if reference_counts.get(edge, 0) > occurrence_index
                    else "false_positive"
                ),
                "creates_duplicate_source": int(conflict_flags["source"]),
                "creates_duplicate_target": int(conflict_flags["target"]),
                "would_break_complete_tp": int(
                    delta["complete_track_true_positives"] < 0
                ),
                "would_create_complete_fp": int(
                    delta["complete_track_false_positives"] > 0
                ),
                "pairwise_tp_delta_if_added": int(delta["pairwise_true_positives"]),
                "pairwise_fp_delta_if_added": int(delta["pairwise_false_positives"]),
                "pairwise_fn_delta_if_added": int(delta["pairwise_false_negatives"]),
                "complete_tp_delta_if_added": int(
                    delta["complete_track_true_positives"]
                ),
                "complete_fp_delta_if_added": int(
                    delta["complete_track_false_positives"]
                ),
                "complete_fn_delta_if_added": int(
                    delta["complete_track_false_negatives"]
                ),
            }
        )
    return rows


def _edge_from_row(row: Mapping[str, Any]) -> TrackEdge:
    return (
        int(row["session_a"]),
        int(row["session_b"]),
        int(row["roi_a"]),
        int(row["roi_b"]),
    )


def _conflict_flags(predicted: np.ndarray, edge: TrackEdge) -> dict[str, bool]:
    session_a, session_b, roi_a, roi_b = edge
    source_rows = tuple(np.flatnonzero(predicted[:, session_a] == roi_a))
    target_rows = tuple(np.flatnonzero(predicted[:, session_b] == roi_b))
    source_conflict = len(source_rows) > 1 or (
        len(source_rows) == 1
        and predicted[int(source_rows[0]), session_b] >= 0
        and int(predicted[int(source_rows[0]), session_b]) != int(roi_b)
    )
    target_conflict = len(target_rows) > 1 or (
        len(target_rows) == 1
        and predicted[int(target_rows[0]), session_a] >= 0
        and int(predicted[int(target_rows[0]), session_a]) != int(roi_a)
    )
    return {"source": bool(source_conflict), "target": bool(target_conflict)}


def write_rows(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    output_format: Literal["csv", "json"] = "csv",
) -> None:
    """Write audit rows with the requested stable column order."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output_path.write_text(
            json.dumps(list(rows), indent=2) + "\n", encoding="utf-8"
        )
        return
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the overlay-audit command parser."""

    parser = suffix.build_arg_parser()
    parser.prog = (
        "python -m "
        "bayescatrack.experiments.track2p_policy_coherence_teacher_overlay_audit"
    )
    parser.description = (
        "Audit Track2p-teacher adjacent rescue edges from ComponentCleanup as "
        "one-edit overlays after CoherenceSuffixStitch."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CoherenceSuffixStitch teacher-overlay audit."""

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
        max_gap=args.max_gap,
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
    result = run_track2p_policy_coherence_teacher_overlay_audit(
        config,
        threshold_method=cast(ThresholdMethod, args.threshold_method),
        iou_distance_threshold=float(args.iou_distance_threshold),
        transform_type=args.transform_type,
        cell_probability_threshold=float(args.cell_probability_threshold),
        cleanup_config=cleanup_config,
        suffix_gate=suffix_gate,
        edge_top_k=int(args.edge_top_k),
        path_beam_width=int(args.path_beam_width),
    )
    write_rows(
        result.rows,
        args.output,
        output_format=cast(Literal["csv", "json"], args.format),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
