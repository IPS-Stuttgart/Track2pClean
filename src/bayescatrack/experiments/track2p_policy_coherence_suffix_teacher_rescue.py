"""Coherence suffix stitch followed by Track2p-teacher adjacent rescue.

This command starts from the current Track2pPolicy-family lead
(ComponentCleanup + coherence suffix stitch) and then applies the guarded
Track2p-teacher adjacent-rescue edit rule.  It is a teacher-hybrid ablation:
Track2p output is an input signal, but manual-GT labels are not used to select
edits.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast

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
from bayescatrack.experiments.track2p_policy_audit import track_edge_counter
from bayescatrack.experiments.track2p_policy_benchmark import track2p_policy_config
from bayescatrack.experiments.track2p_policy_component_audit import (
    ComponentCleanupConfig,
    _evaluated_prediction_rows,
    _normalize_int_track_matrix,
)
from bayescatrack.experiments.track2p_policy_component_residual_audit import (
    _feature_subset_for_edges,
)
from bayescatrack.experiments.track2p_policy_teacher_adjacent_rescue import (
    TeacherEdgeOrder,
    apply_teacher_adjacent_rescue_edges,
    merge_teacher_feature_gates,
    teacher_feature_gate_from_preset,
)

METHOD = "track2p-policy-coherence-suffix-teacher-rescue"


def _subject_row(
    subject_dir: Path,
    *,
    config: Track2pBenchmarkConfig,
    cleanup_config: ComponentCleanupConfig,
    suffix_gate: suffix.CoherenceSuffixStitchGate,
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
    edge_top_k: int,
    path_beam_width: int,
    teacher_edge_order: TeacherEdgeOrder,
    teacher_feature_preset: str,
    max_applied_teacher_edits: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
    baseline_scores = dict(score_track_matrices(cleaned, reference_eval))
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
    suffix_scores = dict(score_track_matrices(stitched, reference_eval))

    teacher_full, _variant = _predict_subject_tracks(
        subject_dir, replace(config, method="track2p-baseline")
    )
    teacher, _reference_again, _teacher_ids = _evaluated_prediction_rows(
        _normalize_int_track_matrix(teacher_full), reference_tracks, config=config
    )
    teacher_edges = set(track_edge_counter(_normalize_int_track_matrix(teacher)))
    edge_features = _feature_subset_for_edges(
        sessions,
        teacher_edges,
        transform_type=config.transform_type,
        threshold_method=threshold_method,
        iou_distance_threshold=float(iou_distance_threshold),
    )
    teacher_report = apply_teacher_adjacent_rescue_edges(
        stitched,
        teacher,
        seed_session=config.seed_session,
        allow_completing_rescue=False,
        allow_source_backfill=True,
        allow_fragment_merges=True,
        edge_order=teacher_edge_order,
        edge_feature_index=edge_features,
        teacher_feature_gate=merge_teacher_feature_gates(
            teacher_feature_gate_from_preset(teacher_feature_preset),
            teacher_feature_gate_from_preset("none") or _empty_teacher_gate(),
        ),
        min_component_observations=1,
        max_applied_edits=max_applied_teacher_edits,
    )
    scores = dict(score_track_matrices(teacher_report.tracks, reference_eval))
    suffix_delta = suffix._score_delta(baseline_scores, suffix_scores)
    teacher_delta = suffix._score_delta(suffix_scores, scores)
    total_delta = suffix._score_delta(baseline_scores, scores)
    row = _score_row(
        subject_dir.name,
        scores,
        suffix_delta=suffix_delta,
        teacher_delta=teacher_delta,
        total_delta=total_delta,
        selected_suffixes=len(selected),
        suffix_candidates=len(paths),
        teacher_candidates=len(teacher_report.rows),
        teacher_applied=sum(
            int(edit.get("applied", 0)) for edit in teacher_report.rows
        ),
    )
    teacher_rows = [
        {**edit, "subject": subject_dir.name, "after_stage": "coherence_suffix"}
        for edit in teacher_report.rows
    ]
    return row, teacher_rows


def _empty_teacher_gate() -> Any:
    from bayescatrack.experiments.track2p_policy_teacher_adjacent_rescue import (
        TeacherEdgeFeatureGate,
    )

    return TeacherEdgeFeatureGate()


def _score_row(
    subject: str,
    scores: dict[str, Any],
    *,
    suffix_delta: dict[str, int],
    teacher_delta: dict[str, int],
    total_delta: dict[str, int],
    selected_suffixes: int,
    suffix_candidates: int,
    teacher_candidates: int,
    teacher_applied: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "subject": subject,
        "variant": "Coherence suffix stitch + Track2p teacher rescue",
        "method": METHOD,
        "reference_source": GROUND_TRUTH_REFERENCE_SOURCE,
        "selected_suffixes": int(selected_suffixes),
        "suffix_candidates": int(suffix_candidates),
        "teacher_candidates": int(teacher_candidates),
        "teacher_applied": int(teacher_applied),
    }
    for key in (
        "pairwise_true_positives",
        "pairwise_false_positives",
        "pairwise_false_negatives",
        "complete_track_true_positives",
        "complete_track_false_positives",
        "complete_track_false_negatives",
    ):
        row[key] = int(scores[key])
    row["pairwise_f1"] = suffix._f1_from_counts(
        row["pairwise_true_positives"],
        row["pairwise_false_positives"],
        row["pairwise_false_negatives"],
    )
    row["pairwise_f1_micro"] = row["pairwise_f1"]
    row["complete_track_f1"] = suffix._f1_from_counts(
        row["complete_track_true_positives"],
        row["complete_track_false_positives"],
        row["complete_track_false_negatives"],
    )
    row["complete_track_f1_micro"] = row["complete_track_f1"]
    for prefix, delta in (
        ("suffix", suffix_delta),
        ("teacher", teacher_delta),
        ("total", total_delta),
    ):
        row[f"{prefix}_pairwise_tp_delta"] = int(delta["pairwise_true_positives"])
        row[f"{prefix}_pairwise_fp_delta"] = int(delta["pairwise_false_positives"])
        row[f"{prefix}_pairwise_fn_delta"] = int(delta["pairwise_false_negatives"])
        row[f"{prefix}_complete_tp_delta"] = int(delta["complete_track_true_positives"])
        row[f"{prefix}_complete_fp_delta"] = int(
            delta["complete_track_false_positives"]
        )
        row[f"{prefix}_complete_fn_delta"] = int(
            delta["complete_track_false_negatives"]
        )
    return row


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the combined suffix-teacher CLI parser."""

    parser = suffix.build_arg_parser()
    parser.prog = (
        "python -m "
        "bayescatrack.experiments.track2p_policy_coherence_suffix_teacher_rescue"
    )
    parser.description = (
        "Run ComponentCleanup plus coherence suffix stitching, followed by a "
        "guarded Track2p-teacher adjacent-rescue pass."
    )
    parser.add_argument(
        "--teacher-edge-order",
        choices=(
            "lexicographic",
            "structural",
            "dynamic-structural",
            "confidence",
            "dynamic-confidence",
            "dynamic-seed-confidence",
        ),
        default="dynamic-confidence",
    )
    parser.add_argument(
        "--teacher-feature-preset",
        choices=(
            "none",
            "local-support",
            "high-confidence",
            "cell-high-confidence",
            "track2p-fn-rescue",
        ),
        default="track2p-fn-rescue",
    )
    parser.add_argument("--max-applied-teacher-edits", type=int, default=2)
    parser.add_argument("--teacher-output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the combined suffix-teacher benchmark row."""

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
    policy_config = track2p_policy_config(
        config,
        transform_type=args.transform_type,
        cell_probability_threshold=float(args.cell_probability_threshold),
    )
    rows: list[dict[str, Any]] = []
    teacher_rows: list[dict[str, Any]] = []
    for subject_dir in discover_subject_dirs(policy_config.data):
        row, edits = _subject_row(
            subject_dir,
            config=policy_config,
            cleanup_config=cleanup_config,
            suffix_gate=suffix_gate,
            threshold_method=cast(ThresholdMethod, args.threshold_method),
            iou_distance_threshold=float(args.iou_distance_threshold),
            edge_top_k=int(args.edge_top_k),
            path_beam_width=int(args.path_beam_width),
            teacher_edge_order=cast(TeacherEdgeOrder, args.teacher_edge_order),
            teacher_feature_preset=str(args.teacher_feature_preset),
            max_applied_teacher_edits=(
                None
                if int(args.max_applied_teacher_edits) < 0
                else int(args.max_applied_teacher_edits)
            ),
        )
        rows.append(row)
        teacher_rows.extend(edits)
    suffix.write_rows(
        rows,
        args.output,
        output_format=cast(Literal["csv", "json"], args.format),
    )
    if args.teacher_output is not None:
        suffix.write_rows(
            teacher_rows,
            args.teacher_output,
            output_format=cast(Literal["csv", "json"], args.format),
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())