"""Gap-rescue plus consensus bridge cleanup for Track2p-policy rows.

This benchmark row combines two orthogonal Track2p-policy improvement levers:

* conservative gap rescue, which can recover an isolated missing adjacent link, and
* consensus bridge cleanup, which cuts risky adjacent continuations only when
  component-level diagnostics and threshold-stability evidence agree.

It is intentionally exposed as a separate row so the frozen policy/component
cleanup results remain auditable.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

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
from bayescatrack.experiments.track2p_emulation_benchmark import emulate_track2p_tracks
from bayescatrack.experiments.track2p_policy_benchmark import (
    TRACK2P_POLICY_DEFAULT_CELL_PROBABILITY_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE,
    ThresholdMethod,
    track2p_policy_config,
)
from bayescatrack.experiments.track2p_policy_component_audit import (
    ComponentAuditOutput,
    ComponentCleanupConfig,
    _diagnostics_by_suite2p_edge,
    _no_prune_config,
    _normalize_int_track_matrix,
    component_audit_rows,
    write_component_rows,
)
from bayescatrack.experiments.track2p_policy_consensus_cleanup import (
    CONSENSUS_MODES,
    ConsensusCleanupConfig,
    ConsensusMode,
    _enrich_component_rows_with_split_plan,
    _evaluated_prediction_rows,
    _float_tuple_arg,
    apply_consensus_bridge_splits,
    plan_consensus_bridge_splits,
)
from bayescatrack.experiments.track2p_policy_pruned_benchmark import (
    emulate_track2p_pruned_tracks,
)
from bayescatrack.experiments.track2p_policy_stability_cleanup import (
    StabilityCleanupConfig,
    edge_support_counts,
)

TRACK2P_POLICY_GAP_CONSENSUS_CLEANUP_METHOD = (
    "track2p-policy-gap-consensus-cleanup"
)
TRACK2P_POLICY_GAP_CONSENSUS_DEFAULT_MAX_GAP = 2


def run_track2p_policy_gap_consensus_cleanup(
    config: Track2pBenchmarkConfig,
    *,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    transform_type: str | None = None,
    cell_probability_threshold: float | None = None,
    max_gap: int | None = None,
    cleanup_config: ConsensusCleanupConfig | None = None,
    apply_splits: bool = True,
) -> ComponentAuditOutput:
    """Run gap-rescue Track2p-policy and consensus split risky bridges."""

    cleanup = cleanup_config or ConsensusCleanupConfig()
    policy_config = track2p_policy_config(
        config,
        transform_type=transform_type,
        cell_probability_threshold=cell_probability_threshold,
        max_gap=(
            TRACK2P_POLICY_GAP_CONSENSUS_DEFAULT_MAX_GAP
            if max_gap is None
            else max_gap
        ),
    )
    if int(policy_config.max_gap) < 1:
        raise ValueError("max_gap must be at least 1")
    subject_dirs = discover_subject_dirs(policy_config.data)
    if not subject_dirs:
        raise ValueError(
            f"No Track2p-style subject directories found under {policy_config.data}"
        )

    results: list[SubjectBenchmarkResult] = []
    component_rows: list[dict[str, float | int | str]] = []
    for subject_dir in subject_dirs:
        reference = _load_reference_for_subject(
            subject_dir, data_root=policy_config.data, config=policy_config
        )
        _validate_reference_for_benchmark(
            reference, subject_dir=subject_dir, config=policy_config
        )
        if reference.source != GROUND_TRUTH_REFERENCE_SOURCE:
            raise ValueError(
                "Track2p-policy gap consensus cleanup requires independent manual "
                "GT references"
            )
        sessions = _load_subject_sessions(subject_dir, policy_config)
        _validate_reference_roi_indices(reference, sessions)

        predicted_full = _normalize_int_track_matrix(
            emulate_track2p_tracks(
                sessions,
                transform_type=policy_config.transform_type,
                threshold_method=threshold_method,
                iou_distance_threshold=cleanup.stability.base_iou_distance_threshold,
                max_gap=int(policy_config.max_gap),
            )
        )
        ensemble_predictions = tuple(
            emulate_track2p_tracks(
                sessions,
                transform_type=policy_config.transform_type,
                threshold_method=threshold_method,
                iou_distance_threshold=threshold,
                max_gap=int(policy_config.max_gap),
            )
            for threshold in cleanup.stability.ensemble_iou_distance_thresholds
        )
        diagnostic_prediction = emulate_track2p_pruned_tracks(
            sessions,
            transform_type=policy_config.transform_type,
            threshold_method=threshold_method,
            iou_distance_threshold=cleanup.stability.base_iou_distance_threshold,
            prune_config=_no_prune_config(),
        )
        diagnostics_by_edge = _diagnostics_by_suite2p_edge(
            sessions, diagnostic_prediction.diagnostics
        )
        support_counts = edge_support_counts(ensemble_predictions)
        reference_tracks = _reference_matrix(
            reference, curated_only=policy_config.curated_only
        )
        predicted_eval, reference_eval, evaluated_track_ids = (
            _evaluated_prediction_rows(
                predicted_full,
                reference_tracks,
                config=policy_config,
            )
        )
        subject_rows = component_audit_rows(
            predicted_eval,
            reference_eval,
            sessions=sessions,
            diagnostics=diagnostic_prediction.diagnostics,
            subject=subject_dir.name,
            config=cleanup.component,
            track_ids=evaluated_track_ids,
            seed_session=policy_config.seed_session,
        )
        split_plan = plan_consensus_bridge_splits(
            predicted_eval,
            diagnostics_by_edge=diagnostics_by_edge,
            support_counts=support_counts,
            config=cleanup.split_config,
            track_ids=evaluated_track_ids,
        )
        enriched_rows = _enrich_component_rows_with_split_plan(
            subject_rows,
            split_plan,
            support_counts=support_counts,
            apply_splits=apply_splits,
        )
        cleaned = (
            apply_consensus_bridge_splits(predicted_full, split_plan)
            if apply_splits
            else predicted_full
        )
        scores = _score_prediction_against_reference(
            cleaned, reference, config=policy_config
        )
        candidate_splits = int(sum(len(splits) for splits in split_plan.values()))
        applied_splits = int(candidate_splits if apply_splits else 0)
        scores = {
            **scores,
            "track2p_policy_variant": TRACK2P_POLICY_GAP_CONSENSUS_CLEANUP_METHOD,
            "track2p_policy_threshold_method": str(threshold_method),
            "track2p_policy_base_iou_distance_threshold": float(
                cleanup.stability.base_iou_distance_threshold
            ),
            "track2p_policy_stability_iou_distance_thresholds": json.dumps(
                list(cleanup.stability.ensemble_iou_distance_thresholds)
            ),
            "track2p_policy_cell_probability_threshold": float(
                policy_config.cell_probability_threshold
            ),
            "track2p_policy_transform_type": str(policy_config.transform_type),
            "track2p_policy_max_gap": int(policy_config.max_gap),
            "track2p_policy_consensus_mode": str(cleanup.mode),
            "track2p_policy_consensus_required_support_votes": int(
                cleanup.stability.required_support_votes
            ),
            "track2p_policy_consensus_max_splits_per_component": int(
                cleanup.max_splits_per_component
            ),
            "track2p_component_apply_splits": int(apply_splits),
            "track2p_component_candidate_splits": candidate_splits,
            "track2p_component_applied_splits": applied_splits,
            "track2p_component_split_risk_threshold": float(
                cleanup.component.split_risk_threshold
            ),
            "track2p_component_split_penalty": float(cleanup.component.split_penalty),
            "track2p_component_min_side_observations": int(
                cleanup.component.min_side_observations
            ),
            "track2p_component_require_complete_track": int(
                cleanup.component.require_complete_track
            ),
        }
        results.append(
            SubjectBenchmarkResult(
                subject=subject_dir.name,
                variant=(
                    "Track2p-policy gap-rescue consensus cleanup"
                    if apply_splits
                    else "Track2p-policy gap-rescue consensus audit"
                ),
                method=cast(Any, TRACK2P_POLICY_GAP_CONSENSUS_CLEANUP_METHOD),
                scores=scores,
                n_sessions=len(sessions),
                reference_source=reference.source,
            )
        )
        component_rows.extend(
            _with_metadata(
                enriched_rows,
                {
                    "threshold_method": str(threshold_method),
                    "base_iou_distance_threshold": float(
                        cleanup.stability.base_iou_distance_threshold
                    ),
                    "stability_iou_distance_thresholds": json.dumps(
                        list(cleanup.stability.ensemble_iou_distance_thresholds)
                    ),
                    "required_support_votes": int(
                        cleanup.stability.required_support_votes
                    ),
                    "cell_probability_threshold": float(
                        policy_config.cell_probability_threshold
                    ),
                    "transform_type": str(policy_config.transform_type),
                    "max_gap": int(policy_config.max_gap),
                    "cleanup_method": TRACK2P_POLICY_GAP_CONSENSUS_CLEANUP_METHOD,
                    "consensus_mode": str(cleanup.mode),
                },
            )
        )
    return ComponentAuditOutput(tuple(results), tuple(component_rows))


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for gap-rescue consensus cleanup."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-policy-gap-consensus-cleanup",
        description=(
            "Run Track2p-policy gap rescue followed by conservative consensus "
            "bridge cleanup."
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
        "--base-iou-distance-threshold",
        type=float,
        default=TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    )
    parser.add_argument(
        "--stability-iou-distance-thresholds",
        type=_float_tuple_arg,
        default=(10.0, 12.0, 14.0),
        help="Comma-separated IoU-distance thresholds used for stability voting.",
    )
    parser.add_argument(
        "--cell-probability-threshold",
        type=float,
        default=TRACK2P_POLICY_DEFAULT_CELL_PROBABILITY_THRESHOLD,
    )
    parser.add_argument(
        "--transform-type", default=TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=TRACK2P_POLICY_GAP_CONSENSUS_DEFAULT_MAX_GAP,
        help=(
            "Maximum session jump for conservative gap rescue before consensus "
            "cleanup. Consecutive Track2p-policy links are still preferred."
        ),
    )
    parser.add_argument(
        "--apply-splits", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--require-complete-track", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--consensus-mode",
        choices=CONSENSUS_MODES,
        default="risk-and-stability",
    )
    parser.add_argument("--threshold-margin-scale", type=float, default=0.10)
    parser.add_argument("--competition-margin-scale", type=float, default=0.20)
    parser.add_argument("--area-ratio-floor", type=float, default=0.45)
    parser.add_argument("--centroid-distance-scale", type=float, default=4.0)
    parser.add_argument("--split-risk-threshold", type=float, default=1.50)
    parser.add_argument("--split-penalty", type=float, default=0.25)
    parser.add_argument("--min-side-observations", type=int, default=2)
    parser.add_argument("--max-splits-per-component", type=int, default=2)
    parser.add_argument("--min-support-fraction", type=float, default=2.0 / 3.0)
    parser.add_argument("--min-support-votes", type=int, default=None)
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
    parser.add_argument("--component-output", type=Path, default=None)
    parser.add_argument("--component-format", choices=("csv", "json"), default="csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Track2p-policy gap consensus cleanup CLI."""

    args = build_arg_parser().parse_args(argv)
    component_config = ComponentCleanupConfig(
        threshold_margin_scale=args.threshold_margin_scale,
        competition_margin_scale=args.competition_margin_scale,
        area_ratio_floor=args.area_ratio_floor,
        centroid_distance_scale=args.centroid_distance_scale,
        split_risk_threshold=args.split_risk_threshold,
        split_penalty=args.split_penalty,
        min_side_observations=args.min_side_observations,
        require_complete_track=args.require_complete_track,
    )
    stability_config = StabilityCleanupConfig(
        iou_distance_thresholds=tuple(args.stability_iou_distance_thresholds),
        base_iou_distance_threshold=args.base_iou_distance_threshold,
        min_support_fraction=args.min_support_fraction,
        min_support_votes=args.min_support_votes,
        min_side_observations=args.min_side_observations,
    )
    cleanup_config = ConsensusCleanupConfig(
        component=component_config,
        stability=stability_config,
        max_splits_per_component=args.max_splits_per_component,
        mode=cast(ConsensusMode, args.consensus_mode),
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
    output = run_track2p_policy_gap_consensus_cleanup(
        config,
        threshold_method=cast(ThresholdMethod, args.threshold_method),
        transform_type=args.transform_type,
        cell_probability_threshold=args.cell_probability_threshold,
        max_gap=args.max_gap,
        cleanup_config=cleanup_config,
        apply_splits=args.apply_splits,
    )
    rows = [result.to_dict() for result in output.results]
    if args.output is not None:
        write_results(rows, args.output, cast(OutputFormat, args.format))
    else:
        from bayescatrack.experiments.track2p_benchmark import _write_stdout

        _write_stdout(rows, cast(OutputFormat, args.format))
    if args.component_output is not None:
        write_component_rows(
            output.component_rows,
            output_path=args.component_output,
            output_format=cast(Literal["csv", "json"], args.component_format),
        )
    return 0


def _with_metadata(
    rows: Sequence[Mapping[str, float | int | str]],
    metadata: Mapping[str, Any],
) -> list[dict[str, float | int | str]]:
    formatted = {key: _format_metadata_value(value) for key, value in metadata.items()}
    return [{**dict(row), **formatted} for row in rows]


def _format_metadata_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


__all__ = (
    "TRACK2P_POLICY_GAP_CONSENSUS_CLEANUP_METHOD",
    "TRACK2P_POLICY_GAP_CONSENSUS_DEFAULT_MAX_GAP",
    "build_arg_parser",
    "run_track2p_policy_gap_consensus_cleanup",
)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
