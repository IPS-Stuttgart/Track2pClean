"""Gap-rescue plus weakest-bridge cleanup for Track2p-policy rows.

The plain Track2p-policy runner can conservatively continue a seed track across
an isolated missing adjacent link when ``max_gap > 1``. The component-cleanup
runner can split false bridges in the resulting track matrix. This module
combines the two operations in a distinct benchmark row: first run gap-rescue
Track2p-policy propagation, then apply weakest-bridge cleanup to the resulting
track matrix.

Unlike the adjacent-only component-cleanup row, this gap-rescue row defaults to
allowing incomplete-track splits and one-observation side fragments. That makes
the cleanup operational on gap-rescued rows such as ``[seed, -1, suffix, ...]``
instead of silently protecting the false suffix whenever the rescued bridge is
weak.
"""

from __future__ import annotations

import argparse
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
    _evaluated_prediction_rows,
    _mark_applied_splits,
    _normalize_int_track_matrix,
    apply_weakest_bridge_splits,
    component_audit_rows,
    write_component_rows,
)
from bayescatrack.experiments.track2p_policy_pruned_benchmark import (
    Track2pPolicyPruneConfig,
    emulate_track2p_pruned_tracks,
)

TRACK2P_POLICY_GAP_COMPONENT_CLEANUP_METHOD = "track2p-policy-gap-component-cleanup"
TRACK2P_POLICY_GAP_COMPONENT_DEFAULT_MAX_GAP = 2
TRACK2P_POLICY_GAP_COMPONENT_DEFAULT_MIN_SIDE_OBSERVATIONS = 1
TRACK2P_POLICY_GAP_COMPONENT_DEFAULT_REQUIRE_COMPLETE_TRACK = False


def run_track2p_policy_gap_component_cleanup(
    config: Track2pBenchmarkConfig,
    *,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    iou_distance_threshold: float = TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    transform_type: str | None = None,
    cell_probability_threshold: float | None = None,
    max_gap: int | None = None,
    cleanup_config: ComponentCleanupConfig | None = None,
    apply_splits: bool = True,
) -> ComponentAuditOutput:
    """Run gap-rescue Track2p-policy and split high-risk component bridges."""

    policy_config = track2p_policy_config(
        config,
        transform_type=transform_type,
        cell_probability_threshold=cell_probability_threshold,
        max_gap=max_gap,
    )
    if int(policy_config.max_gap) < 1:
        raise ValueError("max_gap must be at least 1")
    subject_dirs = discover_subject_dirs(policy_config.data)
    if not subject_dirs:
        raise ValueError(
            f"No Track2p-style subject directories found under {policy_config.data}"
        )

    cleanup_config = cleanup_config or _default_gap_cleanup_config()
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
                "Track2p-policy gap component cleanup requires independent "
                "manual GT references"
            )
        sessions = _load_subject_sessions(subject_dir, policy_config)
        _validate_reference_roi_indices(reference, sessions)
        predicted_full = _normalize_int_track_matrix(
            emulate_track2p_tracks(
                sessions,
                transform_type=policy_config.transform_type,
                threshold_method=threshold_method,
                iou_distance_threshold=float(iou_distance_threshold),
                max_gap=int(policy_config.max_gap),
            )
        )
        diagnostic_prediction = emulate_track2p_pruned_tracks(
            sessions,
            transform_type=policy_config.transform_type,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
            prune_config=_no_prune_config(),
        )
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
        audit_rows = component_audit_rows(
            predicted_eval,
            reference_eval,
            sessions=sessions,
            diagnostics=diagnostic_prediction.diagnostics,
            subject=subject_dir.name,
            config=cleanup_config,
            track_ids=evaluated_track_ids,
            seed_session=policy_config.seed_session,
        )
        subject_rows = _mark_applied_splits(audit_rows, apply_splits=apply_splits)
        cleaned = (
            apply_weakest_bridge_splits(predicted_full, subject_rows)
            if apply_splits
            else predicted_full
        )
        scores = _score_prediction_against_reference(
            cleaned, reference, config=policy_config
        )
        scores = _with_scores_metadata(
            scores,
            threshold_method=threshold_method,
            iou_distance_threshold=iou_distance_threshold,
            cleanup_config=cleanup_config,
            max_gap=policy_config.max_gap,
            cell_probability_threshold=policy_config.cell_probability_threshold,
            transform_type=policy_config.transform_type,
            apply_splits=apply_splits,
            component_rows=subject_rows,
        )
        results.append(
            SubjectBenchmarkResult(
                subject=subject_dir.name,
                variant=(
                    "Track2p-policy gap-rescue weakest-bridge component split"
                    if apply_splits
                    else "Track2p-policy gap-rescue component audit"
                ),
                method=cast(Any, TRACK2P_POLICY_GAP_COMPONENT_CLEANUP_METHOD),
                scores=scores,
                n_sessions=len(sessions),
                reference_source=reference.source,
            )
        )
        component_rows.extend(
            _with_metadata(
                subject_rows,
                {
                    "threshold_method": str(threshold_method),
                    "iou_distance_threshold": float(iou_distance_threshold),
                    "cell_probability_threshold": float(
                        policy_config.cell_probability_threshold
                    ),
                    "transform_type": str(policy_config.transform_type),
                    "max_gap": int(policy_config.max_gap),
                    "cleanup_method": TRACK2P_POLICY_GAP_COMPONENT_CLEANUP_METHOD,
                },
            )
        )
    return ComponentAuditOutput(tuple(results), tuple(component_rows))


def _default_gap_cleanup_config() -> ComponentCleanupConfig:
    """Return cleanup defaults tuned for gap-rescue rows.

    Gap rescue can produce useful but incomplete rows. Requiring a complete row or
    at least two observations on both sides prevents the cleanup from removing a
    weak false suffix after a single seed observation. The gap-specific benchmark
    row therefore uses more permissive side-length defaults while keeping the
    existing risk threshold and scoring metadata intact.
    """

    return ComponentCleanupConfig(
        min_side_observations=(
            TRACK2P_POLICY_GAP_COMPONENT_DEFAULT_MIN_SIDE_OBSERVATIONS
        ),
        require_complete_track=(
            TRACK2P_POLICY_GAP_COMPONENT_DEFAULT_REQUIRE_COMPLETE_TRACK
        ),
    )


def _with_scores_metadata(
    scores: Mapping[str, float | int | str],
    *,
    threshold_method: ThresholdMethod,
    iou_distance_threshold: float,
    cleanup_config: ComponentCleanupConfig,
    max_gap: int,
    cell_probability_threshold: float,
    transform_type: str,
    apply_splits: bool,
    component_rows: Sequence[Mapping[str, float | int | str]],
) -> dict[str, float | int | str]:
    candidate_splits = int(
        sum(int(row["would_split_at_weakest_edge"]) for row in component_rows)
    )
    applied_splits = int(
        sum(int(row["applied_split"]) for row in component_rows) if apply_splits else 0
    )
    return {
        **dict(scores),
        "track2p_policy_threshold_method": str(threshold_method),
        "track2p_policy_iou_distance_threshold": float(iou_distance_threshold),
        "track2p_policy_cell_probability_threshold": float(cell_probability_threshold),
        "track2p_policy_transform_type": str(transform_type),
        "track2p_policy_max_gap": int(max_gap),
        "track2p_component_apply_splits": int(apply_splits),
        "track2p_component_candidate_splits": candidate_splits,
        "track2p_component_applied_splits": applied_splits,
        "track2p_component_split_risk_threshold": float(
            cleanup_config.split_risk_threshold
        ),
        "track2p_component_split_penalty": float(cleanup_config.split_penalty),
        "track2p_component_min_side_observations": int(
            cleanup_config.min_side_observations
        ),
        "track2p_component_require_complete_track": int(
            cleanup_config.require_complete_track
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for gap-rescue component cleanup."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-policy-gap-component-cleanup",
        description=(
            "Run Track2p-policy gap rescue followed by weakest-bridge "
            "component cleanup."
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
    parser.add_argument(
        "--max-gap",
        type=int,
        default=TRACK2P_POLICY_GAP_COMPONENT_DEFAULT_MAX_GAP,
        help=(
            "Maximum session jump for conservative gap rescue before cleanup. "
            "Consecutive Track2p-policy links are still preferred."
        ),
    )
    parser.add_argument(
        "--apply-splits", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--require-complete-track",
        action=argparse.BooleanOptionalAction,
        default=TRACK2P_POLICY_GAP_COMPONENT_DEFAULT_REQUIRE_COMPLETE_TRACK,
        help=(
            "Only split components observed in every session. Defaults to false "
            "for gap-rescue cleanup so weak incomplete rescued suffixes can be "
            "removed."
        ),
    )
    parser.add_argument("--threshold-margin-scale", type=float, default=0.10)
    parser.add_argument("--competition-margin-scale", type=float, default=0.20)
    parser.add_argument("--area-ratio-floor", type=float, default=0.45)
    parser.add_argument("--centroid-distance-scale", type=float, default=4.0)
    parser.add_argument("--split-risk-threshold", type=float, default=1.50)
    parser.add_argument("--split-penalty", type=float, default=0.25)
    parser.add_argument(
        "--min-side-observations",
        type=int,
        default=TRACK2P_POLICY_GAP_COMPONENT_DEFAULT_MIN_SIDE_OBSERVATIONS,
        help=(
            "Minimum observations retained on both sides of a split. Defaults "
            "to one for gap-rescue cleanup so a seed-only left fragment can "
            "shed a weak false suffix."
        ),
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
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("table", "json", "csv"), default="table")
    parser.add_argument("--component-output", type=Path, default=None)
    parser.add_argument("--component-format", choices=("csv", "json"), default="csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Track2p-policy gap-rescue component-cleanup CLI."""

    args = build_arg_parser().parse_args(argv)
    cleanup_config = ComponentCleanupConfig(
        threshold_margin_scale=args.threshold_margin_scale,
        competition_margin_scale=args.competition_margin_scale,
        area_ratio_floor=args.area_ratio_floor,
        centroid_distance_scale=args.centroid_distance_scale,
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
    output = run_track2p_policy_gap_component_cleanup(
        config,
        threshold_method=cast(ThresholdMethod, args.threshold_method),
        iou_distance_threshold=args.iou_distance_threshold,
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
            args.component_output,
            output_format=cast(Literal["csv", "json"], args.component_format),
        )
    return 0


def _no_prune_config() -> Track2pPolicyPruneConfig:
    """Return a pruning config that keeps every threshold-accepted policy edge."""

    return Track2pPolicyPruneConfig(
        threshold_margin=0.0,
        competition_margin=0.0,
        min_area_ratio=0.0,
        centroid_distance=float("inf"),
    )


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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
