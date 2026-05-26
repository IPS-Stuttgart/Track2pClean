"""Runnable benchmark row for gap-aware pruned Track2p-policy tracking."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

from bayescatrack.experiments.track2p_benchmark import (
    OutputFormat,
    SubjectBenchmarkResult,
    Track2pBenchmarkConfig,
    _load_reference_for_subject,
    _load_subject_sessions,
    _score_prediction_against_reference,
    _validate_reference_for_benchmark,
    _validate_reference_roi_indices,
    discover_subject_dirs,
    write_results,
)
from bayescatrack.experiments.track2p_policy_benchmark import (
    TRACK2P_POLICY_DEFAULT_CELL_PROBABILITY_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE,
    ThresholdMethod,
    track2p_policy_config,
)
from bayescatrack.experiments.track2p_policy_gap_pruned import (
    DEFAULT_GAP_PRUNED_MAX_GAP,
    emulate_track2p_gap_pruned_tracks,
)
from bayescatrack.experiments.track2p_policy_pruned_benchmark import (
    Track2pPolicyPruneConfig,
)

TRACK2P_POLICY_GAP_PRUNED_METHOD = "track2p-policy-gap-pruned"


def run_track2p_policy_gap_pruned_benchmark(
    config: Track2pBenchmarkConfig,
    *,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    iou_distance_threshold: float = TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    prune_config: Track2pPolicyPruneConfig | None = None,
    transform_type: str | None = None,
    cell_probability_threshold: float | None = None,
    max_gap: int = DEFAULT_GAP_PRUNED_MAX_GAP,
) -> list[SubjectBenchmarkResult]:
    """Run Track2p-policy with both conservative gap rescue and pruning."""

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

    prune_config = prune_config or Track2pPolicyPruneConfig()
    results: list[SubjectBenchmarkResult] = []
    for subject_dir in subject_dirs:
        reference = _load_reference_for_subject(
            subject_dir, data_root=policy_config.data, config=policy_config
        )
        _validate_reference_for_benchmark(
            reference, subject_dir=subject_dir, config=policy_config
        )
        sessions = _load_subject_sessions(subject_dir, policy_config)
        _validate_reference_roi_indices(reference, sessions)
        prediction = emulate_track2p_gap_pruned_tracks(
            sessions,
            transform_type=policy_config.transform_type,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
            prune_config=prune_config,
            max_gap=int(policy_config.max_gap),
        )
        candidate_edges = int(len(prediction.diagnostics))
        pruned_edges = int(
            sum(diagnostic.pruned for diagnostic in prediction.diagnostics)
        )
        scores = {
            **_score_prediction_against_reference(
                prediction.tracks, reference, config=policy_config
            ),
            "track2p_policy_threshold_method": str(threshold_method),
            "track2p_policy_iou_distance_threshold": float(iou_distance_threshold),
            "track2p_policy_cell_probability_threshold": float(
                policy_config.cell_probability_threshold
            ),
            "track2p_policy_transform_type": str(policy_config.transform_type),
            "track2p_policy_max_gap": int(policy_config.max_gap),
            "track2p_policy_prune_threshold_margin": float(
                prune_config.threshold_margin
            ),
            "track2p_policy_prune_competition_margin": float(
                prune_config.competition_margin
            ),
            "track2p_policy_prune_min_area_ratio": float(prune_config.min_area_ratio),
            "track2p_policy_prune_centroid_distance": float(
                prune_config.centroid_distance
            ),
            "track2p_policy_candidate_edges": candidate_edges,
            "track2p_policy_pruned_edges": pruned_edges,
            "track2p_policy_kept_edges": candidate_edges - pruned_edges,
        }
        results.append(
            SubjectBenchmarkResult(
                subject=subject_dir.name,
                variant=f"Track2p-policy gap-pruned {threshold_method} gap-rescue-{policy_config.max_gap}",
                method=cast(Any, TRACK2P_POLICY_GAP_PRUNED_METHOD),
                scores=scores,
                n_sessions=len(sessions),
                reference_source=reference.source,
            )
        )
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run gap-aware pruned Track2p-policy benchmark."
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
        "--transform-type", default=TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE
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
    parser.add_argument("--max-gap", type=int, default=DEFAULT_GAP_PRUNED_MAX_GAP)
    parser.add_argument(
        "--cell-probability-threshold",
        type=float,
        default=TRACK2P_POLICY_DEFAULT_CELL_PROBABILITY_THRESHOLD,
    )
    parser.add_argument(
        "--prune-threshold-margin",
        type=float,
        default=Track2pPolicyPruneConfig().threshold_margin,
    )
    parser.add_argument(
        "--prune-competition-margin",
        type=float,
        default=Track2pPolicyPruneConfig().competition_margin,
    )
    parser.add_argument(
        "--prune-min-area-ratio",
        type=float,
        default=Track2pPolicyPruneConfig().min_area_ratio,
    )
    parser.add_argument(
        "--prune-centroid-distance",
        type=float,
        default=Track2pPolicyPruneConfig().centroid_distance,
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
    return parser


def main(argv: list[str] | None = None) -> int:
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
        max_gap=args.max_gap,
        allow_track2p_as_reference_for_smoke_test=args.allow_track2p_as_reference_for_smoke_test,
        include_behavior=args.include_behavior,
        include_non_cells=False,
        cell_probability_threshold=args.cell_probability_threshold,
        exclude_overlapping_pixels=False,
        weighted_masks=False,
        weighted_centroids=False,
    )
    prune_config = Track2pPolicyPruneConfig(
        threshold_margin=args.prune_threshold_margin,
        competition_margin=args.prune_competition_margin,
        min_area_ratio=args.prune_min_area_ratio,
        centroid_distance=args.prune_centroid_distance,
    )
    rows = [
        result.to_dict()
        for result in run_track2p_policy_gap_pruned_benchmark(
            config,
            threshold_method=cast(ThresholdMethod, args.threshold_method),
            iou_distance_threshold=args.iou_distance_threshold,
            prune_config=prune_config,
            transform_type=args.transform_type,
            cell_probability_threshold=args.cell_probability_threshold,
            max_gap=args.max_gap,
        )
    ]
    if args.output is not None:
        write_results(rows, args.output, cast(OutputFormat, args.format))
    else:
        from bayescatrack.experiments.track2p_benchmark import _write_stdout

        _write_stdout(rows, cast(OutputFormat, args.format))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
