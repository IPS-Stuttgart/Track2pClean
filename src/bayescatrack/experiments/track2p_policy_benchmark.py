"""First-class Track2p-policy benchmark method for BayesCaTrack.

The policy runner promotes the strongest Track2p-emulation setting from the
standalone diagnostic into a normal benchmarkable method.  It deliberately keeps
Track2p's high-performing inductive bias: hard Suite2p cell filtering,
consecutive-session affine registration, Hungarian matching on registered IoU,
minimum-threshold filtering, and greedy first-session propagation.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast

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
from bayescatrack.experiments.track2p_emulation_benchmark import (
    emulate_track2p_tracks,
)

ThresholdMethod = Literal["otsu", "min"]
TRACK2P_POLICY_METHOD = "track2p-policy"
TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE = "affine"
TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD: ThresholdMethod = "min"
TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD = 12.0
TRACK2P_POLICY_DEFAULT_CELL_PROBABILITY_THRESHOLD = 0.5
TRACK2P_POLICY_DEFAULT_MAX_GAP = 1


def track2p_policy_config(
    config: Track2pBenchmarkConfig,
    *,
    transform_type: str | None = None,
    cell_probability_threshold: float | None = None,
) -> Track2pBenchmarkConfig:
    """Return ``config`` with the first-class Track2p-policy defaults applied."""

    return replace(
        config,
        method="global-assignment",
        transform_type=(
            TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE
            if transform_type is None
            else str(transform_type)
        ),
        max_gap=TRACK2P_POLICY_DEFAULT_MAX_GAP,
        include_non_cells=False,
        cell_probability_threshold=(
            TRACK2P_POLICY_DEFAULT_CELL_PROBABILITY_THRESHOLD
            if cell_probability_threshold is None
            else float(cell_probability_threshold)
        ),
        weighted_masks=False,
        weighted_centroids=False,
        exclude_overlapping_pixels=False,
    )


def run_track2p_policy_benchmark(
    config: Track2pBenchmarkConfig,
    *,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    iou_distance_threshold: float = TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    transform_type: str | None = None,
    cell_probability_threshold: float | None = None,
) -> list[SubjectBenchmarkResult]:
    """Run the Track2p-policy benchmark row over all discovered subjects."""

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
        predicted = emulate_track2p_tracks(
            sessions,
            transform_type=policy_config.transform_type,
            threshold_method=threshold_method,
            iou_distance_threshold=float(iou_distance_threshold),
        )
        scores = _score_prediction_against_reference(
            predicted, reference, config=policy_config
        )
        scores = {
            **scores,
            "track2p_policy_threshold_method": str(threshold_method),
            "track2p_policy_iou_distance_threshold": float(iou_distance_threshold),
            "track2p_policy_cell_probability_threshold": float(
                policy_config.cell_probability_threshold
            ),
            "track2p_policy_transform_type": str(policy_config.transform_type),
        }
        results.append(
            SubjectBenchmarkResult(
                subject=subject_dir.name,
                variant=f"Track2p-policy {threshold_method}",
                method=cast(Any, TRACK2P_POLICY_METHOD),
                scores=scores,
                n_sessions=len(sessions),
                reference_source=reference.source,
            )
        )
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the first-class policy method."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-policy",
        description="Run the first-class Track2p-policy benchmark method.",
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
        "--transform-type",
        default=TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE,
        help="Registration transform; defaults to the tuned Track2p-elastix affine policy.",
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
    """Run the Track2p-policy benchmark CLI."""

    args = build_arg_parser().parse_args(argv)
    config = Track2pBenchmarkConfig(
        data=args.data,
        method="global-assignment",
        plane_name=args.plane_name,
        input_format=args.input_format,
        reference=args.reference,
        reference_kind=args.reference_kind,
        allow_track2p_as_reference_for_smoke_test=args.allow_track2p_as_reference_for_smoke_test,
        seed_session=args.seed_session,
        restrict_to_reference_seed_rois=args.restrict_to_reference_seed_rois,
        transform_type=args.transform_type,
        max_gap=TRACK2P_POLICY_DEFAULT_MAX_GAP,
        include_behavior=args.include_behavior,
        include_non_cells=False,
        cell_probability_threshold=args.cell_probability_threshold,
        weighted_masks=False,
        weighted_centroids=False,
        exclude_overlapping_pixels=False,
    )
    results = run_track2p_policy_benchmark(
        config,
        threshold_method=cast(ThresholdMethod, args.threshold_method),
        iou_distance_threshold=args.iou_distance_threshold,
        transform_type=args.transform_type,
        cell_probability_threshold=args.cell_probability_threshold,
    )
    rows = [result.to_dict() for result in results]
    if args.output is not None:
        write_results(rows, args.output, cast(OutputFormat, args.format))
    else:
        from bayescatrack.experiments.track2p_benchmark import _write_stdout

        _write_stdout(rows, cast(OutputFormat, args.format))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
