"""Reproducible Track2p benchmark harness for BayesCaTrack."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from bayescatrack.association.higher_order_consistency import (
    HigherOrderConsistencyConfig,
)
from bayescatrack.association.pyrecest_global_assignment import (
    AssociationCost,
    GlobalAssignmentRun,
    solve_global_assignment_for_sessions,
    tracks_to_suite2p_index_matrix,
)
from bayescatrack.core.bridge import (
    Track2pSession,
    find_track2p_session_dirs,
    load_track2p_subject,
)
from bayescatrack.evaluation.solver_rejection_ledger import (
    build_solver_rejection_ledger,
    write_solver_rejection_ledger_rows,
)
from bayescatrack.evaluation.track2p_metrics import (
    normalize_track_matrix,
    score_track_matrices,
)
from bayescatrack.ground_truth_eval import load_track2p_ground_truth_csv
from bayescatrack.matching import build_track_rows_from_matches
from bayescatrack.reference import (
    Track2pReference,
    load_aligned_subject_reference,
    load_track2p_reference,
)
from bayescatrack.track2p_registration import REGISTRATION_TRANSFORM_TYPES

ReferenceKind = Literal["auto", "manual-gt", "track2p-output", "aligned-subject-rows"]
BenchmarkMethod = Literal["track2p-baseline", "global-assignment", "oracle-gt-links"]
BenchmarkSplit = Literal["subject", "leave-one-subject-out"]
CalibrationModel = Literal["logistic", "monotone-ranker"]
OutputFormat = Literal["table", "json", "csv"]
GROUND_TRUTH_CSV_NAME = "ground_truth.csv"
GROUND_TRUTH_REFERENCE_SOURCE = "ground_truth_csv"
ALIGNED_REFERENCE_SOURCE = "aligned_subject_rows"
TRACK2P_REFERENCE_SOURCES = frozenset(
    {"track2p_output_suite2p_indices", "track2p_output_match_mat"}
)
GLOBAL_ASSIGNMENT_PRESETS: dict[BenchmarkPreset, dict[str, Any]] = {
    "none": {},
    # Keep the registered-IoU ablation, but use the solver priors that have
    # repeatedly been more appropriate for sparse longitudinal Track2p rows
    # than the conservative smoke-test defaults.
    "registered-iou-tuned": {
        "cost": "registered-iou",
        "max_gap": 2,
        "transform_type": "affine",
        "start_cost": 1.0,
        "end_cost": 1.0,
        "gap_penalty": 0.6,
        "cost_threshold": 2.0,
    },
    # Stronger BayesCaTrack row: use the ROI-aware cost family instead of the
    # IoU-only ablation, while retaining the same tuned global-assignment
    # priors. This makes the paper-facing comparison less dependent on a weak
    # default row without removing the ablation from the CLI.
    "roi-aware-tuned": {
        "cost": "roi-aware",
        "max_gap": 2,
        "transform_type": "affine",
        "start_cost": 1.0,
        "end_cost": 1.0,
        "gap_penalty": 0.6,
        "cost_threshold": 2.0,
    },
}


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class Track2pBenchmarkConfig:
    """Configuration for one Track2p benchmark run."""

    data: Path
    method: BenchmarkMethod
    split: BenchmarkSplit = "subject"
    benchmark_preset: BenchmarkPreset = "none"
    plane_name: str = "plane0"
    input_format: str = "auto"
    reference: Path | None = None
    reference_kind: ReferenceKind = "auto"
    allow_track2p_as_reference_for_smoke_test: bool = False
    curated_only: bool = False
    seed_session: int = 0
    restrict_to_reference_seed_rois: bool = True
    cost: AssociationCost = "registered-iou"
    max_gap: int = 2
    transform_type: str = "affine"
    registration_kwargs: dict[str, Any] | None = None
    start_cost: float = 5.0
    end_cost: float = 5.0
    gap_penalty: float = 1.0
    cost_threshold: float | None = 6.0
    include_behavior: bool = True
    include_non_cells: bool = False
    cell_probability_threshold: float = 0.5
    weighted_masks: bool = False
    exclude_overlapping_pixels: bool = True
    order: str = "xy"
    weighted_centroids: bool = False
    velocity_variance: float = 25.0
    regularization: float = 1.0e-6
    pairwise_cost_kwargs: dict[str, Any] | None = None
    higher_order_consistency_config: (
        HigherOrderConsistencyConfig | dict[str, Any] | None
    ) = None
    activity_tie_breaker_weight: float = 0.0
    activity_tie_breaker_component: str = "activity_tiebreaker_cost"
    activity_tie_breaker_neutral_cost: float = 0.5
    activity_tie_breaker_availability_component: str | None = "activity_tiebreaker_available"
    activity_tie_breaker_max_row_margin: float | None = None
    activity_tie_breaker_max_column_margin: float | None = None
    activity_trace_source: str = "auto"
    activity_event_threshold: float = 0.0
    load_neuropil_traces: bool = False
    progress: bool = False
    calibration_model: CalibrationModel = "logistic"
    monotone_ranker_kwargs: dict[str, Any] | None = None
    solver_ledger: bool = False
    solver_ledger_rank_k: int = 1
    solver_ledger_large_cost: float = 1.0e5
    solver_ledger_output: Path | None = None


@dataclass(frozen=True)
class SubjectBenchmarkResult:
    """One subject-level benchmark result."""

    subject: str
    variant: str
    method: BenchmarkMethod
    scores: Mapping[str, float | int | str]
    n_sessions: int
    reference_source: str

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "subject": self.subject,
            "variant": self.variant,
            "method": self.method,
            "n_sessions": self.n_sessions,
            "reference_source": self.reference_source,
            **dict(self.scores),
        }


class ProgressReporter:
    """Small stderr progress reporter that keeps machine-readable stdout clean."""

    def __init__(self, total: int, *, enabled: bool, label: str) -> None:
        self.total = max(int(total), 1)
        self.enabled = bool(enabled)
        self.label = label
        self.current = 0

    def step(self, message: str) -> None:
        if not self.enabled:
            return
        self.current = min(self.current + 1, self.total)
        filled = int(round(20 * self.current / self.total))
        bar = "#" * filled + "-" * (20 - filled)
        percent = 100.0 * self.current / self.total
        print(
            f"{self.label} [{bar}] {self.current}/{self.total} ({percent:5.1f}%) {message}",
            file=sys.stderr,
            flush=True,
        )


def apply_benchmark_preset(config: Track2pBenchmarkConfig) -> Track2pBenchmarkConfig:
    """Return ``config`` with a named global-assignment benchmark preset applied."""

    preset = config.benchmark_preset
    try:
        overrides = GLOBAL_ASSIGNMENT_PRESETS[preset]
    except KeyError as exc:
        raise ValueError(f"Unknown benchmark preset: {preset!r}") from exc

    if not overrides:
        return config
    if config.method != "global-assignment":
        raise ValueError(
            "--benchmark-preset is only valid with method='global-assignment'"
        )
    return replace(config, **overrides)


def run_track2p_benchmark(
    config: Track2pBenchmarkConfig,
) -> list[SubjectBenchmarkResult]:
    """Run a Track2p benchmark over one subject directory or a dataset root."""

    config = apply_benchmark_preset(config)

    if config.split == "leave-one-subject-out":
        if config.method != "global-assignment" or config.cost != "calibrated":
            raise ValueError(
                "LOSO calibration requires method='global-assignment' and cost='calibrated'"
            )
        if config.calibration_model == "logistic":
            from bayescatrack.experiments.track2p_loso_calibration import (
                run_track2p_loso_calibration,
            )

            return run_track2p_loso_calibration(config).to_benchmark_results()
        if config.calibration_model == "monotone-ranker":
            from bayescatrack.experiments.track2p_monotone_loso_calibration import (
                run_track2p_monotone_loso_calibration,
            )

            return run_track2p_monotone_loso_calibration(
                config,
                monotone_options=_monotone_ranker_options_from_config(config),
            ).to_benchmark_results()
        raise ValueError(f"Unsupported calibration_model: {config.calibration_model!r}")
    if config.calibration_model != "logistic":
        raise ValueError(
            "calibration_model='monotone-ranker' requires "
            "split='leave-one-subject-out' and cost='calibrated'"
        )
    if config.cost == "calibrated":
        raise ValueError("cost='calibrated' requires split='leave-one-subject-out'")

    subject_dirs = discover_subject_dirs(config.data)
    if not subject_dirs:
        raise ValueError(
            f"No Track2p-style subject directories found under {config.data}"
        )

    results: list[SubjectBenchmarkResult] = []
    solver_ledger_rows: list[dict[str, float | int | str]] = []
    progress = ProgressReporter(
        len(subject_dirs), enabled=config.progress, label="benchmark"
    )
    for subject_dir in subject_dirs:
        progress.step(f"running {subject_dir.name}")
        reference = _load_reference_for_subject(
            subject_dir, data_root=config.data, config=config
        )
        _validate_reference_for_benchmark(
            reference, subject_dir=subject_dir, config=config
        )
        if reference.source == GROUND_TRUTH_REFERENCE_SOURCE:
            _validate_reference_roi_indices(
                reference, _load_subject_sessions(subject_dir, config)
            )
        (
            predicted_matrix,
            variant,
            solver_ledger_summary,
            subject_ledger_rows,
        ) = _predict_subject_tracks(
            subject_dir, config, reference=reference
        )
        solver_ledger_rows.extend(subject_ledger_rows)
        scores = _score_prediction_against_reference(
            predicted_matrix, reference, config=config
        )
        scores = {**scores, **solver_ledger_summary}
        results.append(
            SubjectBenchmarkResult(
                subject=subject_dir.name,
                variant=variant,
                method=config.method,
                scores=scores,
                n_sessions=reference.n_sessions,
                reference_source=reference.source,
            )
        )
    if config.solver_ledger_output is not None:
        write_solver_rejection_ledger_rows(solver_ledger_rows, config.solver_ledger_output)
    return results


def _monotone_ranker_options_from_config(
    config: Track2pBenchmarkConfig,
) -> Any:
    """Build monotone ranker options from the main benchmark config."""

    from bayescatrack.association.monotone_ranker import MonotoneRankerOptions

    raw_kwargs = config.monotone_ranker_kwargs
    if raw_kwargs is None:
        kwargs: dict[str, Any] = {}
    elif isinstance(raw_kwargs, Mapping):
        kwargs = dict(raw_kwargs)
    else:
        raise ValueError("monotone_ranker_kwargs must be a JSON object/mapping")
    return MonotoneRankerOptions(**kwargs)


def discover_subject_dirs(data_path: str | Path) -> list[Path]:
    """Find Track2p subject directories beneath ``data_path``."""

    root = Path(data_path)
    if _looks_like_subject_dir(root):
        return [root]
    subjects = [
        child
        for child in sorted(root.iterdir())
        if child.is_dir() and _looks_like_subject_dir(child)
    ]
    return subjects


def format_benchmark_table(rows: Sequence[dict[str, float | int | str]]) -> str:
    """Format benchmark rows as the first paper-facing Markdown table."""

    columns = [
        "variant",
        "pairwise_f1",
        "complete_track_f1",
        "adjacent_link_recall",
        "reference_track_mean_best_session_recall",
        "reference_single_session_near_misses",
        "reference_track_mean_fragment_count",
        "pairwise_precision",
        "pairwise_recall",
        "complete_tracks",
        "mean_track_length",
    ]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] + ["---:"] * (len(columns) - 1)) + " |"
    body = [header, separator]
    for row in rows:
        values = [_format_table_value(row.get(column, "")) for column in columns]
        body.append("| " + " | ".join(values) + " |")
    return "\n".join(body)


def write_results(
    rows: Sequence[dict[str, float | int | str]],
    output_path: Path,
    output_format: OutputFormat,
) -> None:
    """Write benchmark rows as JSON, CSV, or Markdown table."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output_path.write_text(
            json.dumps(list(rows), indent=2) + "\n", encoding="utf-8"
        )
        return
    if output_format == "csv":
        fieldnames = _csv_fieldnames(rows)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return
    output_path.write_text(format_benchmark_table(rows) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p",
        description="Run Track2p baseline and global-assignment ablations on Track2p-style datasets.",
    )
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Track2p dataset root or one subject directory",
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=("track2p-baseline", "global-assignment", "oracle-gt-links"),
        help="Benchmark variant to run",
    )
    parser.add_argument(
        "--benchmark-preset",
        default="none",
        choices=tuple(GLOBAL_ASSIGNMENT_PRESETS),
        help="Optional global-assignment preset for paper-facing rows",
    )
    parser.add_argument(
        "--split",
        default="subject",
        choices=("subject", "leave-one-subject-out"),
        help="Evaluation split policy",
    )
    parser.add_argument(
        "--calibration-model",
        default="logistic",
        choices=("logistic", "monotone-ranker"),
        help="Model used for cost='calibrated' leave-one-subject-out runs",
    )
    parser.add_argument(
        "--plane", dest="plane_name", default="plane0", help="Plane name such as plane0"
    )
    parser.add_argument(
        "--input-format",
        default="auto",
        choices=("auto", "suite2p", "npy"),
        help="Input format for loading sessions",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Optional ground_truth.csv file, ground-truth root, subject directory, or track2p folder",
    )
    parser.add_argument(
        "--reference-kind",
        default="auto",
        choices=("auto", "manual-gt", "track2p-output", "aligned-subject-rows"),
        help="Declared reference type; manual-gt is required for paper-facing runs",
    )
    parser.add_argument(
        "--allow-track2p-as-reference-for-smoke-test",
        action="store_true",
        help="Permit Track2p/aligned-row references for plumbing smoke tests only",
    )
    parser.add_argument(
        "--curated-only",
        action="store_true",
        help="Evaluate only reference tracks marked curated",
    )
    parser.add_argument(
        "--seed-session",
        type=int,
        default=0,
        help="Reference seed session used for sparse-GT filtering",
    )
    parser.add_argument(
        "--restrict-to-reference-seed-rois",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Score only predicted tracks whose seed-session ROI is in the reference seed set",
    )
    parser.add_argument(
        "--cost",
        default="registered-iou",
        choices=(
            "registered-iou",
            "registered-soft-iou",
            "registered-shifted-iou",
            "roi-aware",
            "roi-aware-shifted",
            "calibrated",
        ),
        help="Pairwise cost used by global assignment",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=2,
        help="Maximum forward session gap for global-assignment edges",
    )
    parser.add_argument(
        "--transform-type",
        default="affine",
        choices=REGISTRATION_TRANSFORM_TYPES,
        help="Track2p registration transform type",
    )
    parser.add_argument(
        "--registration-kwargs-json",
        default=None,
        help="JSON object forwarded to the selected registration backend",
    )
    parser.add_argument(
        "--start-cost", type=float, default=5.0, help="PyRecEst track start cost"
    )
    parser.add_argument(
        "--end-cost", type=float, default=5.0, help="PyRecEst track end cost"
    )
    parser.add_argument(
        "--gap-penalty", type=float, default=1.0, help="Penalty per skipped session"
    )
    parser.add_argument(
        "--cost-threshold",
        type=float,
        default=6.0,
        help="Maximum adjusted edge cost admitted by the solver",
    )
    parser.add_argument(
        "--no-cost-threshold",
        action="store_true",
        help="Disable the solver edge-cost threshold",
    )
    parser.add_argument(
        "--include-behavior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load behaviour arrays when present",
    )
    parser.add_argument(
        "--include-non-cells",
        action="store_true",
        help="Keep Suite2p ROIs that fail iscell filtering",
    )
    parser.add_argument(
        "--cell-probability-threshold",
        type=float,
        default=0.5,
        help="Suite2p iscell probability threshold",
    )
    parser.add_argument(
        "--weighted-masks",
        action="store_true",
        help="Use Suite2p lam weights while reconstructing masks",
    )
    parser.add_argument(
        "--exclude-overlapping-pixels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop Suite2p overlap pixels when reconstructing masks",
    )
    parser.add_argument(
        "--order", default="xy", choices=("xy", "yx"), help="Coordinate order for costs"
    )
    parser.add_argument(
        "--weighted-centroids",
        action="store_true",
        help="Use weighted centroids where masks contain weights",
    )
    parser.add_argument(
        "--velocity-variance",
        type=float,
        default=25.0,
        help="Velocity variance for association bundle state moments",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=1.0e-6,
        help="Position covariance regularization",
    )
    parser.add_argument(
        "--pairwise-cost-kwargs-json",
        default=None,
        help="JSON object merged into pairwise cost kwargs",
    )
    parser.add_argument(
        "--higher-order-json",
        default=None,
        help=(
            "JSON object configuring triplet-projected higher-order consistency penalties, "
            'for example {"triplet_weight":0.25,"support_top_k":8,"support_cost_cap":4.0}'
        ),
    )
    parser.add_argument(
        "--monotone-ranker-kwargs-json",
        default=None,
        help="JSON object passed to MonotoneRankerOptions for monotone calibrated LOSO runs",
    )
    add_higher_order_consistency_arguments(parser)
    parser.add_argument(
        "--solver-ledger",
        action="store_true",
        help="Add cause-specific manual-GT edge rejection diagnostics to benchmark rows",
    )
    parser.add_argument(
        "--solver-ledger-rank-k",
        type=int,
        default=1,
        help="Rank cutoff used for top-k rejection causes in the solver ledger",
    )
    parser.add_argument(
        "--solver-ledger-large-cost",
        type=float,
        default=1.0e5,
        help="Cost floor treated as an empty-ROI/large-cost failure in the ledger",
    )
    parser.add_argument(
        "--solver-ledger-output",
        type=Path,
        default=None,
        help="Optional detailed per-GT-edge solver rejection CSV output path",
    )
    parser.add_argument(
        "--activity-tie-breaker-weight",
        type=float,
        default=0.0,
        help="Weak additive activity tie-breaker weight; 0 disables the adjustment",
    )
    parser.add_argument(
        "--activity-tie-breaker-component",
        default="activity_tiebreaker_cost",
        help="Pairwise activity component used for the weak additive tie-breaker",
    )
    parser.add_argument(
        "--activity-tie-breaker-neutral-cost",
        type=float,
        default=0.5,
        help="Activity component value treated as neutral before centering the tie-breaker",
    )
    parser.add_argument(
        "--activity-tie-breaker-availability-component",
        default="activity_tiebreaker_available",
        help="Availability component gating activity adjustments; use 'none' to disable gating",
    )
    parser.add_argument(
        "--activity-tie-breaker-max-row-margin",
        type=float,
        default=None,
        help="Only apply activity to candidates within this cost margin of the row best edge",
    )
    parser.add_argument(
        "--activity-tie-breaker-max-column-margin",
        type=float,
        default=None,
        help="Only apply activity to candidates within this cost margin of the column best edge",
    )
    parser.add_argument(
        "--activity-trace-source",
        default="auto",
        choices=("auto", "traces", "spike_traces", "neuropil_traces"),
        help="Trace source for legacy activity_similarity_cost and additive activity tie-breakers",
    )
    parser.add_argument(
        "--activity-event-threshold",
        type=float,
        default=0.0,
        help="Spike/event threshold used for event-rate activity features",
    )
    parser.add_argument(
        "--load-neuropil-traces",
        action="store_true",
        help="Load Suite2p Fneu.npy so neuropil-ratio activity features are available",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print benchmark progress to stderr",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Optional output file"
    )
    parser.add_argument(
        "--format",
        choices=("table", "json", "csv"),
        default="table",
        help="Stdout/output format",
    )
    return parser


def add_higher_order_consistency_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared CLI knobs for triplet-projected consistency penalties."""

    group = parser.add_argument_group("higher-order consistency")
    group.add_argument(
        "--higher-order-triplet-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for triplet-projected higher-order penalties. "
            "The default 0 disables the adjustment."
        ),
    )
    group.add_argument(
        "--higher-order-support-top-k",
        type=int,
        default=8,
        help="Number of third-session support candidates retained per shared ROI",
    )
    group.add_argument(
        "--higher-order-support-cost-cap",
        type=float,
        default=4.0,
        help="Maximum pairwise edge cost considered valid third-session support",
    )
    group.add_argument(
        "--higher-order-max-penalty",
        type=float,
        default=2.0,
        help="Maximum unweighted penalty added to an unsupported candidate edge",
    )
    group.add_argument(
        "--higher-order-large-cost",
        type=float,
        default=1.0e6,
        help="Sentinel cost above which candidate edges are treated as inadmissible",
    )


def higher_order_consistency_config_from_args(
    args: argparse.Namespace,
) -> HigherOrderConsistencyConfig | None:
    """Return a validated higher-order config from parsed CLI arguments."""

    config = HigherOrderConsistencyConfig(
        triplet_weight=float(args.higher_order_triplet_weight),
        support_top_k=int(args.higher_order_support_top_k),
        support_cost_cap=float(args.higher_order_support_cost_cap),
        max_penalty=float(args.higher_order_max_penalty),
        large_cost=float(args.higher_order_large_cost),
    )
    return config if config.enabled else None


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = _config_from_args(args)
    results = run_track2p_benchmark(config)
    rows = [result.to_dict() for result in results]

    if args.output is not None:
        write_results(rows, args.output, args.format)
    else:
        _write_stdout(rows, args.format)
    return 0


def _predict_subject_tracks(
    subject_dir: Path,
    config: Track2pBenchmarkConfig,
    *,
    reference: Track2pReference | None = None,
) -> tuple[np.ndarray, str, dict[str, float | int], list[dict[str, float | int | str]]]:
    if config.method == "track2p-baseline":
        track2p_dir = subject_dir / "track2p"
        if track2p_dir.exists():
            baseline = load_track2p_reference(track2p_dir, plane_name=config.plane_name)
        else:
            baseline = load_aligned_subject_reference(
                subject_dir,
                plane_name=config.plane_name,
                input_format=config.input_format,
                include_behavior=config.include_behavior,
                include_non_cells=config.include_non_cells,
                cell_probability_threshold=config.cell_probability_threshold,
                weighted_masks=config.weighted_masks,
                exclude_overlapping_pixels=config.exclude_overlapping_pixels,
            )
        return (
            normalize_track_matrix(baseline.suite2p_indices), "Track2p default", {}, []
        )

    if config.method == "oracle-gt-links":
        if reference is None:
            raise ValueError("oracle-gt-links requires a loaded reference")
        return (
            oracle_ground_truth_link_tracks(
                reference,
                curated_only=config.curated_only,
                seed_session=config.seed_session,
            ),
            "Oracle GT consecutive links",
            {},
            [],
        )

    if config.method != "global-assignment":
        raise ValueError(f"Unsupported benchmark method: {config.method!r}")

    sessions = _load_subject_sessions(subject_dir, config)
    assignment = solve_configured_global_assignment(sessions, config)
    predicted = tracks_to_suite2p_index_matrix(assignment.result.tracks, sessions)
    solver_ledger_summary: dict[str, float | int] = {}
    solver_ledger_rows: list[dict[str, float | int | str]] = []
    if config.solver_ledger:
        if reference is None:
            raise ValueError("solver ledger diagnostics require a loaded reference")
        ledger = build_solver_rejection_ledger(
            assignment,
            sessions,
            reference,
            subject=subject_dir.name,
            curated_only=config.curated_only,
            cost_threshold=config.cost_threshold,
            gap_penalty=config.gap_penalty,
            rank_k=config.solver_ledger_rank_k,
            large_cost=config.solver_ledger_large_cost,
        )
        solver_ledger_summary = ledger.summary
        solver_ledger_rows = list(ledger.rows)
    return (
        predicted,
        configured_variant_name(config),
        solver_ledger_summary,
        solver_ledger_rows,
    )


def oracle_ground_truth_link_tracks(
    reference: Track2pReference,
    *,
    curated_only: bool = False,
    seed_session: int = 0,
) -> np.ndarray:
    """Build an oracle prediction by stitching consecutive GT pairwise links.

    This diagnostic variant does not copy the reference track matrix directly.
    It converts each consecutive session pair into explicit GT ROI links and
    then uses the normal BayesCaTrack row-stitching helper. If complete-track F1
    is poor for this oracle, the failure is in track-row assembly, ROI indexing,
    or scoring rather than in registration or association costs.
    """

    reference_matrix = _reference_matrix(reference, curated_only=curated_only)
    start_roi_indices = sorted(
        _reference_seed_roi_set(reference_matrix, seed_session=seed_session)
    )

    if reference.n_sessions == 1:
        return np.asarray(start_roi_indices, dtype=int).reshape(-1, 1)

    consecutive_matches = [
        reference.pairwise_matches(
            session_index,
            session_index + 1,
            curated_only=curated_only,
        )
        for session_index in range(reference.n_sessions - 1)
    ]
    return build_track_rows_from_matches(
        reference.session_names,
        consecutive_matches,
        start_roi_indices=start_roi_indices,
        start_session_index=seed_session,
        fill_value=-1,
    )


def solve_configured_global_assignment(
    sessions: Sequence[Track2pSession],
    config: Track2pBenchmarkConfig,
    *,
    cost: AssociationCost | None = None,
    calibrated_model: Any | None = None,
) -> GlobalAssignmentRun:
    """Run global assignment using the benchmark configuration knobs."""

    return solve_global_assignment_for_sessions(
        sessions,
        max_gap=config.max_gap,
        cost=config.cost if cost is None else cost,
        calibrated_model=calibrated_model,
        transform_type=config.transform_type,
        registration_kwargs=config.registration_kwargs,
        start_cost=config.start_cost,
        end_cost=config.end_cost,
        gap_penalty=config.gap_penalty,
        cost_threshold=config.cost_threshold,
        order=config.order,
        weighted_centroids=config.weighted_centroids,
        velocity_variance=config.velocity_variance,
        regularization=config.regularization,
        pairwise_cost_kwargs=config.pairwise_cost_kwargs,
        activity_tie_breaker_weight=config.activity_tie_breaker_weight,
        activity_tie_breaker_component=config.activity_tie_breaker_component,
        activity_tie_breaker_neutral_cost=config.activity_tie_breaker_neutral_cost,
        activity_tie_breaker_availability_component=config.activity_tie_breaker_availability_component,
        activity_tie_breaker_max_row_margin=config.activity_tie_breaker_max_row_margin,
        activity_tie_breaker_max_column_margin=config.activity_tie_breaker_max_column_margin,
        activity_trace_source=config.activity_trace_source,
        activity_event_threshold=config.activity_event_threshold,
        higher_order_consistency_config=config.higher_order_consistency_config,
    )


def _variant_name(
    cost: AssociationCost, *, activity_tie_breaker_weight: float = 0.0
) -> str:
    if cost == "registered-iou":
        variant = "Same costs + global assignment"
    elif cost == "registered-soft-iou":
        variant = "Soft-IoU costs + global assignment"
    elif cost == "registered-shifted-iou":
        variant = "Shifted-IoU costs + global assignment"
    elif cost == "roi-aware-shifted":
        variant = "Shifted ROI-aware costs + global assignment"
    elif cost == "calibrated":
        variant = "Calibrated costs + global assignment"
    else:
        variant = "BayesCaTrack costs + global assignment"
    if activity_tie_breaker_weight > 0.0:
        variant += f" + activity tie-breaker (w={activity_tie_breaker_weight:g})"
    return variant


def configured_variant_name(config: Track2pBenchmarkConfig) -> str:
    """Return the paper-facing variant label for a full benchmark config."""

    variant = _variant_name(
        config.cost, activity_tie_breaker_weight=config.activity_tie_breaker_weight
    )
    if higher_order_consistency_enabled(config.higher_order_consistency_config):
        variant += " + triplet consistency"
    return variant


def higher_order_consistency_enabled(
    config: HigherOrderConsistencyConfig | Mapping[str, Any] | None,
) -> bool:
    """Return whether a higher-order consistency config changes pairwise costs."""

    resolved = _coerce_higher_order_consistency_config(config)
    return bool(resolved is not None and resolved.enabled)


def higher_order_consistency_score_fields(
    config: Track2pBenchmarkConfig,
) -> dict[str, float | int]:
    """Return reproducibility fields for enabled higher-order consistency."""

    resolved = _coerce_higher_order_consistency_config(
        config.higher_order_consistency_config
    )
    if resolved is None or not resolved.enabled:
        return {}
    return {
        "higher_order_triplet_weight": float(resolved.triplet_weight),
        "higher_order_support_top_k": int(resolved.support_top_k),
        "higher_order_support_cost_cap": float(resolved.support_cost_cap),
        "higher_order_max_penalty": float(resolved.max_penalty),
        "higher_order_large_cost": float(resolved.large_cost),
    }


def _coerce_higher_order_consistency_config(
    config: HigherOrderConsistencyConfig | Mapping[str, Any] | None,
) -> HigherOrderConsistencyConfig | None:
    if config is None:
        return None
    if isinstance(config, HigherOrderConsistencyConfig):
        return config
    return HigherOrderConsistencyConfig(**dict(config))


# pylint: disable=too-many-return-statements,too-many-branches
def _load_reference_for_subject(
    subject_dir: Path, *, data_root: Path, config: Track2pBenchmarkConfig
) -> Track2pReference:
    data_root = Path(data_root)
    if config.reference is None:
        if config.reference_kind == "manual-gt":
            default_ground_truth_path = subject_dir / GROUND_TRUTH_CSV_NAME
            if default_ground_truth_path.exists():
                return _load_ground_truth_csv_reference(
                    default_ground_truth_path, subject_dir=subject_dir
                )
            raise ValueError(
                f"--reference-kind manual-gt was requested, but {default_ground_truth_path} does not exist"
            )
        if config.reference_kind == "track2p-output":
            track2p_dir = subject_dir / "track2p"
            if track2p_dir.exists():
                return load_track2p_reference(track2p_dir, plane_name=config.plane_name)
            raise ValueError(
                f"--reference-kind track2p-output was requested, but {track2p_dir} does not exist"
            )
        if config.reference_kind == "aligned-subject-rows":
            return _load_aligned_reference_for_config(subject_dir, config)

        default_ground_truth_path = subject_dir / GROUND_TRUTH_CSV_NAME
        if default_ground_truth_path.exists():
            return _load_ground_truth_csv_reference(
                default_ground_truth_path, subject_dir=subject_dir
            )
        track2p_dir = subject_dir / "track2p"
        if track2p_dir.exists():
            return load_track2p_reference(track2p_dir, plane_name=config.plane_name)
        return _load_aligned_reference_for_config(subject_dir, config)

    reference_root = Path(config.reference)
    if config.reference_kind == "manual-gt":
        if reference_root.is_file():
            return _load_ground_truth_csv_reference(
                reference_root, subject_dir=subject_dir
            )
        ground_truth_path = _resolve_ground_truth_csv_path(
            subject_dir, data_root=data_root, reference_root=reference_root
        )
        if ground_truth_path is None:
            raise ValueError(
                "--reference-kind manual-gt was requested, but no ground_truth.csv could be resolved "
                f"for subject {subject_dir.name!r} under {reference_root}"
            )
        return _load_ground_truth_csv_reference(
            ground_truth_path, subject_dir=subject_dir
        )

    if config.reference_kind == "track2p-output":
        reference_path = _resolve_track2p_reference_path(
            subject_dir, data_root=data_root, reference_root=reference_root
        )
        if reference_path is None:
            raise ValueError(
                "--reference-kind track2p-output was requested, but no Track2p reference could be resolved "
                f"for subject {subject_dir.name!r} under {reference_root}"
            )
        return load_track2p_reference(reference_path, plane_name=config.plane_name)

    if config.reference_kind == "aligned-subject-rows":
        return _load_aligned_reference_for_config(subject_dir, config)

    ground_truth_path = _resolve_ground_truth_csv_path(
        subject_dir, data_root=data_root, reference_root=reference_root
    )
    if ground_truth_path is not None:
        return _load_ground_truth_csv_reference(
            ground_truth_path, subject_dir=subject_dir
        )

    reference_path = _resolve_track2p_reference_path(
        subject_dir, data_root=data_root, reference_root=reference_root
    )
    if reference_path is not None:
        return load_track2p_reference(reference_path, plane_name=config.plane_name)
    return _load_aligned_reference_for_config(subject_dir, config)


def _validate_reference_for_benchmark(
    reference: Track2pReference,
    *,
    subject_dir: Path,
    config: Track2pBenchmarkConfig,
) -> None:
    if reference.source == GROUND_TRUTH_REFERENCE_SOURCE:
        return
    if config.allow_track2p_as_reference_for_smoke_test:
        return
    raise ValueError(
        f"Subject {subject_dir.name!r} resolved reference source {reference.source!r}, "
        "which is not independent manual ground truth. Provide --reference pointing at "
        "ground_truth.csv or a ground-truth root with --reference-kind manual-gt. "
        "For plumbing checks only, pass --allow-track2p-as-reference-for-smoke-test."
    )


def _resolve_ground_truth_csv_path(
    subject_dir: Path, *, data_root: Path, reference_root: Path
) -> Path | None:
    candidates: list[Path] = []
    if reference_root.is_file():
        candidates.append(reference_root)
    else:
        candidates.extend(
            [
                reference_root / GROUND_TRUTH_CSV_NAME,
                reference_root / subject_dir.name / GROUND_TRUTH_CSV_NAME,
            ]
        )
        relative_subject: Path | None
        try:
            relative_subject = subject_dir.relative_to(data_root)
        except ValueError:
            relative_subject = None
        if relative_subject is not None:
            candidates.append(reference_root / relative_subject / GROUND_TRUTH_CSV_NAME)

    for candidate in candidates:
        if (
            candidate.exists()
            and candidate.is_file()
            and candidate.name.casefold() == GROUND_TRUTH_CSV_NAME
        ):
            return candidate
    return None


def _resolve_track2p_reference_path(
    subject_dir: Path, *, data_root: Path, reference_root: Path
) -> Path | None:
    candidates = [
        reference_root,
        reference_root / subject_dir.name,
        reference_root / subject_dir.name / "track2p",
        reference_root / "track2p",
    ]
    relative_subject: Path | None
    try:
        relative_subject = subject_dir.relative_to(data_root)
    except ValueError:
        relative_subject = None
    if relative_subject is not None:
        candidates.extend(
            [
                reference_root / relative_subject,
                reference_root / relative_subject / "track2p",
            ]
        )
    for candidate in candidates:
        if (candidate / "track_ops.npy").exists() or (
            candidate / "track2p" / "track_ops.npy"
        ).exists():
            return candidate
    return None


def _load_ground_truth_csv_reference(
    ground_truth_path: Path, *, subject_dir: Path
) -> Track2pReference:
    session_names = tuple(
        session_dir.name for session_dir in find_track2p_session_dirs(subject_dir)
    )
    if not session_names:
        raise ValueError(
            f"No Track2p-style sessions were found for ground-truth reference {ground_truth_path}"
        )

    track_table = load_track2p_ground_truth_csv(ground_truth_path)
    if track_table.session_names != session_names:
        if set(track_table.session_names) != set(session_names):
            raise ValueError(
                f"{ground_truth_path} session columns {track_table.session_names!r} do not match subject sessions {session_names!r}"
            )
        track_table = track_table.aligned_to(session_names)

    return Track2pReference(
        session_names=track_table.session_names,
        suite2p_indices=track_table.tracks,
        curated_mask=np.ones((track_table.n_tracks,), dtype=bool),
        source=GROUND_TRUTH_REFERENCE_SOURCE,
    )


def _score_prediction_against_reference(
    predicted_matrix: np.ndarray,
    reference: Track2pReference,
    *,
    config: Track2pBenchmarkConfig,
) -> dict[str, float | int]:
    reference_matrix = _reference_matrix(reference, curated_only=config.curated_only)
    predicted = normalize_track_matrix(predicted_matrix)
    if predicted.shape[1] != reference_matrix.shape[1]:
        raise ValueError(
            "Predicted and reference matrices must have the same number of sessions"
        )

    predicted_before_filter = int(predicted.shape[0])
    reference_seed_rois: set[int] = set()
    if config.restrict_to_reference_seed_rois:
        reference_seed_rois = _reference_seed_roi_set(
            reference_matrix, seed_session=config.seed_session
        )
        predicted = _filter_tracks_by_seed_rois(
            predicted,
            reference_seed_rois,
            seed_session=config.seed_session,
        )
        reference_matrix = _filter_tracks_by_seed_rois(
            reference_matrix,
            reference_seed_rois,
            seed_session=config.seed_session,
        )

    scores = _with_track_damage_diagnostics(
        _with_recomputed_f1_scores(
            score_track_matrices(predicted, reference_matrix)
        ),
        predicted=predicted,
        reference_matrix=reference_matrix,
    )
    if config.restrict_to_reference_seed_rois:
        scores = {
            **scores,
            "seed_session": int(config.seed_session),
            "reference_seed_rois": int(len(reference_seed_rois)),
            "evaluated_prediction_tracks": int(predicted.shape[0]),
            "dropped_prediction_tracks": int(
                predicted_before_filter - predicted.shape[0]
            ),
        }
    higher_order_fields = higher_order_consistency_score_fields(config)
    if higher_order_fields:
        scores = {**scores, **higher_order_fields}
    return scores


def _with_recomputed_f1_scores(
    scores: Mapping[str, float | int],
) -> dict[str, float | int]:
    repaired_scores = dict(scores)
    for prefix in ("pairwise", "complete_track"):
        tp = int(repaired_scores.get(f"{prefix}_true_positives", 0))
        fp = int(repaired_scores.get(f"{prefix}_false_positives", 0))
        fn = int(repaired_scores.get(f"{prefix}_false_negatives", 0))
        repaired_scores[f"{prefix}_f1"] = _f1_from_counts(tp, fp, fn)
    return repaired_scores


def _f1_from_counts(
    true_positives: int, false_positives: int, false_negatives: int
) -> float:
    denominator = 2 * true_positives + false_positives + false_negatives
    if denominator == 0:
        return 1.0
    return float(2 * true_positives / denominator)


def _with_track_damage_diagnostics(
    scores: Mapping[str, float | int],
    *,
    predicted: np.ndarray,
    reference_matrix: np.ndarray,
) -> dict[str, float | int]:
    """Add diagnostics that expose complete-track metric sensitivity.

    Complete-track F1 is intentionally strict: one missing session in an
    otherwise correct longitudinal identity turns the whole row into a false
    negative.  These auxiliary metrics keep that strict headline number while
    revealing whether failures are isolated near misses, fragmentation, or
    broadly missing adjacent links.
    """

    return {
        **dict(scores),
        **_track_damage_diagnostics(predicted, reference_matrix),
    }


def _track_damage_diagnostics(
    predicted_matrix: np.ndarray,
    reference_matrix: np.ndarray,
) -> dict[str, float | int]:
    predicted = normalize_track_matrix(predicted_matrix)
    reference = normalize_track_matrix(reference_matrix)

    reference_tracks = 0
    fully_covered_reference_tracks = 0
    reference_tracks_with_any_match = 0
    single_session_near_misses = 0
    best_session_recalls: list[float] = []
    missing_session_counts: list[int] = []
    fragment_counts: list[int] = []

    for reference_row in reference:
        valid_positions = _valid_track_positions(reference_row)
        n_valid_sessions = len(valid_positions)
        if n_valid_sessions == 0:
            continue

        reference_tracks += 1
        overlaps = [
            _track_overlap_count(predicted_row, valid_positions)
            for predicted_row in predicted
        ]
        best_overlap = max(overlaps, default=0)
        missing_sessions = n_valid_sessions - best_overlap
        fragment_count = sum(1 for overlap in overlaps if overlap > 0)

        best_session_recalls.append(best_overlap / n_valid_sessions)
        missing_session_counts.append(missing_sessions)
        fragment_counts.append(fragment_count)

        if best_overlap == n_valid_sessions:
            fully_covered_reference_tracks += 1
        if best_overlap > 0:
            reference_tracks_with_any_match += 1
        if missing_sessions == 1:
            single_session_near_misses += 1

    adjacent_reference_links = _adjacent_link_set(reference)
    adjacent_predicted_links = _adjacent_link_set(predicted)
    adjacent_true_positives = len(adjacent_predicted_links & adjacent_reference_links)
    adjacent_false_positives = len(adjacent_predicted_links - adjacent_reference_links)
    adjacent_false_negatives = len(adjacent_reference_links - adjacent_predicted_links)

    return {
        "adjacent_link_true_positives": int(adjacent_true_positives),
        "adjacent_link_false_positives": int(adjacent_false_positives),
        "adjacent_link_false_negatives": int(adjacent_false_negatives),
        "adjacent_link_precision": _precision_from_counts(
            adjacent_true_positives, adjacent_false_positives
        ),
        "adjacent_link_recall": _recall_from_counts(
            adjacent_true_positives, adjacent_false_negatives
        ),
        "adjacent_link_f1": _f1_from_counts(
            adjacent_true_positives,
            adjacent_false_positives,
            adjacent_false_negatives,
        ),
        "reference_tracks": int(reference_tracks),
        "reference_tracks_fully_covered": int(fully_covered_reference_tracks),
        "reference_tracks_with_any_match": int(reference_tracks_with_any_match),
        "reference_single_session_near_misses": int(single_session_near_misses),
        "reference_near_miss_fraction": _safe_fraction(
            single_session_near_misses, reference_tracks
        ),
        "reference_track_mean_best_session_recall": _safe_mean(
            best_session_recalls
        ),
        "reference_track_mean_missing_sessions": _safe_mean(
            missing_session_counts
        ),
        "reference_track_mean_fragment_count": _safe_mean(fragment_counts),
        "reference_track_max_fragment_count": int(
            max(fragment_counts, default=0)
        ),
    }


def _valid_track_positions(row: np.ndarray) -> dict[int, int]:
    return {
        session_index: int(cast(Any, roi_index))
        for session_index, roi_index in enumerate(row)
        if _is_valid_roi_index(roi_index)
    }


def _track_overlap_count(
    predicted_row: np.ndarray, reference_positions: Mapping[int, int]
) -> int:
    matches = 0
    for session_index, reference_roi_index in reference_positions.items():
        predicted_roi_index = predicted_row[session_index]
        if not _is_valid_roi_index(predicted_roi_index):
            continue
        if int(cast(Any, predicted_roi_index)) == reference_roi_index:
            matches += 1
    return matches


def _adjacent_link_set(matrix: np.ndarray) -> set[tuple[int, int, int]]:
    links: set[tuple[int, int, int]] = set()
    for row in matrix:
        for session_index in range(row.shape[0] - 1):
            left = row[session_index]
            right = row[session_index + 1]
            if _is_valid_roi_index(left) and _is_valid_roi_index(right):
                links.add(
                    (
                        session_index,
                        int(cast(Any, left)),
                        int(cast(Any, right)),
                    )
                )
    return links


def _precision_from_counts(true_positives: int, false_positives: int) -> float:
    denominator = true_positives + false_positives
    if denominator == 0:
        return 1.0
    return float(true_positives / denominator)


def _recall_from_counts(true_positives: int, false_negatives: int) -> float:
    denominator = true_positives + false_negatives
    if denominator == 0:
        return 1.0
    return float(true_positives / denominator)


def _safe_fraction(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _safe_mean(values: Sequence[float | int]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def _filter_tracks_by_seed_rois(
    predicted_matrix: np.ndarray,
    seed_rois: set[int],
    *,
    seed_session: int,
) -> np.ndarray:
    if not seed_rois:
        return predicted_matrix[:0]
    keep = [
        _is_valid_roi_index(row[seed_session]) and int(row[seed_session]) in seed_rois
        for row in predicted_matrix
    ]
    return predicted_matrix[np.asarray(keep, dtype=bool)]


def _reference_seed_roi_set(
    reference_matrix: np.ndarray, *, seed_session: int
) -> set[int]:
    if seed_session < 0 or seed_session >= reference_matrix.shape[1]:
        raise IndexError(
            f"seed_session {seed_session} out of bounds for {reference_matrix.shape[1]} sessions"
        )
    return {
        int(cast(Any, value))
        for value in reference_matrix[:, seed_session]
        if _is_valid_roi_index(value)
    }


def _is_valid_roi_index(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return False
    try:
        return int(cast(Any, value)) >= 0
    except (TypeError, ValueError):
        return False


def _load_aligned_reference_for_config(
    subject_dir: Path, config: Track2pBenchmarkConfig
) -> Track2pReference:
    return load_aligned_subject_reference(
        subject_dir,
        plane_name=config.plane_name,
        input_format=config.input_format,
        include_behavior=config.include_behavior,
        include_non_cells=config.include_non_cells,
        cell_probability_threshold=config.cell_probability_threshold,
        weighted_masks=config.weighted_masks,
        exclude_overlapping_pixels=config.exclude_overlapping_pixels,
        load_neuropil_traces=config.load_neuropil_traces,
    )


def _load_subject_sessions(
    subject_dir: Path, config: Track2pBenchmarkConfig
) -> list[Track2pSession]:
    return load_track2p_subject(
        subject_dir,
        plane_name=config.plane_name,
        input_format=config.input_format,
        include_behavior=config.include_behavior,
        include_non_cells=config.include_non_cells,
        cell_probability_threshold=config.cell_probability_threshold,
        weighted_masks=config.weighted_masks,
        exclude_overlapping_pixels=config.exclude_overlapping_pixels,
        load_neuropil_traces=config.load_neuropil_traces,
    )


def _validate_reference_roi_indices(
    reference: Track2pReference, sessions: Sequence[Track2pSession]
) -> None:
    sessions = tuple(sessions)
    if len(sessions) != reference.n_sessions:
        raise ValueError(
            f"Reference {reference.source!r} has {reference.n_sessions} sessions, but the loaded subject has {len(sessions)} sessions"
        )

    session_names = tuple(session.session_name for session in sessions)
    if session_names != reference.session_names:
        raise ValueError(
            f"Reference {reference.source!r} session order {reference.session_names!r} does not match loaded sessions {session_names!r}"
        )

    for session_index, session in enumerate(sessions):
        available_indices = _loaded_suite2p_index_set(session)
        referenced_indices = {
            int(value)
            for value in reference.suite2p_indices[:, session_index]
            if _is_valid_roi_index(value)
        }
        missing_indices = sorted(referenced_indices - available_indices)
        if missing_indices:
            raise ValueError(
                _format_reference_roi_index_error(
                    session=session,
                    referenced_indices=referenced_indices,
                    available_indices=available_indices,
                    missing_indices=missing_indices,
                )
            )


def _loaded_suite2p_index_set(session: Track2pSession) -> set[int]:
    roi_indices = session.plane_data.roi_indices
    if roi_indices is None:
        return set(range(session.plane_data.n_rois))
    return {int(value) for value in np.asarray(roi_indices, dtype=int).reshape(-1)}


def _format_reference_roi_index_error(
    *,
    session: Track2pSession,
    referenced_indices: set[int],
    available_indices: set[int],
    missing_indices: Sequence[int],
) -> str:
    preview = ", ".join(str(value) for value in missing_indices[:10])
    suffix = "" if len(missing_indices) <= 10 else f", ... ({len(missing_indices)} total)"
    return (
        "Reference ROI indices are absent from loaded session "
        f"{session.session_name!r}: {preview}{suffix}. "
        f"Loaded ROI index range: {_format_int_range(available_indices)} "
        f"({len(available_indices)} ROIs); reference ROI index range: "
        f"{_format_int_range(referenced_indices)} ({len(referenced_indices)} ROIs). "
        f"{_reference_roi_mismatch_hint(referenced_indices, available_indices, missing_indices)} "
        "Run `bayescatrack benchmark audit-manual-gt-rois --reference-kind manual-gt "
        "--include-non-cells` to distinguish filtered ROIs from a reduced/reindexed "
        "public dataset before interpreting benchmark scores."
    )


def _format_int_range(values: set[int]) -> str:
    if not values:
        return "empty"
    return f"[{min(values)}, {max(values)}]"


def _reference_roi_mismatch_hint(
    referenced_indices: set[int],
    available_indices: set[int],
    missing_indices: Sequence[int],
) -> str:
    if not available_indices:
        return (
            "No Suite2p ROI indices were loaded for this session; check the input format, "
            "plane name, and Suite2p files."
        )
    if not referenced_indices:
        return "The manual-GT column does not contain any non-negative ROI IDs."

    available_max = max(available_indices)
    referenced_max = max(referenced_indices)
    if referenced_max > available_max:
        return (
            "The manual-GT reference appears to use a larger Suite2p/stat.npy row space "
            "than the loaded data. This is the public-subset/reindexing failure mode: "
            "--include-non-cells cannot recover ROI IDs that are outside the loaded range. "
            "Obtain the full pre-Track2p Suite2p outputs or an explicit ROI-ID remapping."
        )

    missing = tuple(int(value) for value in missing_indices)
    if missing and all(value - 1 in available_indices for value in missing):
        return "The missing IDs are all present after subtracting one; check for one-based ROI numbering."
    if missing and all(value + 1 in available_indices for value in missing):
        return "The missing IDs are all present after adding one; check for an off-by-one ROI remapping."
    return (
        "The reference probably uses raw Suite2p stat.npy row indices while the benchmark "
        "loaded a filtered ROI set. Re-run with --include-non-cells or lower "
        "--cell-probability-threshold if this filtering was unintentional."
    )


def _reference_matrix(reference: Track2pReference, *, curated_only: bool) -> np.ndarray:
    matrix = normalize_track_matrix(reference.suite2p_indices)
    if not curated_only:
        return matrix
    if reference.curated_mask is None:
        raise ValueError(
            "--curated-only was requested, but the reference has no curation mask"
        )
    return matrix[np.asarray(reference.curated_mask, dtype=bool)]


def _looks_like_subject_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    if (path / "track2p").exists():
        return True
    return bool(find_track2p_session_dirs(path))


def _json_object_from_arg(raw: str | None, option_name: str) -> dict[str, Any] | None:
    if raw is None:
        return None
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"{option_name} must decode to a JSON object")
    return parsed


def _config_from_args(args: argparse.Namespace) -> Track2pBenchmarkConfig:
    pairwise_cost_kwargs = _json_object_from_arg(
        args.pairwise_cost_kwargs_json, "--pairwise-cost-kwargs-json"
    )
    registration_kwargs = _json_object_from_arg(
        args.registration_kwargs_json, "--registration-kwargs-json"
    )
    higher_order_consistency_config = _json_object_from_arg(
        args.higher_order_json, "--higher-order-json"
    )
    if higher_order_consistency_config is None:
        higher_order_consistency_config = higher_order_consistency_config_from_args(args)
    monotone_ranker_kwargs = _json_object_from_arg(
        args.monotone_ranker_kwargs_json, "--monotone-ranker-kwargs-json"
    )
    activity_tie_breaker_availability_component = _optional_component_name_from_arg(
        args.activity_tie_breaker_availability_component
    )
    return Track2pBenchmarkConfig(
        data=args.data,
        method=args.method,
        split=args.split,
        benchmark_preset=args.benchmark_preset,
        plane_name=args.plane_name,
        input_format=args.input_format,
        reference=args.reference,
        reference_kind=args.reference_kind,
        allow_track2p_as_reference_for_smoke_test=args.allow_track2p_as_reference_for_smoke_test,
        curated_only=args.curated_only,
        seed_session=args.seed_session,
        restrict_to_reference_seed_rois=args.restrict_to_reference_seed_rois,
        cost=args.cost,
        calibration_model=args.calibration_model,
        max_gap=args.max_gap,
        transform_type=args.transform_type,
        registration_kwargs=registration_kwargs,
        start_cost=args.start_cost,
        end_cost=args.end_cost,
        gap_penalty=args.gap_penalty,
        cost_threshold=None if args.no_cost_threshold else args.cost_threshold,
        include_behavior=args.include_behavior,
        include_non_cells=args.include_non_cells,
        cell_probability_threshold=args.cell_probability_threshold,
        weighted_masks=args.weighted_masks,
        exclude_overlapping_pixels=args.exclude_overlapping_pixels,
        order=args.order,
        weighted_centroids=args.weighted_centroids,
        velocity_variance=args.velocity_variance,
        regularization=args.regularization,
        pairwise_cost_kwargs=pairwise_cost_kwargs,
        higher_order_consistency_config=higher_order_consistency_config,
        activity_tie_breaker_weight=args.activity_tie_breaker_weight,
        activity_tie_breaker_component=args.activity_tie_breaker_component,
        activity_tie_breaker_neutral_cost=args.activity_tie_breaker_neutral_cost,
        activity_tie_breaker_availability_component=activity_tie_breaker_availability_component,
        activity_tie_breaker_max_row_margin=args.activity_tie_breaker_max_row_margin,
        activity_tie_breaker_max_column_margin=args.activity_tie_breaker_max_column_margin,
        activity_trace_source=args.activity_trace_source,
        activity_event_threshold=args.activity_event_threshold,
        load_neuropil_traces=args.load_neuropil_traces,
        progress=args.progress,
        monotone_ranker_kwargs=monotone_ranker_kwargs,
        solver_ledger=bool(args.solver_ledger or args.solver_ledger_output is not None),
        solver_ledger_rank_k=args.solver_ledger_rank_k,
        solver_ledger_large_cost=args.solver_ledger_large_cost,
        solver_ledger_output=args.solver_ledger_output,
    )


def _optional_component_name_from_arg(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if stripped.lower() in {"", "none", "null"}:
        return None
    return stripped


def _write_stdout(
    rows: Sequence[dict[str, float | int | str]], output_format: OutputFormat
) -> None:
    if output_format == "json":
        print(json.dumps(list(rows), indent=2))
        return
    if output_format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=_csv_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)
        return
    print(format_benchmark_table(rows))


def _csv_fieldnames(rows: Sequence[dict[str, float | int | str]]) -> list[str]:
    preferred = [
        "subject",
        "variant",
        "method",
        "n_sessions",
        "reference_source",
        "pairwise_f1",
        "complete_track_f1",
        "adjacent_link_f1",
        "adjacent_link_precision",
        "adjacent_link_recall",
        "adjacent_link_true_positives",
        "adjacent_link_false_positives",
        "adjacent_link_false_negatives",
        "reference_tracks",
        "reference_tracks_fully_covered",
        "reference_tracks_with_any_match",
        "reference_single_session_near_misses",
        "reference_near_miss_fraction",
        "reference_track_mean_best_session_recall",
        "reference_track_mean_missing_sessions",
        "reference_track_mean_fragment_count",
        "reference_track_max_fragment_count",
        "pairwise_precision",
        "pairwise_recall",
        "complete_tracks",
        "mean_track_length",
        "seed_session",
        "reference_seed_rois",
        "evaluated_prediction_tracks",
        "dropped_prediction_tracks",
        "training_examples",
        "positive_examples",
        "negative_examples",
        "higher_order_triplet_weight",
        "higher_order_support_top_k",
        "higher_order_support_cost_cap",
        "higher_order_max_penalty",
        "higher_order_large_cost",
        "calibration_model",
        "monotone_rank_constraints",
        "monotone_training_rank_loss",
        "monotone_training_binary_loss",
        "solver_ledger_gt_edges",
        "solver_ledger_selected_edges",
        "solver_ledger_rejected_edges",
        "solver_ledger_selected_rate",
        "solver_ledger_rejected_rate",
        "solver_ledger_true_edge_gated_by_cost_threshold",
        "solver_ledger_true_edge_not_row_top_k",
        "solver_ledger_true_edge_not_column_top_k",
        "solver_ledger_wrong_edge_selected",
        "solver_ledger_mutual_top1_rejected_by_solver_prior",
        "solver_ledger_true_edge_large_cost_or_empty_registered_roi",
        "solver_ledger_reference_roi_missing_from_loaded_session",
        "solver_ledger_measurement_roi_missing_from_loaded_session",
    ]
    remaining = sorted({key for row in rows for key in row} - set(preferred))
    return [key for key in preferred if any(key in row for row in rows)] + remaining


def _format_table_value(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    return str(value)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
