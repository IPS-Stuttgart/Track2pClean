"""Compact Track2p result-failure triage for BayesCaTrack benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from bayescatrack.association.calibrated_costs import (
    ReferenceTrainingOptions,
    collect_reference_pairwise_example_blocks,
)
from bayescatrack.association.pyrecest_global_assignment import (
    AssociationCost,
    session_edge_pairs,
    tracks_to_suite2p_index_matrix,
)
from bayescatrack.evaluation.edge_ranking import (
    ScoreDirection,
    missing_reference_edge_rows,
    rank_labeled_edges,
    score_matrices_from_feature_tensor,
)
from bayescatrack.experiments.track2p_benchmark import (
    ProgressReporter,
    Track2pBenchmarkConfig,
    _load_reference_for_subject,
    _load_subject_sessions,
    _score_prediction_against_reference,
    _validate_reference_for_benchmark,
    _validate_reference_roi_indices,
    discover_subject_dirs,
    oracle_ground_truth_link_tracks,
    solve_configured_global_assignment,
)
from bayescatrack.experiments.track2p_edge_ranking import (
    DEFAULT_SIMILARITY_FEATURES,
    _pairwise_cost_kwargs_for_config,
)

FailureMode = Literal[
    "row-assembly-or-scoring",
    "registration-or-cost-ranking",
    "solver-priors-or-track-stitching",
    "association-ranking",
    "no-dominant-failure",
]
OutputFormat = Literal["table", "json", "csv"]


@dataclass(frozen=True)
class DiagnosisThresholds:
    """Decision thresholds for the compact benchmark triage."""

    min_oracle_complete_f1: float = 0.98
    min_edge_mutual_top1_rate: float = 0.70
    min_pairwise_f1_for_solver_triage: float = 0.70
    min_pairwise_complete_f1_gap: float = 0.15
    max_edge_missing_rate: float = 0.05


@dataclass(frozen=True)
class SubjectFailureDiagnosis:
    """One subject-level result triage row."""

    subject: str
    failure_mode: FailureMode
    recommendation: str
    method_pairwise_f1: float
    method_complete_track_f1: float
    oracle_pairwise_f1: float
    oracle_complete_track_f1: float
    pairwise_complete_f1_gap: float
    edge_score_name: str
    edge_gt_edges: int
    edge_present_edges: int
    edge_missing_edges: int
    edge_missing_rate: float
    edge_row_hit_at_1: float
    edge_column_hit_at_1: float
    edge_mutual_top1_rate: float
    edge_median_row_rank: float
    edge_median_column_rank: float
    next_diagnostic: str
    n_sessions: int
    reference_source: str

    def to_dict(self) -> dict[str, float | int | str]:
        """Return a flat, CSV/JSON-friendly representation."""

        return {
            "subject": self.subject,
            "failure_mode": self.failure_mode,
            "recommendation": self.recommendation,
            "method_pairwise_f1": self.method_pairwise_f1,
            "method_complete_track_f1": self.method_complete_track_f1,
            "oracle_pairwise_f1": self.oracle_pairwise_f1,
            "oracle_complete_track_f1": self.oracle_complete_track_f1,
            "pairwise_complete_f1_gap": self.pairwise_complete_f1_gap,
            "edge_score_name": self.edge_score_name,
            "edge_gt_edges": self.edge_gt_edges,
            "edge_present_edges": self.edge_present_edges,
            "edge_missing_edges": self.edge_missing_edges,
            "edge_missing_rate": self.edge_missing_rate,
            "edge_row_hit_at_1": self.edge_row_hit_at_1,
            "edge_column_hit_at_1": self.edge_column_hit_at_1,
            "edge_mutual_top1_rate": self.edge_mutual_top1_rate,
            "edge_median_row_rank": self.edge_median_row_rank,
            "edge_median_column_rank": self.edge_median_column_rank,
            "next_diagnostic": self.next_diagnostic,
            "n_sessions": self.n_sessions,
            "reference_source": self.reference_source,
        }


def run_track2p_failure_diagnosis(
    config: Track2pBenchmarkConfig,
    *,
    edge_score_name: str = "pairwise_cost_matrix",
    thresholds: DiagnosisThresholds = DiagnosisThresholds(),
) -> list[SubjectFailureDiagnosis]:
    """Run a compact diagnostic triage over Track2p-style subjects.

    The diagnostic separates four common failure modes:

    * oracle GT links do not reconstruct tracks, pointing at scoring/indexing;
    * GT links are missing or not top-ranked, pointing at registration/costs;
    * pairwise links are good but full tracks fragment, pointing at solver priors;
    * pairwise links are weak overall, pointing at the association model.
    """

    if config.split != "subject":
        raise ValueError(
            "Failure diagnosis expects subject-wise evaluation; LOSO calibration is diagnosed through its held-out benchmark rows."
        )
    if config.method != "global-assignment":
        raise ValueError("Failure diagnosis expects method='global-assignment'.")
    if config.cost == "calibrated":
        raise ValueError(
            "Failure diagnosis ranks raw pairwise costs; use a non-calibrated cost such as 'registered-iou' or 'roi-aware'."
        )

    subject_dirs = tuple(discover_subject_dirs(config.data))
    if not subject_dirs:
        raise ValueError(
            f"No Track2p-style subject directories found under {config.data}"
        )

    diagnoses: list[SubjectFailureDiagnosis] = []
    progress = ProgressReporter(
        len(subject_dirs), enabled=config.progress, label="diagnose"
    )
    for subject_dir in subject_dirs:
        progress.step(f"diagnosing {subject_dir.name}")
        reference = _load_reference_for_subject(
            subject_dir, data_root=config.data, config=config
        )
        _validate_reference_for_benchmark(
            reference, subject_dir=subject_dir, config=config
        )
        sessions = _load_subject_sessions(subject_dir, config)
        _validate_reference_roi_indices(reference, sessions)

        assignment = solve_configured_global_assignment(sessions, config)
        predicted = tracks_to_suite2p_index_matrix(assignment.result.tracks, sessions)
        method_scores = _score_prediction_against_reference(
            predicted, reference, config=config
        )

        oracle_tracks = oracle_ground_truth_link_tracks(
            reference,
            curated_only=config.curated_only,
            seed_session=config.seed_session,
        )
        oracle_scores = _score_prediction_against_reference(
            oracle_tracks, reference, config=config
        )

        edge_summary = _subject_edge_ranking_summary(
            subject_dir.name,
            sessions,
            reference,
            config,
            edge_score_name=edge_score_name,
        )
        method_pairwise_f1 = _score_float(method_scores, "pairwise_f1")
        method_complete_f1 = _score_float(method_scores, "complete_track_f1")
        oracle_pairwise_f1 = _score_float(oracle_scores, "pairwise_f1")
        oracle_complete_f1 = _score_float(oracle_scores, "complete_track_f1")
        pairwise_complete_gap = method_pairwise_f1 - method_complete_f1
        failure_mode, recommendation, next_diagnostic = classify_failure_mode(
            oracle_complete_track_f1=oracle_complete_f1,
            method_pairwise_f1=method_pairwise_f1,
            method_complete_track_f1=method_complete_f1,
            edge_mutual_top1_rate=_float(edge_summary["edge_mutual_top1_rate"]),
            edge_missing_rate=_float(edge_summary["edge_missing_rate"]),
            thresholds=thresholds,
        )
        diagnoses.append(
            SubjectFailureDiagnosis(
                subject=subject_dir.name,
                failure_mode=failure_mode,
                recommendation=recommendation,
                method_pairwise_f1=method_pairwise_f1,
                method_complete_track_f1=method_complete_f1,
                oracle_pairwise_f1=oracle_pairwise_f1,
                oracle_complete_track_f1=oracle_complete_f1,
                pairwise_complete_f1_gap=pairwise_complete_gap,
                edge_score_name=edge_score_name,
                edge_gt_edges=int(edge_summary["edge_gt_edges"]),
                edge_present_edges=int(edge_summary["edge_present_edges"]),
                edge_missing_edges=int(edge_summary["edge_missing_edges"]),
                edge_missing_rate=_float(edge_summary["edge_missing_rate"]),
                edge_row_hit_at_1=_float(edge_summary["edge_row_hit_at_1"]),
                edge_column_hit_at_1=_float(edge_summary["edge_column_hit_at_1"]),
                edge_mutual_top1_rate=_float(edge_summary["edge_mutual_top1_rate"]),
                edge_median_row_rank=_float(edge_summary["edge_median_row_rank"]),
                edge_median_column_rank=_float(edge_summary["edge_median_column_rank"]),
                next_diagnostic=next_diagnostic,
                n_sessions=reference.n_sessions,
                reference_source=reference.source,
            )
        )
    return diagnoses


def classify_failure_mode(
    *,
    oracle_complete_track_f1: float,
    method_pairwise_f1: float,
    method_complete_track_f1: float,
    edge_mutual_top1_rate: float,
    edge_missing_rate: float,
    thresholds: DiagnosisThresholds = DiagnosisThresholds(),
) -> tuple[FailureMode, str, str]:
    """Classify which benchmark subsystem should be fixed next."""

    oracle_complete_track_f1 = _finite_or(oracle_complete_track_f1, 0.0)
    method_pairwise_f1 = _finite_or(method_pairwise_f1, 0.0)
    method_complete_track_f1 = _finite_or(method_complete_track_f1, 0.0)
    edge_mutual_top1_rate = _finite_or(edge_mutual_top1_rate, 0.0)
    edge_missing_rate = _finite_or(edge_missing_rate, 1.0)

    if oracle_complete_track_f1 < thresholds.min_oracle_complete_f1:
        return (
            "row-assembly-or-scoring",
            "Oracle GT consecutive links fail to reconstruct complete tracks; inspect ROI index spaces, row stitching, and scoring before tuning costs.",
            "benchmark track2p --method oracle-gt-links",
        )
    if (
        edge_missing_rate > thresholds.max_edge_missing_rate
        or edge_mutual_top1_rate < thresholds.min_edge_mutual_top1_rate
    ):
        return (
            "registration-or-cost-ranking",
            "Manual-GT edges are missing from candidate matrices or are not top-ranked; inspect registration residuals and edge-ranking features.",
            "benchmark registration-qa / benchmark edge-ranking",
        )
    if (
        method_pairwise_f1 >= thresholds.min_pairwise_f1_for_solver_triage
        and method_pairwise_f1 - method_complete_track_f1
        >= thresholds.min_pairwise_complete_f1_gap
    ):
        return (
            "solver-priors-or-track-stitching",
            "Pairwise links are strong but complete tracks lag; tune start/end/gap/threshold priors or inspect track stitching.",
            "benchmark track2p-solver-prior-loso",
        )
    if method_pairwise_f1 < thresholds.min_pairwise_f1_for_solver_triage:
        return (
            "association-ranking",
            "Pairwise F1 is weak even after the oracle/indexing check passes; improve registration-aware costs, features, or calibration.",
            "benchmark edge-ranking / benchmark track2p-loso-calibration",
        )
    return (
        "no-dominant-failure",
        "The compact diagnostics do not isolate a dominant failure mode; compare per-subject ledgers and calibration diagnostics next.",
        "benchmark track2p-teacher-debug",
    )


def format_diagnosis_table(rows: Sequence[Mapping[str, float | int | str]]) -> str:
    """Format diagnosis rows as a compact Markdown table."""

    columns = [
        "subject",
        "failure_mode",
        "method_pairwise_f1",
        "method_complete_track_f1",
        "oracle_complete_track_f1",
        "edge_mutual_top1_rate",
        "edge_missing_rate",
        "next_diagnostic",
    ]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---", "---"] + ["---:"] * 5 + ["---"]) + " |"
    body = [header, separator]
    for row in rows:
        body.append(
            "| "
            + " | ".join(_format_table_value(row.get(column, "")) for column in columns)
            + " |"
        )
    return "\n".join(body)


def write_diagnosis_rows(
    rows: Sequence[Mapping[str, float | int | str]],
    output_path: Path,
    output_format: OutputFormat,
) -> None:
    """Write diagnosis rows as JSON, CSV, or Markdown table."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output_path.write_text(
            json.dumps(list(rows), indent=2) + "\n", encoding="utf-8"
        )
        return
    if output_format == "csv":
        fieldnames = _fieldnames(rows)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return
    output_path.write_text(format_diagnosis_table(rows) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for compact Track2p failure diagnosis."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-diagnose",
        description="Run a compact Track2p result triage that chooses the next diagnostic/fix target.",
    )
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Track2p dataset root or one subject directory",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Manual ground_truth.csv file or ground-truth root",
    )
    parser.add_argument(
        "--reference-kind",
        default="manual-gt",
        choices=("auto", "manual-gt", "track2p-output", "aligned-subject-rows"),
        help="Declared reference type; manual-gt is recommended for paper-facing diagnosis",
    )
    parser.add_argument(
        "--allow-track2p-as-reference-for-smoke-test", action="store_true"
    )
    parser.add_argument("--plane", dest="plane_name", default="plane0")
    parser.add_argument(
        "--input-format", default="auto", choices=("auto", "suite2p", "npy")
    )
    parser.add_argument("--curated-only", action="store_true")
    parser.add_argument("--seed-session", type=int, default=0)
    parser.add_argument(
        "--restrict-to-reference-seed-rois",
        action=argparse.BooleanOptionalAction,
        default=True,
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
        ),
        help="Raw global-assignment cost to diagnose",
    )
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument(
        "--transform-type",
        default="affine",
        choices=("affine", "rigid", "fov-translation", "none"),
    )
    parser.add_argument("--start-cost", type=float, default=5.0)
    parser.add_argument("--end-cost", type=float, default=5.0)
    parser.add_argument("--gap-penalty", type=float, default=1.0)
    parser.add_argument("--cost-threshold", type=float, default=6.0)
    parser.add_argument("--no-cost-threshold", action="store_true")
    parser.add_argument(
        "--include-behavior", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--include-non-cells", action="store_true")
    parser.add_argument("--cell-probability-threshold", type=float, default=0.5)
    parser.add_argument("--weighted-masks", action="store_true")
    parser.add_argument(
        "--exclude-overlapping-pixels",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--order", default="xy", choices=("xy", "yx"))
    parser.add_argument("--weighted-centroids", action="store_true")
    parser.add_argument("--velocity-variance", type=float, default=25.0)
    parser.add_argument("--regularization", type=float, default=1.0e-6)
    parser.add_argument(
        "--pairwise-cost-kwargs-json",
        default=None,
        help="JSON object merged into pairwise cost kwargs",
    )
    parser.add_argument(
        "--edge-score",
        default="pairwise_cost_matrix",
        help="Feature/cost plane used for GT edge-ranking triage",
    )
    parser.add_argument(
        "--min-oracle-complete-f1",
        type=float,
        default=DiagnosisThresholds.min_oracle_complete_f1,
    )
    parser.add_argument(
        "--min-edge-mutual-top1-rate",
        type=float,
        default=DiagnosisThresholds.min_edge_mutual_top1_rate,
    )
    parser.add_argument(
        "--min-pairwise-f1-for-solver-triage",
        type=float,
        default=DiagnosisThresholds.min_pairwise_f1_for_solver_triage,
    )
    parser.add_argument(
        "--min-pairwise-complete-f1-gap",
        type=float,
        default=DiagnosisThresholds.min_pairwise_complete_f1_gap,
    )
    parser.add_argument(
        "--max-edge-missing-rate",
        type=float,
        default=DiagnosisThresholds.max_edge_missing_rate,
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print progress to stderr",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("table", "json", "csv"), default="table")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for compact failure diagnosis."""

    args = build_arg_parser().parse_args(argv)
    config = _config_from_args(args)
    thresholds = DiagnosisThresholds(
        min_oracle_complete_f1=args.min_oracle_complete_f1,
        min_edge_mutual_top1_rate=args.min_edge_mutual_top1_rate,
        min_pairwise_f1_for_solver_triage=args.min_pairwise_f1_for_solver_triage,
        min_pairwise_complete_f1_gap=args.min_pairwise_complete_f1_gap,
        max_edge_missing_rate=args.max_edge_missing_rate,
    )
    diagnoses = run_track2p_failure_diagnosis(
        config, edge_score_name=args.edge_score, thresholds=thresholds
    )
    rows = [diagnosis.to_dict() for diagnosis in diagnoses]
    if args.output is not None:
        write_diagnosis_rows(rows, args.output, cast(OutputFormat, args.format))
    else:
        _write_stdout(rows, cast(OutputFormat, args.format))
    return 0


def _subject_edge_ranking_summary(
    subject_name: str,
    sessions: Sequence[Any],
    reference: Any,
    config: Track2pBenchmarkConfig,
    *,
    edge_score_name: str,
) -> dict[str, float | int]:
    feature_names = (str(edge_score_name),)
    score_directions: dict[str, ScoreDirection] = {}
    if edge_score_name in DEFAULT_SIMILARITY_FEATURES:
        score_directions[edge_score_name] = "similarity"
    options = ReferenceTrainingOptions(
        curated_only=config.curated_only,
        transform_type=config.transform_type,
        order=config.order,
        weighted_centroids=config.weighted_centroids,
        velocity_variance=config.velocity_variance,
        regularization=config.regularization,
        feature_names=feature_names,
        pairwise_cost_kwargs=_pairwise_cost_kwargs_for_config(
            config.cost, config.pairwise_cost_kwargs
        ),
    )
    rows: list[dict[str, float | int | str]] = []
    for block in collect_reference_pairwise_example_blocks(
        sessions,
        reference,
        session_edges=session_edge_pairs(len(sessions), max_gap=config.max_gap),
        options=options,
    ):
        metadata = {
            "subject": subject_name,
            "session_a": int(block.session_a),
            "session_b": int(block.session_b),
            "session_gap": int(block.gap),
        }
        score_matrices = score_matrices_from_feature_tensor(
            block.features, block.feature_names
        )
        rows.extend(
            rank_labeled_edges(
                block.labels,
                score_matrices,
                reference_roi_indices=block.reference_roi_indices,
                measurement_roi_indices=block.measurement_roi_indices,
                score_directions=score_directions,
                metadata=metadata,
            )
        )
        rows.extend(
            missing_reference_edge_rows(
                reference.pairwise_matches(
                    block.session_a, block.session_b, curated_only=config.curated_only
                ),
                reference_roi_indices=block.reference_roi_indices,
                measurement_roi_indices=block.measurement_roi_indices,
                score_names=block.feature_names,
                score_directions=score_directions,
                metadata=metadata,
            )
        )
    return _aggregate_edge_ranking_rows(rows)


def _aggregate_edge_ranking_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float | int]:
    gt_edges = len(rows)
    present_rows = [row for row in rows if _truthy(row.get("edge_present", 0))]
    finite_rows = [row for row in present_rows if _truthy(row.get("true_is_finite", 0))]
    missing_edges = gt_edges - len(present_rows)
    return {
        "edge_gt_edges": int(gt_edges),
        "edge_present_edges": int(len(present_rows)),
        "edge_missing_edges": int(missing_edges),
        "edge_missing_rate": _safe_rate(missing_edges, gt_edges),
        "edge_row_hit_at_1": _row_rate(
            rows, lambda row: _rank_at_most(row, "row_rank", 1)
        ),
        "edge_column_hit_at_1": _row_rate(
            rows, lambda row: _rank_at_most(row, "column_rank", 1)
        ),
        "edge_mutual_top1_rate": _row_rate(
            rows,
            lambda row: _rank_at_most(row, "row_rank", 1)
            and _rank_at_most(row, "column_rank", 1),
        ),
        "edge_median_row_rank": _median_rank(finite_rows, "row_rank"),
        "edge_median_column_rank": _median_rank(finite_rows, "column_rank"),
    }


def _config_from_args(args: argparse.Namespace) -> Track2pBenchmarkConfig:
    pairwise_cost_kwargs = None
    if args.pairwise_cost_kwargs_json is not None:
        parsed = json.loads(args.pairwise_cost_kwargs_json)
        if not isinstance(parsed, dict):
            raise ValueError("--pairwise-cost-kwargs-json must decode to a JSON object")
        pairwise_cost_kwargs = parsed
    return Track2pBenchmarkConfig(
        data=args.data,
        method="global-assignment",
        split="subject",
        plane_name=args.plane_name,
        input_format=args.input_format,
        reference=args.reference,
        reference_kind=args.reference_kind,
        allow_track2p_as_reference_for_smoke_test=args.allow_track2p_as_reference_for_smoke_test,
        curated_only=args.curated_only,
        seed_session=args.seed_session,
        restrict_to_reference_seed_rois=args.restrict_to_reference_seed_rois,
        cost=cast(AssociationCost, args.cost),
        max_gap=args.max_gap,
        transform_type=args.transform_type,
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
        progress=args.progress,
    )


def _write_stdout(
    rows: Sequence[Mapping[str, float | int | str]], output_format: OutputFormat
) -> None:
    if output_format == "json":
        print(json.dumps(list(rows), indent=2))
        return
    if output_format == "csv":
        writer = csv.DictWriter(_StdoutProxy(), fieldnames=_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)
        return
    print(format_diagnosis_table(rows))


def _fieldnames(rows: Sequence[Mapping[str, float | int | str]]) -> list[str]:
    preferred = [
        "subject",
        "failure_mode",
        "recommendation",
        "method_pairwise_f1",
        "method_complete_track_f1",
        "oracle_complete_track_f1",
        "pairwise_complete_f1_gap",
        "edge_mutual_top1_rate",
        "edge_missing_rate",
        "next_diagnostic",
    ]
    row_keys = {key for row in rows for key in row}
    return [key for key in preferred if key in row_keys] + sorted(
        row_keys - set(preferred)
    )


def _format_table_value(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return "nan"
        return f"{float(value):.3f}"
    return str(value)


def _score_float(scores: Mapping[str, float | int | str], key: str) -> float:
    return _float(scores.get(key, np.nan))


def _float(value: object) -> float:
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return float("nan")


def _finite_or(value: float, fallback: float) -> float:
    value = _float(value)
    return value if np.isfinite(value) else fallback


def _truthy(value: object) -> bool:
    try:
        return bool(int(cast(Any, value)))
    except (TypeError, ValueError):
        return False


def _rank_at_most(row: Mapping[str, Any], key: str, rank: int) -> bool:
    if not _truthy(row.get("edge_present", 0)) or not _truthy(
        row.get("true_is_finite", 0)
    ):
        return False
    try:
        return int(cast(Any, row.get(key, 0))) <= rank
    except (TypeError, ValueError):
        return False


def _row_rate(rows: Sequence[Mapping[str, Any]], predicate: Any) -> float:
    if not rows:
        return float("nan")
    return float(sum(1 for row in rows if predicate(row)) / len(rows))


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def _median_rank(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [_float(row.get(key, np.nan)) for row in rows]
    values = [value for value in values if np.isfinite(value)]
    if not values:
        return float("nan")
    return float(np.median(np.asarray(values, dtype=float)))


class _StdoutProxy:
    """File-like proxy that lets csv.DictWriter target stdout without importing sys globally."""

    def write(self, text: str) -> int:
        print(text, end="")
        return len(text)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
