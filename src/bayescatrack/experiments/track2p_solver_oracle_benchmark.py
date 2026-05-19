"""Paper-facing solver-oracle diagnostics for Track2p benchmarks.

The ordinary ``oracle-gt-links`` benchmark bypasses the global solver and checks
whether manual-GT consecutive links can be stitched and scored correctly.  This
module keeps the normal global assignment machinery in the loop and replaces
only selected upstream evidence with oracle information:

``edge-costs``
    Manual-GT edges get zero cost and every other admissible edge gets a large
    cost.  A failure here points at solver/track assembly/scoring.

``rank-k``
    Manual-GT edges are admitted only when their row rank under the configured
    base cost is at most ``k``.  This quantifies how much of the final tracking
    failure is already explained by pairwise edge ranking.

``oracle-registration``
    A manual-GT affine warp is fitted for each session edge before normal costs
    are computed.  This is an upper-bound diagnostic for registration quality;
    it must not be reported as an independent method.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from bayescatrack.association.pyrecest_global_assignment import (
    AssociationCost,
    tracks_to_suite2p_index_matrix,
)
from bayescatrack.experiments.solver_oracles import (
    oracle_edge_costs,
    oracle_rank_k_costs,
    oracle_registration_costs,
    solve_from_pairwise_costs,
)
from bayescatrack.experiments.track2p_benchmark import (
    OutputFormat,
    ProgressReporter,
    ReferenceKind,
    Track2pBenchmarkConfig,
    _load_reference_for_subject,
    _load_subject_sessions,
    _reference_matrix,
    _score_prediction_against_reference,
    _validate_reference_for_benchmark,
    _validate_reference_roi_indices,
    discover_subject_dirs,
)

SolverOracleKind = Literal["edge-costs", "rank-k", "oracle-registration"]
SolverOracleStatus = Literal["ok", "failed"]

DEFAULT_ORACLES: tuple[SolverOracleKind, ...] = (
    "edge-costs",
    "rank-k",
    "oracle-registration",
)
DEFAULT_RANK_KS = (1, 3, 5, 10)
DEFAULT_DETAIL_CSV = "solver_oracles.csv"
DEFAULT_SUMMARY_CSV = "solver_oracles_summary.csv"
DEFAULT_MARKDOWN = "solver_oracles.md"

DETAIL_FIELDNAMES = [
    "subject",
    "variant",
    "oracle",
    "rank_k",
    "base_cost",
    "oracle_registration_cost",
    "method",
    "status",
    "error",
    "n_sessions",
    "reference_source",
    "max_gap",
    "start_cost",
    "end_cost",
    "gap_penalty",
    "cost_threshold",
    "pairwise_f1",
    "complete_track_f1",
    "pairwise_precision",
    "pairwise_recall",
    "pairwise_true_positives",
    "pairwise_false_positives",
    "pairwise_false_negatives",
    "complete_tracks",
    "mean_track_length",
    "reference_seed_rois",
    "evaluated_prediction_tracks",
    "dropped_prediction_tracks",
]

SUMMARY_FIELDNAMES = [
    "variant",
    "oracle",
    "rank_k",
    "base_cost",
    "oracle_registration_cost",
    "ok_subjects",
    "failed_subjects",
    "mean_pairwise_f1",
    "min_pairwise_f1",
    "mean_complete_track_f1",
    "min_complete_track_f1",
    "mean_pairwise_precision",
    "mean_pairwise_recall",
    "mean_complete_tracks",
]


def run_track2p_solver_oracle_benchmark(
    config: Track2pBenchmarkConfig,
    *,
    oracles: Sequence[SolverOracleKind] = DEFAULT_ORACLES,
    rank_ks: Sequence[int] = DEFAULT_RANK_KS,
    oracle_registration_cost: AssociationCost = "registered-iou",
    large_cost: float = 1.0e6,
    min_fit_links: int = 3,
    require_full_rank: bool = True,
    ridge: float = 0.0,
) -> list[dict[str, float | int | str]]:
    """Run solver-oracle variants and return paper-facing result rows."""

    if config.reference_kind not in {"auto", "manual-gt"}:
        raise ValueError("Solver-oracle diagnostics require manual-GT references")
    _validate_oracle_choices(oracles, rank_ks, oracle_registration_cost)
    subject_dirs = tuple(discover_subject_dirs(config.data))
    if not subject_dirs:
        raise ValueError(
            f"No Track2p-style subject directories found under {config.data}"
        )

    rows: list[dict[str, float | int | str]] = []
    progress = ProgressReporter(
        len(subject_dirs), enabled=config.progress, label="solver-oracles"
    )
    for subject_dir in subject_dirs:
        progress.step(f"running {subject_dir.name}")
        reference = _load_reference_for_subject(
            subject_dir, data_root=config.data, config=config
        )
        _validate_reference_for_benchmark(
            reference, subject_dir=subject_dir, config=config
        )
        sessions = _load_subject_sessions(subject_dir, config)
        _validate_reference_roi_indices(reference, sessions)
        reference_matrix = _reference_matrix(reference, curated_only=config.curated_only)
        common_metadata: dict[str, float | int | str] = {
            "subject": subject_dir.name,
            "method": "solver-oracle",
            "n_sessions": int(reference.n_sessions),
            "reference_source": reference.source,
            "max_gap": int(config.max_gap),
            "start_cost": float(config.start_cost),
            "end_cost": float(config.end_cost),
            "gap_penalty": float(config.gap_penalty),
            "cost_threshold": (
                "none"
                if config.cost_threshold is None
                else float(config.cost_threshold)
            ),
        }
        for variant in _expanded_variants(oracles, rank_ks):
            rows.append(
                _run_subject_variant(
                    config,
                    subject_dir.name,
                    sessions,
                    reference,
                    reference_matrix,
                    variant,
                    common_metadata=common_metadata,
                    oracle_registration_cost=oracle_registration_cost,
                    large_cost=large_cost,
                    min_fit_links=min_fit_links,
                    require_full_rank=require_full_rank,
                    ridge=ridge,
                )
            )
    return rows


def summarize_solver_oracle_rows(
    rows: Sequence[Mapping[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    """Aggregate detailed oracle rows by variant for CSV/Markdown output."""

    grouped: dict[tuple[str, str, str, str, str], list[Mapping[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("variant", "")),
            str(row.get("oracle", "")),
            str(row.get("rank_k", "")),
            str(row.get("base_cost", "")),
            str(row.get("oracle_registration_cost", "")),
        )
        grouped[key].append(row)

    summary: list[dict[str, float | int | str]] = []
    for (
        variant,
        oracle,
        rank_k,
        base_cost,
        oracle_registration_cost,
    ), group_rows in sorted(grouped.items()):
        ok_rows = [row for row in group_rows if row.get("status") == "ok"]
        failed_rows = [row for row in group_rows if row.get("status") != "ok"]
        summary.append(
            {
                "variant": variant,
                "oracle": oracle,
                "rank_k": rank_k,
                "base_cost": base_cost,
                "oracle_registration_cost": oracle_registration_cost,
                "ok_subjects": len(ok_rows),
                "failed_subjects": len(failed_rows),
                "mean_pairwise_f1": _mean_metric(ok_rows, "pairwise_f1"),
                "min_pairwise_f1": _min_metric(ok_rows, "pairwise_f1"),
                "mean_complete_track_f1": _mean_metric(
                    ok_rows, "complete_track_f1"
                ),
                "min_complete_track_f1": _min_metric(ok_rows, "complete_track_f1"),
                "mean_pairwise_precision": _mean_metric(
                    ok_rows, "pairwise_precision"
                ),
                "mean_pairwise_recall": _mean_metric(ok_rows, "pairwise_recall"),
                "mean_complete_tracks": _mean_metric(ok_rows, "complete_tracks"),
            }
        )
    return summary


def format_solver_oracle_markdown(
    summary_rows: Sequence[Mapping[str, float | int | str]],
) -> str:
    """Return a paper-facing Markdown table plus interpretation notes."""

    columns = [
        "variant",
        "ok_subjects",
        "failed_subjects",
        "mean_pairwise_f1",
        "min_pairwise_f1",
        "mean_complete_track_f1",
        "min_complete_track_f1",
        "mean_pairwise_recall",
    ]
    lines = [
        "# Solver-oracle Track2p diagnostics",
        "",
        "These diagnostics keep the normal global-assignment solver in the loop and replace only selected upstream evidence with oracle information. They are upper bounds/debugging checks, not deployable methods.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(columns) - 1)) + " |",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(_format_markdown_value(row.get(column, "")) for column in columns)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `oracle edge costs` should be close to perfect. If it is not, inspect solver constraints, track-row assembly, ROI-index mapping, or scoring.",
            "- `oracle rank-k` measures how much final performance is possible when the true edge is only admitted if the base pairwise score ranks it in the top k of its row.",
            "- `oracle registration` measures the upper bound from GT-fitted affine registration while retaining normal pairwise costs and solver priors.",
            "",
        ]
    )
    return "\n".join(lines)


def write_solver_oracle_artifacts(
    rows: Sequence[Mapping[str, float | int | str]],
    output_dir: Path,
    *,
    detail_name: str = DEFAULT_DETAIL_CSV,
    summary_name: str = DEFAULT_SUMMARY_CSV,
    markdown_name: str = DEFAULT_MARKDOWN,
) -> tuple[Path, Path, Path]:
    """Write detailed CSV, summary CSV, and Markdown paper artifact."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / detail_name
    summary_path = output_dir / summary_name
    markdown_path = output_dir / markdown_name
    summary_rows = summarize_solver_oracle_rows(rows)
    _write_csv(rows, detail_path, preferred_fieldnames=DETAIL_FIELDNAMES)
    _write_csv(summary_rows, summary_path, preferred_fieldnames=SUMMARY_FIELDNAMES)
    markdown_path.write_text(
        format_solver_oracle_markdown(summary_rows), encoding="utf-8"
    )
    return detail_path, summary_path, markdown_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for solver-oracle diagnostics."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-solver-oracles",
        description="Run solver-oracle global-assignment diagnostics on Track2p-style datasets.",
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
        help="Manual-GT ground_truth.csv file, ground-truth root, or subject directory",
    )
    parser.add_argument(
        "--reference-kind",
        default="manual-gt",
        choices=("auto", "manual-gt"),
        help="Reference type; solver-oracle diagnostics require manual GT",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark-results") / "solver-oracles",
        help="Directory for detailed CSV, summary CSV, and Markdown artifacts",
    )
    parser.add_argument(
        "--oracles",
        nargs="+",
        default=list(DEFAULT_ORACLES),
        choices=DEFAULT_ORACLES,
        help="Oracle variants to run",
    )
    parser.add_argument(
        "--rank-k",
        dest="rank_ks",
        nargs="+",
        type=int,
        default=list(DEFAULT_RANK_KS),
        help="Top-k values for the rank-k oracle",
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
        help="Base pairwise cost used by rank-k oracle diagnostics",
    )
    parser.add_argument(
        "--oracle-registration-cost",
        default="registered-iou",
        choices=("registered-iou", "roi-aware"),
        help="Normal cost recomputed after manual-GT oracle affine registration",
    )
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument("--transform-type", default="affine", choices=("affine", "rigid", "fov-translation", "none"))
    parser.add_argument("--start-cost", type=float, default=5.0)
    parser.add_argument("--end-cost", type=float, default=5.0)
    parser.add_argument("--gap-penalty", type=float, default=1.0)
    parser.add_argument("--cost-threshold", type=float, default=6.0)
    parser.add_argument(
        "--no-cost-threshold",
        action="store_true",
        help="Disable the solver edge-cost threshold",
    )
    parser.add_argument("--large-cost", type=float, default=1.0e6)
    parser.add_argument("--min-fit-links", type=int, default=3)
    parser.add_argument(
        "--require-full-rank",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require full-rank manual-GT landmarks for oracle affine fits",
    )
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument("--plane", dest="plane_name", default="plane0")
    parser.add_argument("--input-format", default="auto", choices=("auto", "suite2p", "npy"))
    parser.add_argument("--curated-only", action="store_true")
    parser.add_argument("--seed-session", type=int, default=0)
    parser.add_argument(
        "--restrict-to-reference-seed-rois",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--include-behavior",
        action=argparse.BooleanOptionalAction,
        default=True,
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
        help="JSON object merged into base/oracle pairwise cost kwargs",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print progress to stderr",
    )
    parser.add_argument(
        "--format",
        choices=("table", "json", "csv"),
        default="table",
        help="Stdout format; all artifact files are written regardless",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = _config_from_args(args)
    rows = run_track2p_solver_oracle_benchmark(
        config,
        oracles=cast(Sequence[SolverOracleKind], args.oracles),
        rank_ks=args.rank_ks,
        oracle_registration_cost=cast(AssociationCost, args.oracle_registration_cost),
        large_cost=args.large_cost,
        min_fit_links=args.min_fit_links,
        require_full_rank=args.require_full_rank,
        ridge=args.ridge,
    )
    detail_path, summary_path, markdown_path = write_solver_oracle_artifacts(
        rows, args.output_dir
    )
    print(
        f"Wrote solver-oracle artifacts: {detail_path}, {summary_path}, {markdown_path}",
        file=sys.stderr,
    )
    _write_stdout(rows, args.format)
    return 0


def _expanded_variants(
    oracles: Sequence[SolverOracleKind], rank_ks: Sequence[int]
) -> tuple[tuple[SolverOracleKind, int | None], ...]:
    variants: list[tuple[SolverOracleKind, int | None]] = []
    for oracle in dict.fromkeys(oracles):
        if oracle == "rank-k":
            variants.extend((oracle, int(rank_k)) for rank_k in dict.fromkeys(rank_ks))
        else:
            variants.append((oracle, None))
    return tuple(variants)


def _run_subject_variant(
    config: Track2pBenchmarkConfig,
    subject: str,
    sessions: Sequence[Any],
    reference: Any,
    reference_matrix: np.ndarray,
    variant: tuple[SolverOracleKind, int | None],
    *,
    common_metadata: Mapping[str, float | int | str],
    oracle_registration_cost: AssociationCost,
    large_cost: float,
    min_fit_links: int,
    require_full_rank: bool,
    ridge: float,
) -> dict[str, float | int | str]:
    oracle, rank_k = variant
    row = {
        **common_metadata,
        "subject": subject,
        "variant": _variant_name(oracle, rank_k),
        "oracle": oracle,
        "rank_k": "" if rank_k is None else int(rank_k),
        "base_cost": config.cost,
        "oracle_registration_cost": (
            oracle_registration_cost if oracle == "oracle-registration" else ""
        ),
    }
    try:
        pairwise_costs = _pairwise_costs_for_variant(
            config,
            sessions,
            reference_matrix,
            oracle,
            rank_k=rank_k,
            oracle_registration_cost=oracle_registration_cost,
            large_cost=large_cost,
            min_fit_links=min_fit_links,
            require_full_rank=require_full_rank,
            ridge=ridge,
        )
        assignment = solve_from_pairwise_costs(
            pairwise_costs,
            sessions,
            max_gap=config.max_gap,
            start_cost=config.start_cost,
            end_cost=config.end_cost,
            gap_penalty=config.gap_penalty,
            cost_threshold=config.cost_threshold,
        )
        predicted = tracks_to_suite2p_index_matrix(assignment.result.tracks, sessions)
        scores = _score_prediction_against_reference(
            predicted, reference, config=config
        )
        return {**row, "status": "ok", "error": "", **scores}
    except Exception as exc:  # pragma: no cover - exercised on real-data edge cases
        return {**row, "status": "failed", "error": str(exc)}


def _pairwise_costs_for_variant(
    config: Track2pBenchmarkConfig,
    sessions: Sequence[Any],
    reference_matrix: np.ndarray,
    oracle: SolverOracleKind,
    *,
    rank_k: int | None,
    oracle_registration_cost: AssociationCost,
    large_cost: float,
    min_fit_links: int,
    require_full_rank: bool,
    ridge: float,
) -> dict[tuple[int, int], np.ndarray]:
    if oracle == "edge-costs":
        return oracle_edge_costs(
            sessions,
            reference_matrix,
            max_gap=config.max_gap,
            large_cost=large_cost,
        )
    if oracle == "rank-k":
        if rank_k is None:
            raise ValueError("rank-k oracle requires rank_k")
        return oracle_rank_k_costs(
            sessions,
            reference_matrix,
            rank_k=rank_k,
            max_gap=config.max_gap,
            cost=config.cost,
            transform_type=config.transform_type,
            order=config.order,
            weighted_centroids=config.weighted_centroids,
            velocity_variance=config.velocity_variance,
            regularization=config.regularization,
            pairwise_cost_kwargs=config.pairwise_cost_kwargs,
            large_cost=large_cost,
        )
    if oracle == "oracle-registration":
        return oracle_registration_costs(
            sessions,
            reference_matrix,
            max_gap=config.max_gap,
            cost=oracle_registration_cost,
            order=config.order,
            weighted_centroids=config.weighted_centroids,
            velocity_variance=config.velocity_variance,
            regularization=config.regularization,
            pairwise_cost_kwargs=config.pairwise_cost_kwargs,
            large_cost=large_cost,
            min_fit_links=min_fit_links,
            require_full_rank=require_full_rank,
            ridge=ridge,
        )
    raise ValueError(f"Unknown solver oracle {oracle!r}")


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
        plane_name=args.plane_name,
        input_format=args.input_format,
        reference=args.reference,
        reference_kind=cast(ReferenceKind, args.reference_kind),
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


def _validate_oracle_choices(
    oracles: Sequence[SolverOracleKind],
    rank_ks: Sequence[int],
    oracle_registration_cost: AssociationCost,
) -> None:
    if not oracles:
        raise ValueError("At least one solver oracle must be requested")
    if "rank-k" in oracles:
        if not rank_ks:
            raise ValueError("rank-k oracle requires at least one --rank-k value")
        invalid = [rank_k for rank_k in rank_ks if int(rank_k) < 1]
        if invalid:
            raise ValueError(f"rank-k values must be >= 1, got {invalid!r}")
    if oracle_registration_cost not in {"registered-iou", "roi-aware"}:
        raise ValueError("oracle-registration supports registered-iou or roi-aware")


def _variant_name(oracle: SolverOracleKind, rank_k: int | None) -> str:
    if oracle == "edge-costs":
        return "Oracle edge costs + global assignment"
    if oracle == "rank-k":
        return f"Oracle rank-{rank_k} admissible GT edges + global assignment"
    if oracle == "oracle-registration":
        return "Oracle affine registration + normal costs + global assignment"
    raise ValueError(f"Unknown solver oracle {oracle!r}")


def _write_stdout(
    rows: Sequence[Mapping[str, float | int | str]], output_format: OutputFormat
) -> None:
    if output_format == "json":
        print(json.dumps(list(rows), indent=2))
        return
    if output_format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=_csv_fieldnames(rows, []))
        writer.writeheader()
        writer.writerows(rows)
        return
    print(format_solver_oracle_markdown(summarize_solver_oracle_rows(rows)))


def _write_csv(
    rows: Sequence[Mapping[str, float | int | str]],
    output_path: Path,
    *,
    preferred_fieldnames: Sequence[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _csv_fieldnames(rows, preferred_fieldnames)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _csv_fieldnames(
    rows: Sequence[Mapping[str, float | int | str]], preferred: Sequence[str]
) -> list[str]:
    preferred = tuple(preferred)
    keys = {key for row in rows for key in row}
    return [key for key in preferred if key in keys] + sorted(keys - set(preferred))


def _mean_metric(rows: Iterable[Mapping[str, float | int | str]], key: str) -> float | str:
    values = _numeric_values(rows, key)
    if not values:
        return ""
    return float(np.mean(values))


def _min_metric(rows: Iterable[Mapping[str, float | int | str]], key: str) -> float | str:
    values = _numeric_values(rows, key)
    if not values:
        return ""
    return float(np.min(values))


def _numeric_values(
    rows: Iterable[Mapping[str, float | int | str]], key: str
) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value in {None, ""}:
            continue
        try:
            numeric = float(cast(Any, value))
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            values.append(numeric)
    return values


def _format_markdown_value(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    return str(value)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
