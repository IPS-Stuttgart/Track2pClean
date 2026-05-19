"""Empirical shifted-IoU ablation for Track2p raw Suite2p benchmarks.

The single-radius shifted-IoU wrapper is useful for one-off benchmark runs.  This
module adds the paper-facing ablation harness: it evaluates exact registered IoU
and multiple shifted-IoU radii under the same solver settings, then writes
subject-level benchmark rows, aggregate comparison tables, and optional
edge-ranking diagnostics for the same radii.  The resulting artifacts make it
straightforward to answer whether shifted overlap improves the pairwise evidence
or merely changes the global assignment thresholding behaviour.
"""

from __future__ import annotations

# pylint: disable=protected-access

import argparse
import csv
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from bayescatrack.association.pyrecest_global_assignment import AssociationCost
from bayescatrack.association.shifted_overlap import SHIFTED_OVERLAP_KWARG_NAMES
from bayescatrack.experiments.benchmark_comparison import (
    ComparisonInput,
    aggregate_rows,
    load_labeled_rows,
    write_comparison,
    write_metric_csv,
    write_reference_gap_csv,
    write_subject_deficit_summary,
    write_subject_gap_summary,
    write_subject_metric_csv,
)
from bayescatrack.experiments.track2p_benchmark import (
    Track2pBenchmarkConfig,
    _config_from_args,
    build_arg_parser as build_track2p_arg_parser,
    run_track2p_benchmark,
    write_results,
)
from bayescatrack.experiments.track2p_edge_ranking import (
    DEFAULT_EDGE_RANKING_FEATURES,
    DEFAULT_SIMILARITY_FEATURES,
    run_track2p_edge_ranking,
)
from bayescatrack.experiments.track2p_fov_affine_benchmark import (
    _enable_fov_affine_choice,
    _register_plane_pair_with_fov_affine,
)

DEFAULT_RADII = (0, 2, 4, 6, 8)
DEFAULT_OUTPUT_DIR = Path("results/shifted_iou_ablation")
DEFAULT_SHIFT_PENALTY_WEIGHT = 0.25
SHIFTED_EDGE_RANKING_FEATURES = tuple(
    dict.fromkeys(
        (*DEFAULT_EDGE_RANKING_FEATURES,)
        + (
            "shifted_iou",
            "shifted_iou_cost",
            "shifted_mask_cosine_similarity",
            "shifted_mask_cosine_cost",
            "shifted_iou_shift_norm",
            "shifted_iou_shift_penalty_cost",
            "iou_for_cost",
            "mask_cosine_for_cost",
        )
    )
)
SHIFTED_SIMILARITY_FEATURES = tuple(
    dict.fromkeys(
        (*DEFAULT_SIMILARITY_FEATURES,)
        + (
            "shifted_iou",
            "shifted_mask_cosine_similarity",
            "iou_for_cost",
            "mask_cosine_for_cost",
        )
    )
)
EDGE_OVERVIEW_FIELDNAMES = (
    "approach",
    "shifted_iou_radius",
    "score_name",
    "summary_rows",
    "gt_edges",
    "present_edges",
    "missing_edges",
    "finite_true_edges",
    "row_hit_at_1",
    "row_hit_at_3",
    "row_hit_at_5",
    "row_hit_at_10",
    "column_hit_at_1",
    "column_hit_at_3",
    "column_hit_at_5",
    "column_hit_at_10",
    "mutual_top1_rate",
    "mean_row_margin",
    "mean_column_margin",
    "mean_median_row_rank",
    "mean_median_column_rank",
)


@dataclass(frozen=True)
class ShiftedIouAblationArtifacts:
    """Paths written by the shifted-IoU ablation harness."""

    output_dir: Path
    benchmark_rows: Path
    benchmark_comparison_md: Path
    benchmark_comparison_csv: Path
    benchmark_metric_csv: Path
    subject_metric_csv: Path
    subject_gap_summary_md: Path
    subject_deficit_summary_md: Path
    edge_ranking_rows: Path | None = None
    edge_ranking_summary: Path | None = None
    edge_ranking_overview_csv: Path | None = None
    edge_ranking_overview_md: Path | None = None

    def existing_paths(self) -> tuple[Path, ...]:
        """Return non-optional output paths for CLI status messages."""

        return tuple(path for path in self.__dict__.values() if isinstance(path, Path))


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for the shifted-IoU empirical ablation."""

    parser = build_track2p_arg_parser()
    parser.prog = "bayescatrack benchmark track2p-shifted-iou-ablation"
    parser.description = (
        "Run exact-IoU versus shifted-IoU radii under identical Track2p "
        "global-assignment settings and export benchmark plus edge-ranking artifacts."
    )
    _remove_optional_argument(parser, "--output")
    _remove_optional_argument(parser, "--format")
    _set_argument_default(
        parser,
        "method",
        "global-assignment",
        help_text="Benchmark method; this ablation defaults to global-assignment.",
    )
    _set_argument_default(
        parser,
        "split",
        "subject",
        help_text="Evaluation split; this ablation currently runs subject-level variants.",
    )
    _restrict_cost_choices(parser)
    _enable_fov_affine_choice(parser)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for ablation artifacts (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--radii",
        default=",".join(str(radius) for radius in DEFAULT_RADII),
        help="Comma-separated shifted-IoU radii; include 0 for the exact-IoU baseline.",
    )
    parser.add_argument(
        "--shifted-iou-additive-weight",
        type=float,
        default=0.0,
        help=(
            "Optional additive shifted-IoU cost weight. The main shifted-IoU "
            "ablation replaces the registered-IoU term; this adds a second term."
        ),
    )
    parser.add_argument(
        "--shifted-mask-cosine-weight",
        type=float,
        default=0.0,
        help="Optional additive best-shift mask-cosine cost weight.",
    )
    parser.add_argument(
        "--shifted-iou-shift-penalty-weight",
        type=float,
        default=DEFAULT_SHIFT_PENALTY_WEIGHT,
        help=(
            "Cost weight for the residual local shift selected by shifted IoU. "
            "Use 0 to test unregularized shifted overlap."
        ),
    )
    parser.add_argument(
        "--shifted-iou-shift-penalty-scale",
        type=float,
        default=None,
        help="Positive scale for the residual-shift penalty; defaults to the radius.",
    )
    parser.add_argument(
        "--edge-ranking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also export per-edge and summary manual-GT edge-ranking diagnostics.",
    )
    parser.add_argument(
        "--edge-ranking-feature",
        dest="edge_ranking_features",
        action="append",
        default=None,
        help="Feature/component to rank; repeat to override the default ablation set.",
    )
    parser.add_argument(
        "--similarity-feature",
        dest="similarity_features",
        action="append",
        default=None,
        help="Declare a ranked feature where larger values are better; repeat as needed.",
    )
    parser.add_argument(
        "--reference-approach",
        default="r0-exact",
        help="Approach label used for benchmark gap summaries.",
    )
    parser.add_argument(
        "--subject-gap-summary-limit",
        type=int,
        default=12,
        help="Maximum rows in subject-level gap and deficit Markdown summaries.",
    )
    return parser


def run_shifted_iou_ablation(
    base_config: Track2pBenchmarkConfig,
    *,
    output_dir: Path,
    radii: Sequence[int],
    shifted_iou_additive_weight: float = 0.0,
    shifted_mask_cosine_weight: float = 0.0,
    shifted_iou_shift_penalty_weight: float = DEFAULT_SHIFT_PENALTY_WEIGHT,
    shifted_iou_shift_penalty_scale: float | None = None,
    edge_ranking: bool = True,
    edge_ranking_features: Sequence[str] | None = None,
    similarity_features: Sequence[str] | None = None,
    reference_approach: str | None = "r0-exact",
    subject_gap_summary_limit: int = 12,
) -> ShiftedIouAblationArtifacts:
    """Run the shifted-IoU ablation and write paper-facing artifacts."""

    radii = tuple(dict.fromkeys(int(radius) for radius in radii))
    _validate_ablation_config(
        base_config,
        radii=radii,
        shifted_iou_additive_weight=shifted_iou_additive_weight,
        shifted_mask_cosine_weight=shifted_mask_cosine_weight,
        shifted_iou_shift_penalty_weight=shifted_iou_shift_penalty_weight,
        shifted_iou_shift_penalty_scale=shifted_iou_shift_penalty_scale,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_inputs: list[ComparisonInput] = []
    benchmark_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, str]] = []
    edge_summary_rows: list[dict[str, str]] = []

    with _patched_fov_affine_registration():
        for radius in radii:
            config = _config_for_radius(
                base_config,
                radius,
                shifted_iou_additive_weight=shifted_iou_additive_weight,
                shifted_mask_cosine_weight=shifted_mask_cosine_weight,
                shifted_iou_shift_penalty_weight=shifted_iou_shift_penalty_weight,
                shifted_iou_shift_penalty_scale=shifted_iou_shift_penalty_scale,
            )
            approach = _approach_label(radius)
            benchmark_path = output_dir / f"benchmark_{approach}.csv"
            rows = [result.to_dict() for result in run_track2p_benchmark(config)]
            context = _ablation_context(
                approach,
                radius,
                shifted_iou_additive_weight=shifted_iou_additive_weight,
                shifted_mask_cosine_weight=shifted_mask_cosine_weight,
                shifted_iou_shift_penalty_weight=shifted_iou_shift_penalty_weight,
                shifted_iou_shift_penalty_scale=shifted_iou_shift_penalty_scale,
            )
            for row in rows:
                row.update(context)
            write_results(rows, benchmark_path, "csv")
            benchmark_inputs.append(ComparisonInput(label=approach, path=benchmark_path))
            benchmark_rows.extend(rows)

            if edge_ranking:
                edge_path = output_dir / f"edge_ranking_{approach}.csv"
                edge_summary_path = output_dir / f"edge_ranking_{approach}_summary.csv"
                run_track2p_edge_ranking(
                    config,
                    edge_path,
                    summary_output_path=edge_summary_path,
                    feature_names=_edge_ranking_features_for_radius(
                        radius, edge_ranking_features
                    ),
                    similarity_features=_similarity_features_for_radius(
                        radius, similarity_features
                    ),
                )
                edge_rows.extend(_read_csv_with_context(edge_path, context))
                edge_summary_rows.extend(
                    _read_csv_with_context(edge_summary_path, context)
                )

    benchmark_rows_path = output_dir / "shifted_iou_benchmark_rows.csv"
    _write_csv(benchmark_rows, benchmark_rows_path)

    subject_rows = load_labeled_rows(benchmark_inputs)
    comparison_rows = aggregate_rows(subject_rows)
    benchmark_comparison_md = output_dir / "shifted_iou_benchmark_comparison.md"
    benchmark_comparison_csv = output_dir / "shifted_iou_benchmark_comparison.csv"
    benchmark_metric_csv = output_dir / "shifted_iou_metric_ranks.csv"
    subject_metric_csv = output_dir / "shifted_iou_subject_metric_ranks.csv"
    subject_gap_summary_md = output_dir / "shifted_iou_subject_gap_summary.md"
    subject_deficit_summary_md = output_dir / "shifted_iou_subject_deficit_summary.md"
    write_comparison(
        comparison_rows,
        benchmark_comparison_md,
        "markdown",
        highlight_best=True,
        include_best_summary=True,
        include_reference_gap_summary=reference_approach is not None,
        reference_approach=reference_approach,
    )
    write_comparison(comparison_rows, benchmark_comparison_csv, "csv")
    write_reference_gap_csv(
        comparison_rows,
        output_dir / "shifted_iou_reference_gaps.csv",
        reference_approach=reference_approach,
    )
    write_metric_csv(
        comparison_rows,
        benchmark_metric_csv,
        reference_approach=reference_approach,
    )
    write_subject_metric_csv(
        subject_rows,
        subject_metric_csv,
        reference_approach=reference_approach,
    )
    write_subject_gap_summary(
        subject_rows,
        subject_gap_summary_md,
        reference_approach=reference_approach,
        limit=subject_gap_summary_limit,
    )
    write_subject_deficit_summary(
        subject_rows,
        subject_deficit_summary_md,
        reference_approach=reference_approach,
        limit=subject_gap_summary_limit,
    )

    if not edge_ranking:
        return ShiftedIouAblationArtifacts(
            output_dir=output_dir,
            benchmark_rows=benchmark_rows_path,
            benchmark_comparison_md=benchmark_comparison_md,
            benchmark_comparison_csv=benchmark_comparison_csv,
            benchmark_metric_csv=benchmark_metric_csv,
            subject_metric_csv=subject_metric_csv,
            subject_gap_summary_md=subject_gap_summary_md,
            subject_deficit_summary_md=subject_deficit_summary_md,
        )

    edge_rows_path = output_dir / "shifted_iou_edge_ranking_rows.csv"
    edge_summary_path = output_dir / "shifted_iou_edge_ranking_summary.csv"
    edge_overview_csv = output_dir / "shifted_iou_edge_ranking_overview.csv"
    edge_overview_md = output_dir / "shifted_iou_edge_ranking_overview.md"
    edge_overview_rows = _aggregate_edge_ranking_summary_rows(edge_summary_rows)
    _write_csv(edge_rows, edge_rows_path)
    _write_csv(edge_summary_rows, edge_summary_path)
    _write_csv(edge_overview_rows, edge_overview_csv, fieldnames=EDGE_OVERVIEW_FIELDNAMES)
    _write_edge_overview_markdown(edge_overview_rows, edge_overview_md)
    return ShiftedIouAblationArtifacts(
        output_dir=output_dir,
        benchmark_rows=benchmark_rows_path,
        benchmark_comparison_md=benchmark_comparison_md,
        benchmark_comparison_csv=benchmark_comparison_csv,
        benchmark_metric_csv=benchmark_metric_csv,
        subject_metric_csv=subject_metric_csv,
        subject_gap_summary_md=subject_gap_summary_md,
        subject_deficit_summary_md=subject_deficit_summary_md,
        edge_ranking_rows=edge_rows_path,
        edge_ranking_summary=edge_summary_path,
        edge_ranking_overview_csv=edge_overview_csv,
        edge_ranking_overview_md=edge_overview_md,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the shifted-IoU empirical ablation CLI."""

    args = build_arg_parser().parse_args(argv)
    radii = _parse_radii(args.radii)
    artifacts = run_shifted_iou_ablation(
        _config_from_args(args),
        output_dir=args.output_dir,
        radii=radii,
        shifted_iou_additive_weight=args.shifted_iou_additive_weight,
        shifted_mask_cosine_weight=args.shifted_mask_cosine_weight,
        shifted_iou_shift_penalty_weight=args.shifted_iou_shift_penalty_weight,
        shifted_iou_shift_penalty_scale=args.shifted_iou_shift_penalty_scale,
        edge_ranking=args.edge_ranking,
        edge_ranking_features=args.edge_ranking_features,
        similarity_features=args.similarity_features,
        reference_approach=args.reference_approach or None,
        subject_gap_summary_limit=args.subject_gap_summary_limit,
    )
    print("Wrote shifted-IoU ablation artifacts:")
    for path in artifacts.existing_paths():
        print(f"- {path}")
    return 0


def _parse_radii(text: str) -> tuple[int, ...]:
    radii: list[int] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        radius = int(item)
        if radius < 0:
            raise ValueError("Shifted-IoU radii must be non-negative")
        radii.append(radius)
    radii = list(dict.fromkeys(radii))
    if not radii:
        raise ValueError("At least one shifted-IoU radius is required")
    return tuple(radii)


def _config_for_radius(
    base_config: Track2pBenchmarkConfig,
    radius: int,
    *,
    shifted_iou_additive_weight: float,
    shifted_mask_cosine_weight: float,
    shifted_iou_shift_penalty_weight: float,
    shifted_iou_shift_penalty_scale: float | None,
) -> Track2pBenchmarkConfig:
    cost = _cost_for_radius(base_config.cost, radius)
    pairwise_kwargs: dict[str, Any] = dict(base_config.pairwise_cost_kwargs or {})
    if radius <= 0:
        for key in SHIFTED_OVERLAP_KWARG_NAMES:
            pairwise_kwargs.pop(key, None)
        return replace(
            base_config,
            cost=cost,
            pairwise_cost_kwargs=pairwise_kwargs or None,
        )

    pairwise_kwargs.update(
        {
            "shifted_iou_radius": int(radius),
            "use_shifted_iou_for_iou_cost": True,
            "shifted_iou_weight": float(shifted_iou_additive_weight),
            "shifted_mask_cosine_weight": float(shifted_mask_cosine_weight),
            "shifted_iou_shift_penalty_weight": float(
                shifted_iou_shift_penalty_weight
            ),
        }
    )
    if cost == "roi-aware-shifted":
        pairwise_kwargs["use_shifted_mask_cosine_for_mask_cosine_cost"] = True
    if shifted_iou_shift_penalty_scale is not None:
        pairwise_kwargs["shifted_iou_shift_penalty_scale"] = float(
            shifted_iou_shift_penalty_scale
        )
    return replace(base_config, cost=cost, pairwise_cost_kwargs=pairwise_kwargs)


def _cost_for_radius(cost: AssociationCost, radius: int) -> AssociationCost:
    if cost in {"registered-iou", "registered-shifted-iou"}:
        return "registered-iou" if radius <= 0 else "registered-shifted-iou"
    if cost in {"roi-aware", "roi-aware-shifted"}:
        return "roi-aware" if radius <= 0 else "roi-aware-shifted"
    raise ValueError(
        "Shifted-IoU ablation requires --cost registered-iou or --cost roi-aware"
    )


def _approach_label(radius: int) -> str:
    return "r0-exact" if int(radius) == 0 else f"shifted-r{int(radius)}"


def _ablation_context(
    approach: str,
    radius: int,
    *,
    shifted_iou_additive_weight: float,
    shifted_mask_cosine_weight: float,
    shifted_iou_shift_penalty_weight: float,
    shifted_iou_shift_penalty_scale: float | None,
) -> dict[str, float | int | str]:
    return {
        "approach": approach,
        "shifted_iou_radius": int(radius),
        "shifted_iou_additive_weight": float(shifted_iou_additive_weight),
        "shifted_mask_cosine_weight": float(shifted_mask_cosine_weight),
        "shifted_iou_shift_penalty_weight": float(
            shifted_iou_shift_penalty_weight
        ),
        "shifted_iou_shift_penalty_scale": (
            ""
            if shifted_iou_shift_penalty_scale is None
            else float(shifted_iou_shift_penalty_scale)
        ),
    }


def _edge_ranking_features_for_radius(
    radius: int, requested: Sequence[str] | None
) -> tuple[str, ...]:
    if requested is not None:
        return tuple(dict.fromkeys(str(feature) for feature in requested))
    return SHIFTED_EDGE_RANKING_FEATURES if radius > 0 else DEFAULT_EDGE_RANKING_FEATURES


def _similarity_features_for_radius(
    radius: int, requested: Sequence[str] | None
) -> tuple[str, ...]:
    if requested is not None:
        return tuple(dict.fromkeys(str(feature) for feature in requested))
    return SHIFTED_SIMILARITY_FEATURES if radius > 0 else DEFAULT_SIMILARITY_FEATURES


def _aggregate_edge_ranking_summary_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, float | int | str]]:
    groups: "OrderedDict[tuple[str, str, str], list[Mapping[str, Any]]]" = OrderedDict()
    for row in rows:
        key = (
            str(row.get("approach", "")),
            str(row.get("shifted_iou_radius", "")),
            str(row.get("score_name", "")),
        )
        groups.setdefault(key, []).append(row)

    aggregate_rows_out: list[dict[str, float | int | str]] = []
    for (approach, radius, score_name), group_rows in groups.items():
        gt_edges = _sum_float(group_rows, "gt_edges")
        present_edges = _sum_float(group_rows, "present_edges")
        missing_edges = _sum_float(group_rows, "missing_edges")
        finite_true_edges = _sum_float(group_rows, "finite_true_edges")
        aggregate_rows_out.append(
            {
                "approach": approach,
                "shifted_iou_radius": _int_or_string(radius),
                "score_name": score_name,
                "summary_rows": len(group_rows),
                "gt_edges": int(gt_edges),
                "present_edges": int(present_edges),
                "missing_edges": int(missing_edges),
                "finite_true_edges": int(finite_true_edges),
                "row_hit_at_1": _weighted_mean(group_rows, "row_hit_at_1", "gt_edges"),
                "row_hit_at_3": _weighted_mean(group_rows, "row_hit_at_3", "gt_edges"),
                "row_hit_at_5": _weighted_mean(group_rows, "row_hit_at_5", "gt_edges"),
                "row_hit_at_10": _weighted_mean(group_rows, "row_hit_at_10", "gt_edges"),
                "column_hit_at_1": _weighted_mean(
                    group_rows, "column_hit_at_1", "gt_edges"
                ),
                "column_hit_at_3": _weighted_mean(
                    group_rows, "column_hit_at_3", "gt_edges"
                ),
                "column_hit_at_5": _weighted_mean(
                    group_rows, "column_hit_at_5", "gt_edges"
                ),
                "column_hit_at_10": _weighted_mean(
                    group_rows, "column_hit_at_10", "gt_edges"
                ),
                "mutual_top1_rate": _weighted_mean(
                    group_rows, "mutual_top1_rate", "gt_edges"
                ),
                "mean_row_margin": _weighted_mean(
                    group_rows, "mean_row_margin", "finite_true_edges"
                ),
                "mean_column_margin": _weighted_mean(
                    group_rows, "mean_column_margin", "finite_true_edges"
                ),
                "mean_median_row_rank": _weighted_mean(
                    group_rows, "median_row_rank", "finite_true_edges"
                ),
                "mean_median_column_rank": _weighted_mean(
                    group_rows, "median_column_rank", "finite_true_edges"
                ),
            }
        )
    return aggregate_rows_out


@contextmanager
def _patched_fov_affine_registration() -> Any:
    import bayescatrack.association.calibrated_costs as calibrated_costs
    import bayescatrack.association.pyrecest_global_assignment as assignment

    original_assignment_register = assignment.register_plane_pair
    original_calibrated_register = calibrated_costs.register_plane_pair
    assignment.register_plane_pair = _register_plane_pair_with_fov_affine
    calibrated_costs.register_plane_pair = _register_plane_pair_with_fov_affine
    try:
        yield
    finally:
        assignment.register_plane_pair = original_assignment_register
        calibrated_costs.register_plane_pair = original_calibrated_register


def _validate_ablation_config(
    config: Track2pBenchmarkConfig,
    *,
    radii: Sequence[int],
    shifted_iou_additive_weight: float,
    shifted_mask_cosine_weight: float,
    shifted_iou_shift_penalty_weight: float,
    shifted_iou_shift_penalty_scale: float | None,
) -> None:
    if config.method != "global-assignment":
        raise ValueError("Shifted-IoU ablation requires method='global-assignment'")
    if config.split != "subject":
        raise ValueError("Shifted-IoU ablation currently requires split='subject'")
    if config.cost not in {"registered-iou", "roi-aware"}:
        raise ValueError(
            "Shifted-IoU ablation requires --cost registered-iou or --cost roi-aware"
        )
    if not radii:
        raise ValueError("At least one shifted-IoU radius is required")
    if any(int(radius) < 0 for radius in radii):
        raise ValueError("Shifted-IoU radii must be non-negative")
    if shifted_iou_additive_weight < 0.0:
        raise ValueError("--shifted-iou-additive-weight must be non-negative")
    if shifted_mask_cosine_weight < 0.0:
        raise ValueError("--shifted-mask-cosine-weight must be non-negative")
    if shifted_iou_shift_penalty_weight < 0.0:
        raise ValueError("--shifted-iou-shift-penalty-weight must be non-negative")
    if shifted_iou_shift_penalty_scale is not None and shifted_iou_shift_penalty_scale <= 0.0:
        raise ValueError("--shifted-iou-shift-penalty-scale must be strictly positive")


def _remove_optional_argument(parser: argparse.ArgumentParser, option_string: str) -> None:
    action = parser._option_string_actions.get(  # pylint: disable=protected-access
        option_string
    )
    if action is None:
        return
    for option in action.option_strings:
        parser._option_string_actions.pop(option, None)  # pylint: disable=protected-access
    if action in parser._actions:  # pylint: disable=protected-access
        parser._actions.remove(action)  # pylint: disable=protected-access
    for group in parser._action_groups:  # pylint: disable=protected-access
        if action in group._group_actions:  # pylint: disable=protected-access
            group._group_actions.remove(action)  # pylint: disable=protected-access


def _set_argument_default(
    parser: argparse.ArgumentParser,
    dest: str,
    default: Any,
    *,
    help_text: str | None = None,
) -> None:
    for action in parser._actions:  # pylint: disable=protected-access
        if action.dest == dest:
            action.required = False
            action.default = default
            if help_text is not None:
                action.help = help_text
            return
    raise RuntimeError(f"Could not find parser action {dest!r}")


def _restrict_cost_choices(parser: argparse.ArgumentParser) -> None:
    for action in parser._actions:  # pylint: disable=protected-access
        if action.dest == "cost":
            action.choices = ("registered-iou", "roi-aware")
            action.default = "registered-iou"
            action.help = (
                "Base cost to ablate. Radius 0 uses this exact cost; positive "
                "radii use the corresponding shifted-IoU variant."
            )
            return
    raise RuntimeError("Could not find --cost action")


def _read_csv_with_context(
    path: Path, context: Mapping[str, float | int | str]
) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {**row, **{key: str(value) for key, value in context.items()}}
            for row in csv.DictReader(handle)
        ]


def _write_csv(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(fieldnames) if fieldnames is not None else _fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    keys = {key for row in rows for key in row}
    preferred = [
        "approach",
        "shifted_iou_radius",
        "subject",
        "variant",
        "method",
        "pairwise_f1",
        "complete_track_f1",
    ]
    return [key for key in preferred if key in keys] + sorted(keys - set(preferred))


def _write_edge_overview_markdown(
    rows: Sequence[Mapping[str, Any]], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "approach",
        "score_name",
        "gt_edges",
        "row_hit_at_1",
        "row_hit_at_3",
        "mutual_top1_rate",
        "mean_row_margin",
        "mean_column_margin",
    )
    body = [
        "### Shifted-IoU edge-ranking overview",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---", "---"] + ["---:"] * (len(columns) - 2)) + " |",
    ]
    for row in rows:
        cells = (_format_cell(row.get(column, "")) for column in columns)
        body.append("| " + " | ".join(cells) + " |")
    output_path.write_text("\n".join(body) + "\n", encoding="utf-8")


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _weighted_mean(
    rows: Sequence[Mapping[str, Any]], value_key: str, weight_key: str
) -> float:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        weight = _safe_float(row.get(weight_key))
        value = _safe_float(row.get(value_key))
        if weight <= 0.0:
            continue
        numerator += value * weight
        denominator += weight
    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)


def _sum_float(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(sum(_safe_float(row.get(key)) for row in rows))


def _safe_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int_or_string(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
