"""Parameter sweep for Track2p-policy weakest-bridge component cleanup."""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import product
from pathlib import Path
from typing import Any, Literal, cast

from bayescatrack.experiments.benchmark_comparison import aggregate_rows
from bayescatrack.experiments.track2p_benchmark import (
    OutputFormat,
    Track2pBenchmarkConfig,
    write_results,
)
from bayescatrack.experiments.track2p_policy_benchmark import (
    TRACK2P_POLICY_DEFAULT_CELL_PROBABILITY_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_MAX_GAP,
    TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE,
    ThresholdMethod,
)
from bayescatrack.experiments.track2p_policy_component_audit import (
    ComponentCleanupConfig,
    run_track2p_policy_component_audit,
)

ComponentSweepObjective = Literal[
    "complete_track_f1_micro",
    "pairwise_f1_micro",
    "mean_micro_f1",
    "complete_track_f1_macro",
]


@dataclass(frozen=True)
class ComponentCleanupSweepConfig:
    """Grid and objective for selecting a cleanup operating point."""

    split_risk_thresholds: tuple[float, ...] = (1.0, 1.25, 1.5, 1.75, 2.0)
    split_penalties: tuple[float, ...] = (0.0, 0.25, 0.5)
    min_side_observations: tuple[int, ...] = (2,)
    base_cleanup: ComponentCleanupConfig = field(default_factory=ComponentCleanupConfig)
    objective: ComponentSweepObjective = "complete_track_f1_micro"
    best_only: bool = False

    def __post_init__(self) -> None:
        if not self.split_risk_thresholds:
            raise ValueError("split_risk_thresholds must not be empty")
        if not self.split_penalties:
            raise ValueError("split_penalties must not be empty")
        if not self.min_side_observations:
            raise ValueError("min_side_observations must not be empty")
        if any(float(value) < 0.0 for value in self.split_risk_thresholds):
            raise ValueError("split_risk_thresholds entries must be non-negative")
        if any(float(value) < 0.0 for value in self.split_penalties):
            raise ValueError("split_penalties entries must be non-negative")
        if any(int(value) < 1 for value in self.min_side_observations):
            raise ValueError("min_side_observations entries must be at least 1")


@dataclass(frozen=True)
class ComponentCleanupSweepOutput:
    """Subject rows and aggregate candidate ranking from a cleanup sweep."""

    rows: tuple[dict[str, float | int | str], ...]
    aggregate_rows: tuple[dict[str, float | int | str], ...]
    best_candidate: str
    objective: ComponentSweepObjective

    def best_rows(self) -> tuple[dict[str, float | int | str], ...]:
        """Return subject rows for the selected candidate only."""

        return tuple(row for row in self.rows if int(row["component_sweep_best"]) == 1)


def run_track2p_policy_component_sweep(
    config: Track2pBenchmarkConfig,
    *,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    iou_distance_threshold: float = TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    transform_type: str | None = None,
    cell_probability_threshold: float | None = None,
    sweep_config: ComponentCleanupSweepConfig | None = None,
) -> ComponentCleanupSweepOutput:
    """Evaluate cleanup settings and mark the best aggregate candidate."""

    sweep_config = sweep_config or ComponentCleanupSweepConfig()
    candidate_rows: list[tuple[str, ComponentCleanupConfig, list[dict[str, Any]]]] = []
    aggregate_input: list[dict[str, str]] = []
    for index, cleanup_config in enumerate(_cleanup_grid(sweep_config), start=1):
        candidate = _candidate_name(index, cleanup_config)
        output = run_track2p_policy_component_audit(
            config,
            threshold_method=threshold_method,
            iou_distance_threshold=iou_distance_threshold,
            transform_type=transform_type,
            cell_probability_threshold=cell_probability_threshold,
            cleanup_config=cleanup_config,
            apply_splits=True,
        )
        rows = [result.to_dict() for result in output.results]
        candidate_rows.append((candidate, cleanup_config, rows))
        aggregate_input.extend(_aggregate_input_rows(candidate, rows))

    ranked = _rank_aggregates(aggregate_rows(aggregate_input), objective=sweep_config.objective)
    best_candidate = str(ranked[0]["approach"])
    ranks = {str(row["approach"]): int(row["component_sweep_rank"]) for row in ranked}
    objectives = {str(row["approach"]): float(row["component_sweep_objective"]) for row in ranked}
    rows = _annotate_subject_rows(candidate_rows, best_candidate, ranks, objectives)
    if sweep_config.best_only:
        rows = [row for row in rows if int(row["component_sweep_best"]) == 1]
    return ComponentCleanupSweepOutput(
        rows=tuple(rows),
        aggregate_rows=tuple(ranked),
        best_candidate=best_candidate,
        objective=sweep_config.objective,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for component-cleanup parameter sweeps."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-policy-component-sweep",
        description="Sweep Track2p-policy weakest-bridge component cleanup settings.",
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument(
        "--reference-kind",
        choices=("auto", "manual-gt", "track2p-output", "aligned-subject-rows"),
        default="manual-gt",
    )
    parser.add_argument("--plane", dest="plane_name", default="plane0")
    parser.add_argument("--input-format", choices=("auto", "suite2p", "npy"), default="suite2p")
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
    parser.add_argument("--transform-type", default=TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE)
    parser.add_argument("--threshold-margin-scale", type=float, default=0.10)
    parser.add_argument("--competition-margin-scale", type=float, default=0.20)
    parser.add_argument("--area-ratio-floor", type=float, default=0.45)
    parser.add_argument("--centroid-distance-scale", type=float, default=4.0)
    parser.add_argument(
        "--require-complete-track",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only split components observed in every session unless disabled.",
    )
    parser.add_argument(
        "--split-risk-thresholds",
        default="1.0,1.25,1.5,1.75,2.0",
        help="Comma-separated split-risk thresholds to evaluate.",
    )
    parser.add_argument(
        "--split-penalties",
        default="0.0,0.25,0.5",
        help="Comma-separated split penalties to evaluate.",
    )
    parser.add_argument(
        "--min-side-observations",
        default="2",
        help="Comma-separated minimum fragment lengths to evaluate.",
    )
    parser.add_argument(
        "--objective",
        choices=(
            "complete_track_f1_micro",
            "pairwise_f1_micro",
            "mean_micro_f1",
            "complete_track_f1_macro",
        ),
        default="complete_track_f1_micro",
        help="Aggregate metric used to select the best cleanup candidate.",
    )
    parser.add_argument(
        "--best-only",
        action="store_true",
        help="Write only subject rows for the selected best candidate.",
    )
    parser.add_argument(
        "--restrict-to-reference-seed-rois",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed-session", type=int, default=0)
    parser.add_argument("--allow-track2p-as-reference-for-smoke-test", action="store_true")
    parser.add_argument("--include-behavior", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("table", "json", "csv"), default="table")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Track2p-policy component-cleanup sweep CLI."""

    args = build_arg_parser().parse_args(argv)
    base_cleanup = ComponentCleanupConfig(
        threshold_margin_scale=args.threshold_margin_scale,
        competition_margin_scale=args.competition_margin_scale,
        area_ratio_floor=args.area_ratio_floor,
        centroid_distance_scale=args.centroid_distance_scale,
        require_complete_track=args.require_complete_track,
    )
    sweep_config = ComponentCleanupSweepConfig(
        split_risk_thresholds=_float_tuple_arg(args.split_risk_thresholds, name="split-risk-thresholds"),
        split_penalties=_float_tuple_arg(args.split_penalties, name="split-penalties"),
        min_side_observations=_int_tuple_arg(args.min_side_observations, name="min-side-observations"),
        base_cleanup=base_cleanup,
        objective=cast(ComponentSweepObjective, args.objective),
        best_only=bool(args.best_only),
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
        max_gap=TRACK2P_POLICY_DEFAULT_MAX_GAP,
        allow_track2p_as_reference_for_smoke_test=args.allow_track2p_as_reference_for_smoke_test,
        include_behavior=args.include_behavior,
        include_non_cells=False,
        cell_probability_threshold=args.cell_probability_threshold,
        exclude_overlapping_pixels=False,
        weighted_masks=False,
        weighted_centroids=False,
    )
    output = run_track2p_policy_component_sweep(
        config,
        threshold_method=cast(ThresholdMethod, args.threshold_method),
        iou_distance_threshold=float(args.iou_distance_threshold),
        transform_type=args.transform_type,
        cell_probability_threshold=float(args.cell_probability_threshold),
        sweep_config=sweep_config,
    )
    rows = [dict(row) for row in output.rows]
    if args.output is not None:
        write_results(rows, args.output, cast(OutputFormat, args.format))
    else:
        from bayescatrack.experiments.track2p_benchmark import _write_stdout

        _write_stdout(rows, cast(OutputFormat, args.format))
    return 0


def _cleanup_grid(config: ComponentCleanupSweepConfig) -> tuple[ComponentCleanupConfig, ...]:
    return tuple(
        replace(
            config.base_cleanup,
            split_risk_threshold=float(risk),
            split_penalty=float(penalty),
            min_side_observations=int(min_side),
        )
        for risk, penalty, min_side in product(
            config.split_risk_thresholds,
            config.split_penalties,
            config.min_side_observations,
        )
    )


def _candidate_name(index: int, config: ComponentCleanupConfig) -> str:
    return (
        f"component-cleanup-{index:02d}"
        f"-risk{config.split_risk_threshold:g}"
        f"-penalty{config.split_penalty:g}"
        f"-side{config.min_side_observations}"
    )


def _aggregate_input_rows(candidate: str, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    return [{"approach": candidate, **{key: str(value) for key, value in row.items()}} for row in rows]


def _rank_aggregates(
    rows: Sequence[dict[str, float | int | str]],
    *,
    objective: ComponentSweepObjective,
) -> list[dict[str, float | int | str]]:
    enriched = [
        {
            **row,
            "component_sweep_objective": _objective_value(row, objective),
            "component_sweep_objective_name": objective,
        }
        for row in rows
    ]
    ranked = sorted(
        enriched,
        key=lambda row: (
            -float(row["component_sweep_objective"]),
            -float(row["complete_track_f1_micro"]),
            -float(row["pairwise_f1_micro"]),
            str(row["approach"]),
        ),
    )
    return [{**row, "component_sweep_rank": rank} for rank, row in enumerate(ranked, start=1)]


def _objective_value(row: Mapping[str, float | int | str], objective: ComponentSweepObjective) -> float:
    if objective == "mean_micro_f1":
        return 0.5 * (float(row["complete_track_f1_micro"]) + float(row["pairwise_f1_micro"]))
    return float(row[objective])


def _annotate_subject_rows(
    candidate_rows: Sequence[tuple[str, ComponentCleanupConfig, Sequence[Mapping[str, Any]]]],
    best_candidate: str,
    ranks: Mapping[str, int],
    objectives: Mapping[str, float],
) -> list[dict[str, float | int | str]]:
    annotated: list[dict[str, float | int | str]] = []
    for candidate, cleanup_config, rows in candidate_rows:
        for row in rows:
            annotated.append(
                {
                    **dict(row),
                    "component_sweep_candidate": candidate,
                    "component_sweep_rank": int(ranks[candidate]),
                    "component_sweep_best": int(candidate == best_candidate),
                    "component_sweep_objective": float(objectives[candidate]),
                    "component_sweep_split_risk_threshold": float(cleanup_config.split_risk_threshold),
                    "component_sweep_split_penalty": float(cleanup_config.split_penalty),
                    "component_sweep_min_side_observations": int(cleanup_config.min_side_observations),
                }
            )
    return annotated


def _float_tuple_arg(value: str, *, name: str) -> tuple[float, ...]:
    values = tuple(float(token.strip()) for token in str(value).split(",") if token.strip())
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    if any(not math.isfinite(item) for item in values):
        raise ValueError(f"{name} values must be finite")
    return values


def _int_tuple_arg(value: str, *, name: str) -> tuple[int, ...]:
    values = tuple(int(token.strip()) for token in str(value).split(",") if token.strip())
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    if any(item < 1 for item in values):
        raise ValueError(f"{name} values must be at least 1")
    return values


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
