"""Pareto-safe Track2p-policy component-cleanup sweep selection.

This module wraps the existing Track2p-policy component-cleanup sweep with a
baseline safety selector.  The original sweep already supports a pairwise-F1
floor relative to the no-split baseline.  This selector adds the symmetric
complete-track-F1 floor so an operating point is only selected ahead of the
baseline when it is safe on both benchmark metrics.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

from bayescatrack.experiments import track2p_policy_component_sweep as base_sweep
from bayescatrack.experiments.track2p_benchmark import (
    OutputFormat,
    Track2pBenchmarkConfig,
    write_results,
)
from bayescatrack.experiments.track2p_policy_benchmark import (
    TRACK2P_POLICY_DEFAULT_MAX_GAP,
    ThresholdMethod,
)
from bayescatrack.experiments.track2p_policy_component_audit import (
    ComponentCleanupConfig,
)
from bayescatrack.experiments.track2p_policy_component_sweep import (
    NO_SPLIT_COMPONENT_CANDIDATE,
    ComponentCleanupSweepConfig,
    ComponentCleanupSweepOutput,
    ComponentSweepObjective,
)

DEFAULT_COMPLETE_TRACK_F1_FLOOR_DELTA = 0.0


def rerank_component_sweep_output(
    output: ComponentCleanupSweepOutput,
    *,
    objective: ComponentSweepObjective | None = None,
    pairwise_f1_floor_delta: float | None = 0.0,
    complete_track_f1_floor_delta: float | None = DEFAULT_COMPLETE_TRACK_F1_FLOOR_DELTA,
) -> ComponentCleanupSweepOutput:
    """Return a copy of ``output`` with Pareto-safe aggregate ranks.

    The no-split baseline remains feasible by construction.  Any cleanup
    candidate must satisfy all enabled baseline-relative floors before it can
    outrank the baseline.  This avoids selecting a cleanup configuration that
    improves complete-track identity but pays for it with a pairwise regression,
    or vice versa when users optimize pairwise or macro objectives.
    """

    objective = objective or output.objective
    ranked = rank_component_sweep_aggregates(
        output.aggregate_rows,
        objective=objective,
        pairwise_f1_floor_delta=pairwise_f1_floor_delta,
        complete_track_f1_floor_delta=complete_track_f1_floor_delta,
    )
    if not ranked:
        return ComponentCleanupSweepOutput(
            rows=output.rows,
            aggregate_rows=(),
            best_candidate=output.best_candidate,
            objective=objective,
        )

    best_candidate = str(ranked[0]["approach"])
    ranks = {str(row["approach"]): int(row["component_sweep_rank"]) for row in ranked}
    objectives = {
        str(row["approach"]): float(row["component_sweep_objective"]) for row in ranked
    }
    safety_metadata = {str(row["approach"]): _safety_metadata(row) for row in ranked}
    annotated_rows = []
    for row in output.rows:
        candidate = str(row["component_sweep_candidate"])
        rank = ranks.get(candidate, int(row.get("component_sweep_rank", 0)))
        annotated_rows.append(
            {
                **dict(row),
                **safety_metadata.get(candidate, {}),
                "component_sweep_rank": rank,
                "component_sweep_best": int(candidate == best_candidate),
                "component_sweep_objective": objectives.get(
                    candidate, float(row.get("component_sweep_objective", math.nan))
                ),
                "component_pareto_rank": rank,
                "component_pareto_best": int(candidate == best_candidate),
            }
        )
    return ComponentCleanupSweepOutput(
        rows=tuple(annotated_rows),
        aggregate_rows=tuple(ranked),
        best_candidate=best_candidate,
        objective=objective,
    )


def rank_component_sweep_aggregates(
    rows: Sequence[Mapping[str, float | int | str]],
    *,
    objective: ComponentSweepObjective,
    pairwise_f1_floor_delta: float | None = 0.0,
    complete_track_f1_floor_delta: float | None = DEFAULT_COMPLETE_TRACK_F1_FLOOR_DELTA,
) -> list[dict[str, float | int | str]]:
    """Rank aggregate sweep rows under baseline-relative metric floors."""

    baseline = _baseline_row(rows)
    pairwise_floor = _baseline_floor(
        baseline, "pairwise_f1_micro", pairwise_f1_floor_delta
    )
    complete_floor = _baseline_floor(
        baseline, "complete_track_f1_micro", complete_track_f1_floor_delta
    )
    enriched = []
    for row in rows:
        approach = str(row["approach"])
        pairwise_feasible = _floor_feasible(
            row,
            metric="pairwise_f1_micro",
            floor=pairwise_floor,
            baseline_approach=approach == NO_SPLIT_COMPONENT_CANDIDATE,
        )
        complete_feasible = _floor_feasible(
            row,
            metric="complete_track_f1_micro",
            floor=complete_floor,
            baseline_approach=approach == NO_SPLIT_COMPONENT_CANDIDATE,
        )
        safe_feasible = pairwise_feasible and complete_feasible
        enriched.append(
            {
                **dict(row),
                "component_sweep_objective": _objective_value(row, objective),
                "component_sweep_objective_name": objective,
                "component_sweep_pairwise_f1_floor": _floor_metadata(pairwise_floor),
                "component_sweep_complete_track_f1_floor": _floor_metadata(
                    complete_floor
                ),
                "component_sweep_pairwise_floor_feasible": int(pairwise_feasible),
                "component_sweep_complete_floor_feasible": int(complete_feasible),
                "component_sweep_baseline_safe_feasible": int(safe_feasible),
            }
        )
    ranked = sorted(
        enriched,
        key=lambda row: (
            -int(row["component_sweep_baseline_safe_feasible"]),
            -float(row["component_sweep_objective"]),
            -float(row["complete_track_f1_micro"]),
            -float(row["pairwise_f1_micro"]),
            str(row["approach"]),
        ),
    )
    return [
        {
            **row,
            "component_sweep_rank": rank,
            "component_pareto_rank": rank,
        }
        for rank, row in enumerate(ranked, start=1)
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the Pareto-safe component sweep."""

    parser = base_sweep.build_arg_parser()
    parser.prog = "bayescatrack benchmark track2p-policy-component-pareto-sweep"
    parser.description = (
        "Sweep Track2p-policy component cleanup settings with pairwise and "
        "complete-track baseline safety floors."
    )
    parser.add_argument(
        "--complete-track-f1-floor-delta",
        type=float,
        default=DEFAULT_COMPLETE_TRACK_F1_FLOOR_DELTA,
        help=(
            "Minimum allowed complete-track micro-F1 relative to the no-split "
            "baseline. Use a negative value to permit a small complete-track drop."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Pareto-safe Track2p-policy component-cleanup sweep CLI."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if (
        args.require_complete_track is not None
        and args.require_complete_track_options is not None
    ):
        parser.error(
            "use either --require-complete-track-options or "
            "--require-complete-track/--no-require-complete-track, not both"
        )

    base_cleanup = ComponentCleanupConfig(
        threshold_margin_scale=args.threshold_margin_scale,
        competition_margin_scale=args.competition_margin_scale,
        area_ratio_floor=args.area_ratio_floor,
        centroid_distance_scale=args.centroid_distance_scale,
    )
    sweep_kwargs: dict[str, Any] = {
        "split_risk_thresholds": base_sweep._float_tuple_arg(
            args.split_risk_thresholds, name="split-risk-thresholds"
        ),
        "split_penalties": base_sweep._float_tuple_arg(
            args.split_penalties, name="split-penalties"
        ),
        "min_side_observations": base_sweep._int_tuple_arg(
            args.min_side_observations, name="min-side-observations"
        ),
        "base_cleanup": base_cleanup,
        "objective": cast(ComponentSweepObjective, args.objective),
        "best_only": False,
        "include_baseline": bool(args.include_baseline),
        "pairwise_f1_floor_delta": args.pairwise_f1_floor_delta,
    }
    if args.require_complete_track_options is not None:
        sweep_kwargs["require_complete_track_options"] = base_sweep._bool_tuple_arg(
            args.require_complete_track_options, name="require-complete-track-options"
        )
    elif args.require_complete_track is not None:
        sweep_kwargs["require_complete_track_options"] = (
            bool(args.require_complete_track),
        )
    sweep_config = ComponentCleanupSweepConfig(**sweep_kwargs)
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
    output = base_sweep.run_track2p_policy_component_sweep(
        config,
        threshold_method=cast(ThresholdMethod, args.threshold_method),
        iou_distance_threshold=float(args.iou_distance_threshold),
        transform_type=args.transform_type,
        cell_probability_threshold=float(args.cell_probability_threshold),
        sweep_config=sweep_config,
    )
    output = rerank_component_sweep_output(
        output,
        objective=sweep_config.objective,
        pairwise_f1_floor_delta=args.pairwise_f1_floor_delta,
        complete_track_f1_floor_delta=args.complete_track_f1_floor_delta,
    )
    selected_rows = output.best_rows() if args.best_only else output.rows
    rows = [dict(row) for row in selected_rows]
    if args.output is not None:
        write_results(rows, args.output, cast(OutputFormat, args.format))
    else:
        from bayescatrack.experiments.track2p_benchmark import _write_stdout

        _write_stdout(rows, cast(OutputFormat, args.format))
    return 0


def _baseline_row(
    rows: Sequence[Mapping[str, float | int | str]],
) -> Mapping[str, float | int | str] | None:
    return next(
        (row for row in rows if str(row["approach"]) == NO_SPLIT_COMPONENT_CANDIDATE),
        None,
    )


def _baseline_floor(
    baseline: Mapping[str, float | int | str] | None,
    metric: str,
    delta: float | None,
) -> float | None:
    if baseline is None or delta is None:
        return None
    return float(baseline[metric]) + _finite_delta(delta, name=f"{metric}_floor_delta")


def _finite_delta(value: float, *, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite when provided")
    return numeric


def _floor_feasible(
    row: Mapping[str, float | int | str],
    *,
    metric: str,
    floor: float | None,
    baseline_approach: bool,
) -> bool:
    if floor is None or baseline_approach:
        return True
    return float(row[metric]) >= floor - 1e-12


def _floor_metadata(value: float | None) -> float | str:
    return "" if value is None else float(value)


def _objective_value(
    row: Mapping[str, float | int | str], objective: ComponentSweepObjective
) -> float:
    if objective == "mean_micro_f1":
        return 0.5 * (
            float(row["complete_track_f1_micro"]) + float(row["pairwise_f1_micro"])
        )
    return float(row[objective])


def _safety_metadata(
    row: Mapping[str, float | int | str],
) -> dict[str, float | int | str]:
    keys = (
        "component_sweep_pairwise_f1_floor",
        "component_sweep_complete_track_f1_floor",
        "component_sweep_pairwise_floor_feasible",
        "component_sweep_complete_floor_feasible",
        "component_sweep_baseline_safe_feasible",
    )
    return {key: row[key] for key in keys if key in row}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
