"""Parameter sweep for Track2p-policy weakest-bridge component cleanup."""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import product
from typing import Any, Literal, cast

from bayescatrack.experiments.benchmark_comparison import aggregate_rows
from bayescatrack.experiments.track2p_benchmark import (
    OutputFormat,
    Track2pBenchmarkConfig,
    write_results,
)
from bayescatrack.experiments.track2p_policy_benchmark import (
    TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_MAX_GAP,
    TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    ThresholdMethod,
)
from bayescatrack.experiments.track2p_policy_component_audit import (
    ComponentCleanupConfig,
)
from bayescatrack.experiments.track2p_policy_component_audit import (
    build_arg_parser as _build_component_audit_parser,
)
from bayescatrack.experiments.track2p_policy_component_audit import (
    run_track2p_policy_component_audit,
)

ComponentSweepObjective = Literal[
    "complete_track_f1_micro",
    "pairwise_f1_micro",
    "mean_micro_f1",
    "complete_track_f1_macro",
]
COMPONENT_SWEEP_OBJECTIVES = (
    "complete_track_f1_micro",
    "pairwise_f1_micro",
    "mean_micro_f1",
    "complete_track_f1_macro",
)

NO_SPLIT_COMPONENT_CANDIDATE = "component-cleanup-00-no-split"


@dataclass(frozen=True)
class ComponentCleanupSweepConfig:
    """Grid and objective for selecting a cleanup operating point."""

    split_risk_thresholds: tuple[float, ...] = (1.0, 1.25, 1.5, 1.75, 2.0)
    split_penalties: tuple[float, ...] = (0.0, 0.25, 0.5)
    min_side_observations: tuple[int, ...] = (2,)
    require_complete_track_options: tuple[bool, ...] = (True, False)
    include_baseline: bool = True
    pairwise_f1_floor_delta: float | None = 0.0
    base_cleanup: ComponentCleanupConfig = field(default_factory=ComponentCleanupConfig)
    objective: ComponentSweepObjective = "complete_track_f1_micro"
    best_only: bool = False

    def __post_init__(self) -> None:
        split_risk_thresholds = _finite_nonnegative_tuple(
            self.split_risk_thresholds, name="split_risk_thresholds"
        )
        split_penalties = _finite_nonnegative_tuple(
            self.split_penalties, name="split_penalties"
        )
        min_side_observations = _positive_int_tuple(
            self.min_side_observations, name="min_side_observations"
        )
        require_complete_track_options = _boolean_tuple(
            self.require_complete_track_options, name="require_complete_track_options"
        )
        if str(self.objective) not in COMPONENT_SWEEP_OBJECTIVES:
            raise ValueError(
                "objective must be one of: " + ", ".join(COMPONENT_SWEEP_OBJECTIVES)
            )
        pairwise_f1_floor_delta = (
            None
            if self.pairwise_f1_floor_delta is None
            else _finite_value(
                self.pairwise_f1_floor_delta, name="pairwise_f1_floor_delta"
            )
        )
        object.__setattr__(self, "split_risk_thresholds", split_risk_thresholds)
        object.__setattr__(self, "split_penalties", split_penalties)
        object.__setattr__(self, "min_side_observations", min_side_observations)
        object.__setattr__(
            self, "require_complete_track_options", require_complete_track_options
        )
        object.__setattr__(
            self,
            "include_baseline",
            _strict_boolean_value(self.include_baseline, name="include_baseline"),
        )
        object.__setattr__(
            self,
            "best_only",
            _strict_boolean_value(self.best_only, name="best_only"),
        )
        object.__setattr__(self, "pairwise_f1_floor_delta", pairwise_f1_floor_delta)


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
    if sweep_config.include_baseline:
        output = run_track2p_policy_component_audit(
            config,
            threshold_method=threshold_method,
            iou_distance_threshold=iou_distance_threshold,
            transform_type=transform_type,
            cell_probability_threshold=cell_probability_threshold,
            cleanup_config=sweep_config.base_cleanup,
            apply_splits=False,
        )
        rows = [result.to_dict() for result in output.results]
        candidate_rows.append(
            (NO_SPLIT_COMPONENT_CANDIDATE, sweep_config.base_cleanup, rows)
        )
        aggregate_input.extend(
            _aggregate_input_rows(NO_SPLIT_COMPONENT_CANDIDATE, rows)
        )

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

    aggregate = aggregate_rows(aggregate_input)
    ranked = _rank_aggregates(
        aggregate,
        objective=sweep_config.objective,
        pairwise_f1_floor=_pairwise_f1_floor(aggregate, sweep_config),
    )
    best_candidate = str(ranked[0]["approach"])
    ranks = {str(row["approach"]): int(row["component_sweep_rank"]) for row in ranked}
    objectives = {
        str(row["approach"]): float(row["component_sweep_objective"]) for row in ranked
    }
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

    parser = _build_component_audit_parser()
    parser.prog = "bayescatrack benchmark track2p-policy-component-sweep"
    parser.description = (
        "Sweep Track2p-policy weakest-bridge component cleanup settings."
    )
    _remove_parser_options(
        parser,
        "--apply-splits",
        "--component-output",
        "--component-format",
        "--split-risk-threshold",
        "--split-penalty",
        "--min-side-observations",
        "--require-complete-track",
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
        "--require-complete-track",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Evaluate only one complete-track guard setting. By default the "
            "sweep evaluates both complete-track-only and partial-track cleanup."
        ),
    )
    parser.add_argument(
        "--require-complete-track-options",
        default=None,
        help=(
            "Comma-separated complete-track guard settings to evaluate, e.g. "
            "true,false. Defaults to true,false unless --require-complete-track "
            "or --no-require-complete-track is used."
        ),
    )
    parser.add_argument(
        "--objective",
        choices=COMPONENT_SWEEP_OBJECTIVES,
        default="complete_track_f1_micro",
        help="Aggregate metric used to select the best cleanup candidate.",
    )
    parser.add_argument(
        "--best-only",
        action="store_true",
        help="Write only subject rows for the selected best candidate.",
    )
    parser.add_argument(
        "--include-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include a no-split Track2p-policy baseline in the sweep so the "
            "selected cleanup cannot be worse than doing nothing under the "
            "chosen aggregate objective."
        ),
    )
    parser.add_argument(
        "--pairwise-f1-floor-delta",
        type=float,
        default=0.0,
        help=(
            "Minimum allowed pairwise micro-F1 relative to the no-split baseline. "
            "Use a negative value to permit a small pairwise drop."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Track2p-policy component-cleanup sweep CLI."""

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
        "split_risk_thresholds": _float_tuple_arg(
            args.split_risk_thresholds, name="split-risk-thresholds"
        ),
        "split_penalties": _float_tuple_arg(
            args.split_penalties, name="split-penalties"
        ),
        "min_side_observations": _int_tuple_arg(
            args.min_side_observations, name="min-side-observations"
        ),
        "base_cleanup": base_cleanup,
        "objective": cast(ComponentSweepObjective, args.objective),
        "best_only": bool(args.best_only),
        "include_baseline": bool(args.include_baseline),
        "pairwise_f1_floor_delta": args.pairwise_f1_floor_delta,
    }
    if args.require_complete_track_options is not None:
        sweep_kwargs["require_complete_track_options"] = _bool_tuple_arg(
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


def _remove_parser_options(
    parser: argparse.ArgumentParser, *option_strings: str
) -> None:
    option_string_set = set(option_strings)
    actions = [
        action
        for action in parser._actions
        if option_string_set.intersection(action.option_strings)
    ]
    for action in actions:
        for option_string in action.option_strings:
            parser._option_string_actions.pop(option_string, None)
        if action in parser._actions:
            parser._actions.remove(action)
        for group in parser._action_groups:
            if action in group._group_actions:
                group._group_actions.remove(action)


def _finite_nonnegative_tuple(values: Sequence[Any], *, name: str) -> tuple[float, ...]:
    normalized = tuple(_finite_nonnegative_value(value, name=name) for value in values)
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _finite_value(value: Any, *, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} entries must be finite values") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} entries must be finite values")
    return numeric


def _finite_nonnegative_value(value: Any, *, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} entries must be finite non-negative values") from exc
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} entries must be finite non-negative values")
    return numeric


def _positive_int_tuple(values: Sequence[Any], *, name: str) -> tuple[int, ...]:
    normalized = tuple(_positive_int_value(value, name=name) for value in values)
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _positive_int_value(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} entries must be positive integers")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} entries must be positive integers") from exc
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric < 1.0:
        raise ValueError(f"{name} entries must be positive integers")
    return int(numeric)


def _boolean_tuple(values: Sequence[Any], *, name: str) -> tuple[bool, ...]:
    normalized = tuple(_boolean_value(value, name=name) for value in values)
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _boolean_value(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "y"}:
            return True
        if token in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"{name} entries must be boolean values")


def _strict_boolean_value(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _cleanup_grid(
    config: ComponentCleanupSweepConfig,
) -> tuple[ComponentCleanupConfig, ...]:
    return tuple(
        replace(
            config.base_cleanup,
            split_risk_threshold=float(risk),
            split_penalty=float(penalty),
            min_side_observations=int(min_side),
            require_complete_track=bool(require_complete_track),
        )
        for risk, penalty, min_side, require_complete_track in product(
            config.split_risk_thresholds,
            config.split_penalties,
            config.min_side_observations,
            config.require_complete_track_options,
        )
    )


def _candidate_name(index: int, config: ComponentCleanupConfig) -> str:
    completeness = "complete" if config.require_complete_track else "partial"
    return (
        f"component-cleanup-{index:02d}"
        f"-risk{config.split_risk_threshold:g}"
        f"-penalty{config.split_penalty:g}"
        f"-side{config.min_side_observations}"
        f"-{completeness}"
    )


def _aggregate_input_rows(
    candidate: str, rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, str]]:
    return [
        {"approach": candidate, **{key: str(value) for key, value in row.items()}}
        for row in rows
    ]


def _rank_aggregates(
    rows: Sequence[dict[str, float | int | str]],
    *,
    objective: ComponentSweepObjective,
    pairwise_f1_floor: float | None = None,
) -> list[dict[str, float | int | str]]:
    enriched = []
    for row in rows:
        approach = str(row["approach"])
        floor_feasible = (
            pairwise_f1_floor is None
            or approach == NO_SPLIT_COMPONENT_CANDIDATE
            or float(row["pairwise_f1_micro"]) >= pairwise_f1_floor - 1e-12
        )
        enriched.append(
            {
                **row,
                "component_sweep_objective": _objective_value(row, objective),
                "component_sweep_objective_name": objective,
                "component_sweep_pairwise_f1_floor": (
                    "" if pairwise_f1_floor is None else float(pairwise_f1_floor)
                ),
                "component_sweep_pairwise_floor_feasible": int(floor_feasible),
            }
        )
    ranked = sorted(
        enriched,
        key=lambda row: (
            -int(row["component_sweep_pairwise_floor_feasible"]),
            -float(row["component_sweep_objective"]),
            -float(row["complete_track_f1_micro"]),
            -float(row["pairwise_f1_micro"]),
            str(row["approach"]),
        ),
    )
    return [
        {**row, "component_sweep_rank": rank}
        for rank, row in enumerate(ranked, start=1)
    ]


def _pairwise_f1_floor(
    rows: Sequence[dict[str, float | int | str]],
    config: ComponentCleanupSweepConfig,
) -> float | None:
    if not config.include_baseline or config.pairwise_f1_floor_delta is None:
        return None
    baseline = next(
        (row for row in rows if str(row["approach"]) == NO_SPLIT_COMPONENT_CANDIDATE),
        None,
    )
    if baseline is None:
        return None
    return float(baseline["pairwise_f1_micro"]) + float(config.pairwise_f1_floor_delta)


def _objective_value(
    row: Mapping[str, float | int | str], objective: ComponentSweepObjective
) -> float:
    if objective == "mean_micro_f1":
        return 0.5 * (
            float(row["complete_track_f1_micro"]) + float(row["pairwise_f1_micro"])
        )
    return float(row[objective])


def _annotate_subject_rows(
    candidate_rows: Sequence[
        tuple[str, ComponentCleanupConfig, Sequence[Mapping[str, Any]]]
    ],
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
                    "component_sweep_split_risk_threshold": float(
                        cleanup_config.split_risk_threshold
                    ),
                    "component_sweep_split_penalty": float(
                        cleanup_config.split_penalty
                    ),
                    "component_sweep_min_side_observations": int(
                        cleanup_config.min_side_observations
                    ),
                    "component_sweep_require_complete_track": int(
                        cleanup_config.require_complete_track
                    ),
                }
            )
    return annotated


def _float_tuple_arg(value: str, *, name: str) -> tuple[float, ...]:
    tokens = tuple(token.strip() for token in str(value).split(",") if token.strip())
    return _finite_nonnegative_tuple(tokens, name=name)


def _int_tuple_arg(value: str, *, name: str) -> tuple[int, ...]:
    tokens = tuple(token.strip() for token in str(value).split(",") if token.strip())
    return _positive_int_tuple(tokens, name=name)


def _bool_tuple_arg(value: str, *, name: str) -> tuple[bool, ...]:
    tokens = tuple(token.strip() for token in str(value).split(",") if token.strip())
    return _boolean_tuple(tokens, name=name)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
