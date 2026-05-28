"""Compact parameter sweep for Track2p-policy gap-consensus cleanup."""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import product
from typing import Any, Literal, cast

from bayescatrack.experiments.benchmark_comparison import aggregate_rows
from bayescatrack.experiments.track2p_benchmark import OutputFormat, Track2pBenchmarkConfig, write_results
from bayescatrack.experiments.track2p_policy_benchmark import TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD, ThresholdMethod
from bayescatrack.experiments.track2p_policy_component_audit import ComponentCleanupConfig
from bayescatrack.experiments.track2p_policy_consensus_cleanup import CONSENSUS_MODES, ConsensusCleanupConfig, ConsensusMode
from bayescatrack.experiments.track2p_policy_gap_consensus_cleanup import (
    TRACK2P_POLICY_GAP_CONSENSUS_DEFAULT_MAX_GAP,
    build_arg_parser as _build_gap_consensus_parser,
    run_track2p_policy_gap_consensus_cleanup,
)
from bayescatrack.experiments.track2p_policy_stability_cleanup import StabilityCleanupConfig

GapConsensusSweepObjective = Literal["complete_track_f1_micro", "pairwise_f1_micro", "mean_micro_f1", "complete_track_f1_macro"]
GAP_CONSENSUS_SWEEP_OBJECTIVES = ("complete_track_f1_micro", "pairwise_f1_micro", "mean_micro_f1", "complete_track_f1_macro")


@dataclass(frozen=True)
class GapConsensusSweepConfig:
    """Grid and objective for selecting a cleanup operating point."""

    base_iou_distance_thresholds: tuple[float, ...] = (12.0, 14.0, 16.0)
    split_risk_thresholds: tuple[float, ...] = (1.25, 1.5, 1.75)
    split_penalties: tuple[float, ...] = (0.0, 0.25)
    min_side_observations: tuple[int, ...] = (2,)
    require_complete_track_options: tuple[bool, ...] = (True, False)
    max_splits_per_component: tuple[int, ...] = (1, 2)
    max_gaps: tuple[int, ...] = (TRACK2P_POLICY_GAP_CONSENSUS_DEFAULT_MAX_GAP,)
    consensus_modes: tuple[ConsensusMode, ...] = ("risk-and-stability",)
    stability_iou_distance_thresholds: tuple[float, ...] = (10.0, 12.0, 14.0)
    min_support_fraction: float = 2.0 / 3.0
    min_support_votes: int | None = None
    base_component: ComponentCleanupConfig = field(default_factory=ComponentCleanupConfig)
    objective: GapConsensusSweepObjective = "complete_track_f1_micro"
    best_only: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_iou_distance_thresholds", _finite_nonnegative_tuple(self.base_iou_distance_thresholds, name="base_iou_distance_thresholds"))
        object.__setattr__(self, "split_risk_thresholds", _finite_nonnegative_tuple(self.split_risk_thresholds, name="split_risk_thresholds"))
        object.__setattr__(self, "split_penalties", _finite_nonnegative_tuple(self.split_penalties, name="split_penalties"))
        object.__setattr__(self, "min_side_observations", _positive_int_tuple(self.min_side_observations, name="min_side_observations"))
        object.__setattr__(self, "require_complete_track_options", _boolean_tuple(self.require_complete_track_options, name="require_complete_track_options"))
        object.__setattr__(self, "max_splits_per_component", _positive_int_tuple(self.max_splits_per_component, name="max_splits_per_component"))
        object.__setattr__(self, "max_gaps", _positive_int_tuple(self.max_gaps, name="max_gaps"))
        object.__setattr__(self, "consensus_modes", _consensus_mode_tuple(self.consensus_modes))
        object.__setattr__(self, "stability_iou_distance_thresholds", _finite_nonnegative_tuple(self.stability_iou_distance_thresholds, name="stability_iou_distance_thresholds"))
        object.__setattr__(self, "min_support_fraction", _support_fraction_value(self.min_support_fraction, name="min_support_fraction"))
        if self.min_support_votes is not None:
            object.__setattr__(self, "min_support_votes", _positive_int_value(self.min_support_votes, name="min_support_votes"))
        if str(self.objective) not in GAP_CONSENSUS_SWEEP_OBJECTIVES:
            raise ValueError("objective must be one of: " + ", ".join(GAP_CONSENSUS_SWEEP_OBJECTIVES))


@dataclass(frozen=True)
class GapConsensusSweepOutput:
    """Subject rows and aggregate candidate ranking from a cleanup sweep."""

    rows: tuple[dict[str, float | int | str], ...]
    aggregate_rows: tuple[dict[str, float | int | str], ...]
    best_candidate: str
    objective: GapConsensusSweepObjective

    def best_rows(self) -> tuple[dict[str, float | int | str], ...]:
        """Return subject rows for the selected candidate only."""

        return tuple(row for row in self.rows if int(row["gap_consensus_sweep_best"]) == 1)


def run_track2p_policy_gap_consensus_sweep(
    config: Track2pBenchmarkConfig,
    *,
    threshold_method: ThresholdMethod = TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    transform_type: str | None = None,
    cell_probability_threshold: float | None = None,
    sweep_config: GapConsensusSweepConfig | None = None,
) -> GapConsensusSweepOutput:
    """Evaluate cleanup settings and mark the best aggregate candidate."""

    sweep_config = sweep_config or GapConsensusSweepConfig()
    candidate_rows: list[tuple[str, ConsensusCleanupConfig, int, list[dict[str, Any]]]] = []
    aggregate_input: list[dict[str, str]] = []
    for index, cleanup_config, max_gap in _cleanup_grid(sweep_config):
        candidate = _candidate_name(index, cleanup_config, max_gap)
        output = run_track2p_policy_gap_consensus_cleanup(
            config,
            threshold_method=threshold_method,
            transform_type=transform_type,
            cell_probability_threshold=cell_probability_threshold,
            max_gap=max_gap,
            cleanup_config=cleanup_config,
            apply_splits=True,
        )
        rows = [result.to_dict() for result in output.results]
        candidate_rows.append((candidate, cleanup_config, max_gap, rows))
        aggregate_input.extend(_aggregate_input_rows(candidate, rows))

    ranked = _rank_aggregates(aggregate_rows(aggregate_input), objective=sweep_config.objective)
    best_candidate = str(ranked[0]["approach"])
    ranks = {str(row["approach"]): int(row["gap_consensus_sweep_rank"]) for row in ranked}
    objectives = {str(row["approach"]): float(row["gap_consensus_sweep_objective"]) for row in ranked}
    rows = _annotate_subject_rows(candidate_rows, best_candidate, ranks, objectives)
    if sweep_config.best_only:
        rows = [row for row in rows if int(row["gap_consensus_sweep_best"]) == 1]
    return GapConsensusSweepOutput(tuple(rows), tuple(ranked), best_candidate, sweep_config.objective)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for gap-consensus cleanup sweeps."""

    parser = _build_gap_consensus_parser()
    parser.prog = "bayescatrack benchmark track2p-policy-gap-consensus-sweep"
    parser.description = "Sweep Track2p-policy gap-rescue consensus cleanup settings."
    _remove_parser_options(
        parser,
        "--apply-splits",
        "--component-output",
        "--component-format",
        "--base-iou-distance-threshold",
        "--split-risk-threshold",
        "--split-penalty",
        "--min-side-observations",
        "--require-complete-track",
        "--max-splits-per-component",
        "--max-gap",
        "--consensus-mode",
    )
    parser.add_argument("--base-iou-distance-thresholds", default="12,14,16")
    parser.add_argument("--split-risk-thresholds", default="1.25,1.5,1.75")
    parser.add_argument("--split-penalties", default="0,0.25")
    parser.add_argument("--min-side-observations", default="2")
    parser.add_argument("--require-complete-track-options", default="true,false")
    parser.add_argument("--max-splits-per-component", default="1,2")
    parser.add_argument("--max-gaps", default=str(TRACK2P_POLICY_GAP_CONSENSUS_DEFAULT_MAX_GAP))
    parser.add_argument("--consensus-modes", default="risk-and-stability")
    parser.add_argument("--objective", choices=GAP_CONSENSUS_SWEEP_OBJECTIVES, default="complete_track_f1_micro")
    parser.add_argument("--best-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Track2p-policy gap-consensus sweep CLI."""

    args = build_arg_parser().parse_args(argv)
    sweep_config = GapConsensusSweepConfig(
        base_iou_distance_thresholds=_float_tuple_arg(args.base_iou_distance_thresholds, name="base-iou-distance-thresholds"),
        split_risk_thresholds=_float_tuple_arg(args.split_risk_thresholds, name="split-risk-thresholds"),
        split_penalties=_float_tuple_arg(args.split_penalties, name="split-penalties"),
        min_side_observations=_int_tuple_arg(args.min_side_observations, name="min-side-observations"),
        require_complete_track_options=_bool_tuple_arg(args.require_complete_track_options, name="require-complete-track-options"),
        max_splits_per_component=_int_tuple_arg(args.max_splits_per_component, name="max-splits-per-component"),
        max_gaps=_int_tuple_arg(args.max_gaps, name="max-gaps"),
        consensus_modes=_consensus_mode_arg(args.consensus_modes),
        stability_iou_distance_thresholds=tuple(args.stability_iou_distance_thresholds),
        min_support_fraction=float(args.min_support_fraction),
        min_support_votes=args.min_support_votes,
        base_component=ComponentCleanupConfig(
            threshold_margin_scale=args.threshold_margin_scale,
            competition_margin_scale=args.competition_margin_scale,
            area_ratio_floor=args.area_ratio_floor,
            centroid_distance_scale=args.centroid_distance_scale,
        ),
        objective=cast(GapConsensusSweepObjective, args.objective),
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
        max_gap=TRACK2P_POLICY_GAP_CONSENSUS_DEFAULT_MAX_GAP,
        allow_track2p_as_reference_for_smoke_test=args.allow_track2p_as_reference_for_smoke_test,
        include_behavior=args.include_behavior,
        include_non_cells=False,
        cell_probability_threshold=args.cell_probability_threshold,
        exclude_overlapping_pixels=False,
        weighted_masks=False,
        weighted_centroids=False,
    )
    output = run_track2p_policy_gap_consensus_sweep(
        config,
        threshold_method=cast(ThresholdMethod, args.threshold_method),
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


def _cleanup_grid(config: GapConsensusSweepConfig) -> tuple[tuple[int, ConsensusCleanupConfig, int], ...]:
    output: list[tuple[int, ConsensusCleanupConfig, int]] = []
    for index, (base_iou, risk, penalty, min_side, complete, max_splits, max_gap, mode) in enumerate(
        product(
            config.base_iou_distance_thresholds,
            config.split_risk_thresholds,
            config.split_penalties,
            config.min_side_observations,
            config.require_complete_track_options,
            config.max_splits_per_component,
            config.max_gaps,
            config.consensus_modes,
        ),
        start=1,
    ):
        component = replace(config.base_component, split_risk_threshold=float(risk), split_penalty=float(penalty), min_side_observations=int(min_side), require_complete_track=bool(complete))
        stability = StabilityCleanupConfig(
            iou_distance_thresholds=config.stability_iou_distance_thresholds,
            base_iou_distance_threshold=float(base_iou),
            min_support_fraction=config.min_support_fraction,
            min_support_votes=config.min_support_votes,
            min_side_observations=int(min_side),
        )
        cleanup = ConsensusCleanupConfig(component=component, stability=stability, max_splits_per_component=int(max_splits), mode=mode)
        output.append((index, cleanup, int(max_gap)))
    return tuple(output)


def _candidate_name(index: int, cleanup: ConsensusCleanupConfig, max_gap: int) -> str:
    component = cleanup.component
    stability = cleanup.stability
    completeness = "complete" if component.require_complete_track else "partial"
    return (
        f"gap-consensus-{index:03d}"
        f"-base{stability.base_iou_distance_threshold:g}"
        f"-risk{component.split_risk_threshold:g}"
        f"-penalty{component.split_penalty:g}"
        f"-side{component.min_side_observations}"
        f"-splits{cleanup.max_splits_per_component}"
        f"-gap{max_gap}"
        f"-{cleanup.mode}"
        f"-{completeness}"
    )


def _aggregate_input_rows(candidate: str, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    return [{"approach": candidate, **{key: str(value) for key, value in row.items()}} for row in rows]


def _rank_aggregates(rows: Sequence[dict[str, float | int | str]], *, objective: GapConsensusSweepObjective) -> list[dict[str, float | int | str]]:
    enriched = [{**row, "gap_consensus_sweep_objective": _objective_value(row, objective), "gap_consensus_sweep_objective_name": objective} for row in rows]
    ranked = sorted(enriched, key=lambda row: (-float(row["gap_consensus_sweep_objective"]), -float(row["complete_track_f1_micro"]), -float(row["pairwise_f1_micro"]), str(row["approach"])))
    return [{**row, "gap_consensus_sweep_rank": rank} for rank, row in enumerate(ranked, start=1)]


def _objective_value(row: Mapping[str, float | int | str], objective: GapConsensusSweepObjective) -> float:
    if objective == "mean_micro_f1":
        return 0.5 * (float(row["complete_track_f1_micro"]) + float(row["pairwise_f1_micro"]))
    return float(row[objective])


def _annotate_subject_rows(candidate_rows: Sequence[tuple[str, ConsensusCleanupConfig, int, Sequence[Mapping[str, Any]]]], best_candidate: str, ranks: Mapping[str, int], objectives: Mapping[str, float]) -> list[dict[str, float | int | str]]:
    annotated: list[dict[str, float | int | str]] = []
    for candidate, cleanup, max_gap, rows in candidate_rows:
        for row in rows:
            annotated.append({**dict(row), "gap_consensus_sweep_candidate": candidate, "gap_consensus_sweep_rank": int(ranks[candidate]), "gap_consensus_sweep_best": int(candidate == best_candidate), "gap_consensus_sweep_objective": float(objectives[candidate]), "gap_consensus_sweep_base_iou_distance_threshold": float(cleanup.stability.base_iou_distance_threshold), "gap_consensus_sweep_split_risk_threshold": float(cleanup.component.split_risk_threshold), "gap_consensus_sweep_split_penalty": float(cleanup.component.split_penalty), "gap_consensus_sweep_min_side_observations": int(cleanup.component.min_side_observations), "gap_consensus_sweep_require_complete_track": int(cleanup.component.require_complete_track), "gap_consensus_sweep_max_splits_per_component": int(cleanup.max_splits_per_component), "gap_consensus_sweep_max_gap": int(max_gap), "gap_consensus_sweep_mode": str(cleanup.mode)})
    return annotated


def _remove_parser_options(parser: argparse.ArgumentParser, *option_strings: str) -> None:
    option_string_set = set(option_strings)
    actions = [action for action in parser._actions if option_string_set.intersection(action.option_strings)]
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


def _finite_nonnegative_value(value: Any, *, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} entries must be finite non-negative values") from exc
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} entries must be finite non-negative values")
    return numeric


def _support_fraction_value(value: Any, *, name: str) -> float:
    numeric = _finite_nonnegative_value(value, name=name)
    if not 0.0 < numeric <= 1.0:
        raise ValueError(f"{name} entries must lie in (0, 1]")
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


def _consensus_mode_tuple(values: Sequence[Any]) -> tuple[ConsensusMode, ...]:
    normalized = tuple(_consensus_mode_value(value) for value in values)
    if not normalized:
        raise ValueError("consensus_modes must not be empty")
    return normalized


def _consensus_mode_value(value: Any) -> ConsensusMode:
    token = str(value)
    if token not in CONSENSUS_MODES:
        raise ValueError("unsupported consensus mode")
    return cast(ConsensusMode, token)


def _float_tuple_arg(value: str, *, name: str) -> tuple[float, ...]:
    tokens = tuple(token.strip() for token in str(value).split(",") if token.strip())
    return _finite_nonnegative_tuple(tokens, name=name)


def _int_tuple_arg(value: str, *, name: str) -> tuple[int, ...]:
    tokens = tuple(token.strip() for token in str(value).split(",") if token.strip())
    return _positive_int_tuple(tokens, name=name)


def _bool_tuple_arg(value: str, *, name: str) -> tuple[bool, ...]:
    tokens = tuple(token.strip() for token in str(value).split(",") if token.strip())
    return _boolean_tuple(tokens, name=name)


def _consensus_mode_arg(value: str) -> tuple[ConsensusMode, ...]:
    tokens = tuple(token.strip() for token in str(value).split(",") if token.strip())
    return _consensus_mode_tuple(tokens)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
