"""Parameter sweep for Track2p-policy weakest-bridge component cleanup."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import product
from typing import Any, Literal

from bayescatrack.experiments.benchmark_comparison import aggregate_rows
from bayescatrack.experiments.track2p_benchmark import Track2pBenchmarkConfig
from bayescatrack.experiments.track2p_policy_benchmark import (
    TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
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
COMPONENT_SWEEP_OBJECTIVES = (
    "complete_track_f1_micro",
    "pairwise_f1_micro",
    "mean_micro_f1",
    "complete_track_f1_macro",
)


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
        risks = _finite_nonnegative_tuple(
            self.split_risk_thresholds, name="split_risk_thresholds"
        )
        penalties = _finite_nonnegative_tuple(
            self.split_penalties, name="split_penalties"
        )
        min_sides = tuple(int(value) for value in self.min_side_observations)
        if not min_sides:
            raise ValueError("min_side_observations must not be empty")
        if any(value < 1 for value in min_sides):
            raise ValueError("min_side_observations entries must be at least 1")
        if str(self.objective) not in COMPONENT_SWEEP_OBJECTIVES:
            raise ValueError(
                "objective must be one of: " + ", ".join(COMPONENT_SWEEP_OBJECTIVES)
            )
        object.__setattr__(self, "split_risk_thresholds", risks)
        object.__setattr__(self, "split_penalties", penalties)
        object.__setattr__(self, "min_side_observations", min_sides)


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

    ranked = _rank_aggregates(
        aggregate_rows(aggregate_input), objective=sweep_config.objective
    )
    best_candidate = str(ranked[0]["approach"])
    ranks = {str(row["approach"]): int(row["component_sweep_rank"]) for row in ranked}
    objectives = {
        str(row["approach"]): float(row["component_sweep_objective"])
        for row in ranked
    }
    rows = _annotate_subject_rows(candidate_rows, best_candidate, ranks, objectives)
    if sweep_config.best_only:
        rows = [row for row in rows if int(row["component_sweep_best"]) == 1]
    return ComponentCleanupSweepOutput(
        tuple(rows), tuple(ranked), best_candidate, sweep_config.objective
    )


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
) -> list[dict[str, float | int | str]]:
    ranked = sorted(
        (
            {
                **row,
                "component_sweep_objective": _objective_value(row, objective),
                "component_sweep_objective_name": objective,
            }
            for row in rows
        ),
        key=lambda row: (
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


def _objective_value(
    row: Mapping[str, float | int | str], objective: ComponentSweepObjective
) -> float:
    if objective == "mean_micro_f1":
        return 0.5 * (
            float(row["complete_track_f1_micro"]) + float(row["pairwise_f1_micro"])
        )
    return float(row[objective])


def _annotate_subject_rows(
    candidate_rows: Sequence[tuple[str, ComponentCleanupConfig, Sequence[Mapping[str, Any]]]],
    best_candidate: str,
    ranks: Mapping[str, int],
    objectives: Mapping[str, float],
) -> list[dict[str, float | int | str]]:
    return [
        {
            **dict(row),
            "component_sweep_candidate": candidate,
            "component_sweep_rank": int(ranks[candidate]),
            "component_sweep_best": int(candidate == best_candidate),
            "component_sweep_objective": float(objectives[candidate]),
            "component_sweep_split_risk_threshold": float(
                cleanup_config.split_risk_threshold
            ),
            "component_sweep_split_penalty": float(cleanup_config.split_penalty),
            "component_sweep_min_side_observations": int(
                cleanup_config.min_side_observations
            ),
        }
        for candidate, cleanup_config, rows in candidate_rows
        for row in rows
    ]


def _finite_nonnegative_tuple(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in values)
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    if any(not math.isfinite(value) or value < 0.0 for value in normalized):
        raise ValueError(f"{name} entries must be finite non-negative values")
    return normalized
