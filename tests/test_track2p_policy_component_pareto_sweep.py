from __future__ import annotations

import math

from bayescatrack.experiments.track2p_policy_component_pareto_sweep import (
    rank_component_sweep_aggregates,
    rerank_component_sweep_output,
)
from bayescatrack.experiments.track2p_policy_component_sweep import (
    NO_SPLIT_COMPONENT_CANDIDATE,
    ComponentCleanupSweepOutput,
)


def test_pareto_sweep_rank_rejects_complete_track_regression() -> None:
    rows = (
        _aggregate_row(NO_SPLIT_COMPONENT_CANDIDATE, pairwise=0.92, complete=0.80),
        _aggregate_row("component-cleanup-risky", pairwise=0.95, complete=0.70),
        _aggregate_row("component-cleanup-safe", pairwise=0.93, complete=0.82),
    )

    ranked = rank_component_sweep_aggregates(
        rows,
        objective="pairwise_f1_micro",
        pairwise_f1_floor_delta=0.0,
        complete_track_f1_floor_delta=0.0,
    )

    assert [row["approach"] for row in ranked] == [
        "component-cleanup-safe",
        NO_SPLIT_COMPONENT_CANDIDATE,
        "component-cleanup-risky",
    ]
    risky = next(row for row in ranked if row["approach"] == "component-cleanup-risky")
    assert risky["component_sweep_complete_floor_feasible"] == 0
    assert risky["component_sweep_baseline_safe_feasible"] == 0


def test_pareto_sweep_can_disable_complete_track_floor() -> None:
    rows = (
        _aggregate_row(NO_SPLIT_COMPONENT_CANDIDATE, pairwise=0.92, complete=0.80),
        _aggregate_row("component-cleanup-risky", pairwise=0.95, complete=0.70),
    )

    ranked = rank_component_sweep_aggregates(
        rows,
        objective="pairwise_f1_micro",
        pairwise_f1_floor_delta=0.0,
        complete_track_f1_floor_delta=None,
    )

    assert ranked[0]["approach"] == "component-cleanup-risky"
    assert ranked[0]["component_sweep_complete_track_f1_floor"] == ""


def test_pareto_sweep_ranker_places_nonfinite_candidates_last() -> None:
    rows = (
        _aggregate_row(NO_SPLIT_COMPONENT_CANDIDATE, pairwise=0.80, complete=0.80),
        _aggregate_row("component-cleanup-nan", pairwise=float("nan"), complete=0.90),
        _aggregate_row("component-cleanup-safe", pairwise=0.82, complete=0.82),
    )

    ranked = rank_component_sweep_aggregates(
        rows,
        objective="pairwise_f1_micro",
        pairwise_f1_floor_delta=0.0,
        complete_track_f1_floor_delta=0.0,
    )
    nan_row = next(row for row in ranked if row["approach"] == "component-cleanup-nan")

    assert ranked[0]["approach"] == "component-cleanup-safe"
    assert nan_row["component_sweep_pairwise_floor_feasible"] == 0
    assert nan_row["component_sweep_baseline_safe_feasible"] == 0
    assert math.isnan(float(nan_row["component_sweep_objective"]))


def test_pareto_sweep_ignores_nonfinite_baseline_floor_metric() -> None:
    rows = (
        _aggregate_row(
            NO_SPLIT_COMPONENT_CANDIDATE,
            pairwise=float("nan"),
            complete=0.80,
        ),
        _aggregate_row("component-cleanup-safe", pairwise=0.82, complete=0.82),
    )

    ranked = rank_component_sweep_aggregates(
        rows,
        objective="complete_track_f1_micro",
        pairwise_f1_floor_delta=0.0,
        complete_track_f1_floor_delta=0.0,
    )

    assert ranked[0]["approach"] == "component-cleanup-safe"
    assert ranked[0]["component_sweep_pairwise_f1_floor"] == ""
    assert ranked[0]["component_sweep_pairwise_floor_feasible"] == 1
    assert ranked[0]["component_sweep_complete_floor_feasible"] == 1


def test_pareto_sweep_rerank_updates_subject_rows() -> None:
    output = ComponentCleanupSweepOutput(
        rows=(
            {
                "subject": "jm000",
                "component_sweep_candidate": NO_SPLIT_COMPONENT_CANDIDATE,
                "component_sweep_rank": 1,
                "component_sweep_best": 1,
                "component_sweep_objective": 0.8,
            },
            {
                "subject": "jm000",
                "component_sweep_candidate": "component-cleanup-safe",
                "component_sweep_rank": 2,
                "component_sweep_best": 0,
                "component_sweep_objective": 0.82,
            },
        ),
        aggregate_rows=(
            _aggregate_row(NO_SPLIT_COMPONENT_CANDIDATE, pairwise=0.90, complete=0.80),
            _aggregate_row("component-cleanup-safe", pairwise=0.91, complete=0.82),
        ),
        best_candidate=NO_SPLIT_COMPONENT_CANDIDATE,
        objective="complete_track_f1_micro",
    )

    reranked = rerank_component_sweep_output(output)

    assert reranked.best_candidate == "component-cleanup-safe"
    assert (
        reranked.best_rows()[0]["component_sweep_candidate"] == "component-cleanup-safe"
    )
    assert reranked.best_rows()[0]["component_pareto_best"] == 1


def _aggregate_row(
    approach: str, *, pairwise: float, complete: float
) -> dict[str, float | int | str]:
    return {
        "approach": approach,
        "pairwise_f1_micro": pairwise,
        "complete_track_f1_micro": complete,
        "complete_track_f1_macro": complete,
    }
