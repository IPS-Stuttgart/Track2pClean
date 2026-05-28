from __future__ import annotations

import math
from pathlib import Path

import pytest
from bayescatrack.experiments import track2p_policy_gap_consensus_sweep as sweep_module
from bayescatrack.experiments.track2p_benchmark import (
    SubjectBenchmarkResult,
    Track2pBenchmarkConfig,
)
from bayescatrack.experiments.track2p_policy_component_audit import ComponentAuditOutput
from bayescatrack.experiments.track2p_policy_gap_consensus_sweep import (
    GapConsensusSweepConfig,
    run_track2p_policy_gap_consensus_sweep,
)


def test_gap_consensus_sweep_marks_best_candidate(monkeypatch) -> None:
    calls = []

    def fake_gap_consensus(config, *, cleanup_config, max_gap, **kwargs):
        calls.append((cleanup_config, max_gap))
        tp = 9 if cleanup_config.stability.base_iou_distance_threshold == 14.0 else 8
        counts = (tp, 10 - tp, 10 - tp)
        return _sweep_output(counts, counts)

    monkeypatch.setattr(
        sweep_module, "run_track2p_policy_gap_consensus_cleanup", fake_gap_consensus
    )

    output = run_track2p_policy_gap_consensus_sweep(
        Track2pBenchmarkConfig(data=Path("unused"), method="global-assignment"),
        sweep_config=GapConsensusSweepConfig(
            base_iou_distance_thresholds=(12.0, 14.0),
            split_risk_thresholds=(1.5,),
            split_penalties=(0.25,),
            require_complete_track_options=(True,),
            max_splits_per_component=(1,),
            max_gaps=(2,),
        ),
    )

    assert len(calls) == 2
    assert "base14" in output.best_candidate
    assert [row["gap_consensus_sweep_rank"] for row in output.aggregate_rows] == [1, 2]
    assert (
        output.best_rows()[0]["gap_consensus_sweep_base_iou_distance_threshold"] == 14.0
    )
    assert output.best_rows()[0]["gap_consensus_sweep_max_gap"] == 2


def test_gap_consensus_sweep_best_only_filters_rows(monkeypatch) -> None:
    def fake_gap_consensus(config, *, cleanup_config, **kwargs):
        tp = 9 if cleanup_config.component.split_penalty == 0.5 else 8
        counts = (tp, 10 - tp, 10 - tp)
        return _sweep_output(counts, counts)

    monkeypatch.setattr(
        sweep_module, "run_track2p_policy_gap_consensus_cleanup", fake_gap_consensus
    )

    output = run_track2p_policy_gap_consensus_sweep(
        Track2pBenchmarkConfig(data=Path("unused"), method="global-assignment"),
        sweep_config=GapConsensusSweepConfig(
            base_iou_distance_thresholds=(12.0,),
            split_risk_thresholds=(1.5,),
            split_penalties=(0.0, 0.5),
            require_complete_track_options=(True,),
            max_splits_per_component=(1,),
            best_only=True,
        ),
    )

    assert len(output.rows) == 1
    assert output.rows[0]["gap_consensus_sweep_best"] == 1
    assert output.rows[0]["gap_consensus_sweep_split_penalty"] == 0.5
    assert len(output.aggregate_rows) == 2


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"base_iou_distance_thresholds": (math.inf,)}, "finite non-negative"),
        ({"split_risk_thresholds": (math.nan,)}, "finite non-negative"),
        ({"min_side_observations": (1.5,)}, "positive integers"),
        ({"require_complete_track_options": ("maybe",)}, "boolean"),
        ({"max_gaps": (0,)}, "positive integers"),
        ({"consensus_modes": ("unsupported",)}, "unsupported consensus mode"),
        ({"min_support_fraction": 0.0}, "lie in"),
    ],
)
def test_gap_consensus_sweep_config_rejects_invalid_grid_entries(
    kwargs, message
) -> None:
    with pytest.raises(ValueError, match=message):
        GapConsensusSweepConfig(**kwargs)


def _sweep_output(
    pairwise_counts: tuple[int, int, int], complete_counts: tuple[int, int, int]
) -> ComponentAuditOutput:
    return ComponentAuditOutput(
        (
            SubjectBenchmarkResult(
                subject="jm000",
                variant="synthetic",
                method="track2p-policy-gap-consensus-cleanup",
                scores={
                    "pairwise_f1": _f1(*pairwise_counts),
                    "pairwise_true_positives": pairwise_counts[0],
                    "pairwise_false_positives": pairwise_counts[1],
                    "pairwise_false_negatives": pairwise_counts[2],
                    "complete_track_f1": _f1(*complete_counts),
                    "complete_track_true_positives": complete_counts[0],
                    "complete_track_false_positives": complete_counts[1],
                    "complete_track_false_negatives": complete_counts[2],
                },
                n_sessions=4,
                reference_source="ground_truth_csv",
            ),
        ),
        (),
    )


def _f1(tp: int, fp: int, fn: int) -> float:
    return 2.0 * tp / (2 * tp + fp + fn)
