from __future__ import annotations

import math
from pathlib import Path

import pytest
from bayescatrack.experiments import track2p_policy_component_sweep as sweep_module
from bayescatrack.experiments.track2p_benchmark import (
    SubjectBenchmarkResult,
    Track2pBenchmarkConfig,
)
from bayescatrack.experiments.track2p_policy_component_audit import (
    ComponentAuditOutput,
)
from bayescatrack.experiments.track2p_policy_component_sweep import (
    ComponentCleanupSweepConfig,
    ComponentCleanupSweepOutput,
    run_track2p_policy_component_sweep,
)


def test_component_cleanup_sweep_marks_best_candidate(monkeypatch) -> None:
    calls = []

    def fake_component_audit(config, *, cleanup_config, **kwargs):
        calls.append(cleanup_config)
        if cleanup_config.split_risk_threshold == 1.0:
            complete_counts = (8, 2, 2)
            pairwise_counts = (9, 1, 1)
        else:
            complete_counts = (9, 1, 1)
            pairwise_counts = (8, 2, 2)
        return _sweep_output(pairwise_counts, complete_counts)

    monkeypatch.setattr(
        "bayescatrack.experiments.track2p_policy_component_sweep."
        "run_track2p_policy_component_audit",
        fake_component_audit,
    )

    output = run_track2p_policy_component_sweep(
        Track2pBenchmarkConfig(data=Path("unused"), method="global-assignment"),
        sweep_config=ComponentCleanupSweepConfig(
            split_risk_thresholds=(1.0, 2.0),
            split_penalties=(0.25,),
            require_complete_track_options=(True,),
            objective="complete_track_f1_micro",
            best_only=False,
            include_baseline=False,
        ),
    )

    assert len(calls) == 2
    assert output.best_candidate.endswith("risk2-penalty0.25-side2-complete")
    assert [row["component_sweep_rank"] for row in output.aggregate_rows] == [1, 2]
    best_rows = [row for row in output.rows if row["component_sweep_best"] == 1]
    assert len(best_rows) == 1
    assert best_rows[0]["component_sweep_split_risk_threshold"] == 2.0


def test_component_cleanup_sweep_best_only_filters_rows(monkeypatch) -> None:
    def fake_component_audit(config, *, cleanup_config, **kwargs):
        tp = 9 if cleanup_config.split_penalty == 0.5 else 8
        counts = (tp, 10 - tp, 10 - tp)
        return _sweep_output(counts, counts)

    monkeypatch.setattr(
        "bayescatrack.experiments.track2p_policy_component_sweep."
        "run_track2p_policy_component_audit",
        fake_component_audit,
    )

    output = run_track2p_policy_component_sweep(
        Track2pBenchmarkConfig(data=Path("unused"), method="global-assignment"),
        sweep_config=ComponentCleanupSweepConfig(
            split_risk_thresholds=(1.5,),
            split_penalties=(0.0, 0.5),
            require_complete_track_options=(True,),
            best_only=True,
            include_baseline=False,
        ),
    )

    assert len(output.rows) == 1
    assert output.rows[0]["component_sweep_best"] == 1
    assert output.rows[0]["component_sweep_split_penalty"] == 0.5
    assert len(output.aggregate_rows) == 2


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"split_risk_thresholds": (math.inf,)}, "finite non-negative"),
        ({"split_penalties": (math.nan,)}, "finite non-negative"),
        ({"min_side_observations": (1.5,)}, "positive integers"),
        ({"min_side_observations": (True,)}, "positive integers"),
        ({"require_complete_track_options": ("maybe",)}, "boolean"),
    ],
)
def test_component_cleanup_sweep_config_rejects_invalid_grid_entries(
    kwargs, message
) -> None:
    with pytest.raises(ValueError, match=message):
        ComponentCleanupSweepConfig(**kwargs)


def test_component_cleanup_sweep_config_rejects_unknown_objective() -> None:
    with pytest.raises(ValueError, match="objective must be one of"):
        ComponentCleanupSweepConfig(objective="not-a-metric")  # type: ignore[arg-type]


def test_component_cleanup_sweep_config_canonicalizes_valid_grid_entries() -> None:
    config = ComponentCleanupSweepConfig(
        split_risk_thresholds=(1, "2.5"),  # type: ignore[list-item]
        split_penalties=("0.0",),  # type: ignore[list-item]
        min_side_observations=("2",),  # type: ignore[list-item]
        require_complete_track_options=("true", "false", "1", "0"),  # type: ignore[list-item]
    )

    assert config.split_risk_thresholds == (1.0, 2.5)
    assert config.split_penalties == (0.0,)
    assert config.min_side_observations == (2,)
    assert config.require_complete_track_options == (True, False, True, False)


def test_component_cleanup_sweep_includes_no_split_baseline_by_default(
    monkeypatch,
) -> None:
    calls = []

    def fake_component_audit(config, *, cleanup_config, apply_splits, **kwargs):
        calls.append(apply_splits)
        counts = (10, 0, 0) if not apply_splits else (8, 2, 2)
        return _sweep_output(counts, counts)

    monkeypatch.setattr(
        "bayescatrack.experiments.track2p_policy_component_sweep."
        "run_track2p_policy_component_audit",
        fake_component_audit,
    )

    output = run_track2p_policy_component_sweep(
        Track2pBenchmarkConfig(data=Path("unused"), method="global-assignment"),
        sweep_config=ComponentCleanupSweepConfig(
            split_risk_thresholds=(1.5,),
            split_penalties=(0.25,),
            require_complete_track_options=(True,),
        ),
    )

    assert calls == [False, True]
    assert output.best_candidate == "component-cleanup-00-no-split"
    assert output.aggregate_rows[0]["component_sweep_pairwise_floor_feasible"] == 1


def test_component_cleanup_sweep_pairwise_floor_rejects_fp_heavy_cleanup(
    monkeypatch,
) -> None:
    def fake_component_audit(config, *, cleanup_config, apply_splits, **kwargs):
        if not apply_splits:
            return _sweep_output(
                pairwise_counts=(10, 0, 0),
                complete_counts=(8, 2, 2),
            )
        return _sweep_output(
            pairwise_counts=(8, 12, 0),
            complete_counts=(9, 1, 1),
        )

    monkeypatch.setattr(
        "bayescatrack.experiments.track2p_policy_component_sweep."
        "run_track2p_policy_component_audit",
        fake_component_audit,
    )

    output = run_track2p_policy_component_sweep(
        Track2pBenchmarkConfig(data=Path("unused"), method="global-assignment"),
        sweep_config=ComponentCleanupSweepConfig(
            split_risk_thresholds=(1.5,),
            split_penalties=(0.25,),
            require_complete_track_options=(True,),
            objective="complete_track_f1_micro",
        ),
    )

    assert output.best_candidate == "component-cleanup-00-no-split"
    aggregate_by_candidate = {
        str(row["approach"]): row for row in output.aggregate_rows
    }
    assert (
        aggregate_by_candidate[
            "component-cleanup-01-risk1.5-penalty0.25-side2-complete"
        ]["component_sweep_pairwise_floor_feasible"]
        == 0
    )


def test_component_cleanup_sweep_can_select_partial_track_guard(monkeypatch) -> None:
    calls = []

    def fake_component_audit(config, *, cleanup_config, **kwargs):
        calls.append(cleanup_config)
        if cleanup_config.require_complete_track:
            counts = (7, 3, 3)
        else:
            counts = (9, 1, 1)
        return _sweep_output(counts, counts)

    monkeypatch.setattr(
        "bayescatrack.experiments.track2p_policy_component_sweep."
        "run_track2p_policy_component_audit",
        fake_component_audit,
    )

    output = run_track2p_policy_component_sweep(
        Track2pBenchmarkConfig(data=Path("unused"), method="global-assignment"),
        sweep_config=ComponentCleanupSweepConfig(
            split_risk_thresholds=(1.5,),
            split_penalties=(0.25,),
            require_complete_track_options=(True, False),
            objective="complete_track_f1_micro",
            include_baseline=False,
        ),
    )

    assert [call.require_complete_track for call in calls] == [True, False]
    assert output.best_candidate.endswith("-partial")
    best_rows = output.best_rows()
    assert len(best_rows) == 1
    assert best_rows[0]["component_sweep_require_complete_track"] == 0


def test_component_cleanup_sweep_cli_legacy_guard_flag_is_honored(
    monkeypatch, tmp_path
) -> None:
    captured: dict[str, tuple[bool, ...]] = {}

    def fake_component_sweep(config, *, sweep_config, **kwargs):
        captured["require_complete_track_options"] = (
            sweep_config.require_complete_track_options
        )
        return ComponentCleanupSweepOutput(
            rows=(),
            aggregate_rows=(),
            best_candidate="none",
            objective=sweep_config.objective,
        )

    monkeypatch.setattr(
        sweep_module, "run_track2p_policy_component_sweep", fake_component_sweep
    )
    monkeypatch.setattr(sweep_module, "write_results", lambda *args, **kwargs: None)

    assert (
        sweep_module.main(
            [
                "--data",
                str(tmp_path),
                "--output",
                str(tmp_path / "rows.csv"),
                "--no-require-complete-track",
            ]
        )
        == 0
    )

    assert captured["require_complete_track_options"] == (False,)


def test_component_cleanup_sweep_cli_accepts_guard_options(
    monkeypatch, tmp_path
) -> None:
    captured: dict[str, tuple[bool, ...]] = {}

    def fake_component_sweep(config, *, sweep_config, **kwargs):
        captured["require_complete_track_options"] = (
            sweep_config.require_complete_track_options
        )
        return ComponentCleanupSweepOutput(
            rows=(),
            aggregate_rows=(),
            best_candidate="none",
            objective=sweep_config.objective,
        )

    monkeypatch.setattr(
        sweep_module, "run_track2p_policy_component_sweep", fake_component_sweep
    )
    monkeypatch.setattr(sweep_module, "write_results", lambda *args, **kwargs: None)

    assert (
        sweep_module.main(
            [
                "--data",
                str(tmp_path),
                "--output",
                str(tmp_path / "rows.csv"),
                "--require-complete-track-options",
                "false,true",
            ]
        )
        == 0
    )

    assert captured["require_complete_track_options"] == (False, True)


def test_component_cleanup_sweep_cli_rejects_conflicting_guard_options(
    tmp_path,
) -> None:
    with pytest.raises(SystemExit):
        sweep_module.main(
            [
                "--data",
                str(tmp_path),
                "--output",
                str(tmp_path / "rows.csv"),
                "--require-complete-track-options",
                "true,false",
                "--no-require-complete-track",
            ]
        )


def _sweep_output(
    pairwise_counts: tuple[int, int, int],
    complete_counts: tuple[int, int, int],
) -> ComponentAuditOutput:
    return ComponentAuditOutput(
        (
            SubjectBenchmarkResult(
                subject="jm000",
                variant="synthetic",
                method="global-assignment",
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
