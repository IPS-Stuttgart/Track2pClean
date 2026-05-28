from __future__ import annotations

from pathlib import Path

from bayescatrack.experiments import (
    track2p_policy_gap_consensus_guarded_sweep as guarded,
)
from bayescatrack.experiments.track2p_benchmark import Track2pBenchmarkConfig


def test_guarded_config_includes_adjacent_and_gap_rescue_defaults() -> None:
    config = guarded.guarded_gap_consensus_sweep_config(
        base_iou_distance_thresholds=(12.0,),
        split_risk_thresholds=(1.5,),
        split_penalties=(0.25,),
        require_complete_track_options=(True,),
        max_splits_per_component=(1,),
    )

    assert config.max_gaps == (1, 2)


def test_guarded_sweep_uses_guarded_default_config(monkeypatch) -> None:
    captured = {}

    def fake_sweep(config, **kwargs):
        captured["config"] = config
        captured["sweep_config"] = kwargs["sweep_config"]
        return "sentinel"

    monkeypatch.setattr(guarded, "run_track2p_policy_gap_consensus_sweep", fake_sweep)

    result = guarded.run_track2p_policy_gap_consensus_guarded_sweep(
        Track2pBenchmarkConfig(data=Path("unused"), method="global-assignment")
    )

    assert result == "sentinel"
    assert captured["sweep_config"].max_gaps == (1, 2)


def test_guarded_sweep_respects_explicit_sweep_config(monkeypatch) -> None:
    explicit = guarded.guarded_gap_consensus_sweep_config(max_gaps=(3,))
    captured = {}

    def fake_sweep(config, **kwargs):
        captured["sweep_config"] = kwargs["sweep_config"]
        return "sentinel"

    monkeypatch.setattr(guarded, "run_track2p_policy_gap_consensus_sweep", fake_sweep)

    result = guarded.run_track2p_policy_gap_consensus_guarded_sweep(
        Track2pBenchmarkConfig(data=Path("unused"), method="global-assignment"),
        sweep_config=explicit,
    )

    assert result == "sentinel"
    assert captured["sweep_config"] is explicit
    assert captured["sweep_config"].max_gaps == (3,)


def test_guarded_main_injects_max_gaps_by_default(monkeypatch) -> None:
    captured = {}

    def fake_main(argv):
        captured["argv"] = tuple(argv)
        return 0

    monkeypatch.setattr(guarded, "_sweep_main", fake_main)

    assert guarded.main(["--data", "unused"]) == 0
    assert captured["argv"] == ("--data", "unused", "--max-gaps", "1,2")


def test_guarded_main_respects_explicit_max_gaps(monkeypatch) -> None:
    captured = {}

    def fake_main(argv):
        captured["argv"] = tuple(argv)
        return 0

    monkeypatch.setattr(guarded, "_sweep_main", fake_main)

    assert guarded.main(["--data", "unused", "--max-gaps=3"]) == 0
    assert captured["argv"] == ("--data", "unused", "--max-gaps=3")
