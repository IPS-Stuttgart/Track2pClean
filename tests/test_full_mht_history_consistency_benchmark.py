from __future__ import annotations

from bayescatrack.experiments import track2p_policy_full_mht_benchmark as base
from bayescatrack.experiments import (
    track2p_policy_full_mht_history_consistency_benchmark as wrapper,
)


def test_history_consistency_args_are_stripped_before_base_runner() -> None:
    config, remaining = wrapper._history_config_from_args(
        [
            "--history-consistency-weight",
            "2.5",
            "--history-consistency-min-history-edges",
            "3",
            "--history-consistency-min-feature-scale",
            "0.1",
            "--history-consistency-joint-margin",
            "0.75",
            "--history-consistency-score-clip",
            "4.0",
            "--data",
            "data-root",
            "--output",
            "out.csv",
        ]
    )

    assert config.weight == 2.5
    assert config.min_history_edges == 3
    assert config.min_feature_scale == 0.1
    assert config.joint_margin == 0.75
    assert config.score_clip == 4.0
    assert remaining == ["--data", "data-root", "--output", "out.csv"]


def test_history_consistency_wrapper_patches_and_restores_base_runner() -> None:
    original_method = base.METHOD
    original_expand = base._expand_hypothesis_scan
    config, _remaining = wrapper._history_config_from_args(
        ["--history-consistency-weight", "1.0"]
    )

    with wrapper._patched_full_mht_runner(config):
        assert base.METHOD == wrapper.METHOD
        assert base._expand_hypothesis_scan is (
            wrapper._expand_hypothesis_scan_with_history_consistency
        )

    assert base.METHOD == original_method
    assert base._expand_hypothesis_scan is original_expand


def test_history_consistency_main_delegates_clean_args(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_main(args: list[str]) -> int:
        seen["args"] = list(args)
        seen["method"] = base.METHOD
        seen["expand"] = base._expand_hypothesis_scan
        return 17

    monkeypatch.setattr(base, "main", fake_main)

    status = wrapper.main(
        [
            "--history-consistency-weight",
            "1.25",
            "--data",
            "data-root",
            "--output",
            "out.csv",
        ]
    )

    assert status == 17
    assert seen["args"] == ["--data", "data-root", "--output", "out.csv"]
    assert seen["method"] == wrapper.METHOD
    assert seen["expand"] is wrapper._expand_hypothesis_scan_with_history_consistency
    assert base.METHOD != wrapper.METHOD
