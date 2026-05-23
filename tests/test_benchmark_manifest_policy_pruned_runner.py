from pathlib import Path

from bayescatrack.experiments.benchmark_manifest import (
    _run_config,
    _runner_kwargs,
    _runner_name,
)


def test_policy_pruned_runner_alias_is_supported() -> None:
    assert _runner_name("track2p-policy-pruned") == "track2p-policy-pruned"


def test_policy_pruned_runner_config_defaults_method() -> None:
    config = _run_config(
        "track2p-policy-pruned",
        {
            "data": "data-root",
            "reference": "gt-root",
            "reference_kind": "manual-gt",
        },
        base_dir=Path("/tmp/benchmark"),
    )

    assert config.method == "global-assignment"
    assert config.data == Path("/tmp/benchmark/data-root")
    assert config.reference == Path("/tmp/benchmark/gt-root")


def test_policy_pruned_runner_kwargs_are_runner_specific() -> None:
    assert _runner_kwargs(
        {
            "threshold_method": "min",
            "iou_distance_threshold": 12.0,
            "prune_threshold_margin": 0.02,
            "prune_competition_margin": 0.02,
            "cell_probability_threshold": 0.5,
        },
        "track2p-policy-pruned",
    ) == {
        "threshold_method": "min",
        "iou_distance_threshold": 12.0,
        "prune_threshold_margin": 0.02,
        "prune_competition_margin": 0.02,
    }
