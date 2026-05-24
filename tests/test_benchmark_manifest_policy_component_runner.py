from pathlib import Path

from bayescatrack.experiments.benchmark_manifest import (
    _run_config,
    _runner_kwargs,
    _runner_name,
)


def test_policy_component_runner_aliases_are_supported() -> None:
    assert (
        _runner_name("track2p-policy-component-audit")
        == "track2p-policy-component-audit"
    )
    assert _runner_name("track2p-component-cleanup") == (
        "track2p-policy-component-audit"
    )


def test_policy_component_runner_config_defaults_method() -> None:
    config = _run_config(
        "track2p-policy-component-audit",
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
    assert config.include_non_cells is False
    assert config.weighted_masks is False
    assert config.weighted_centroids is False
    assert config.exclude_overlapping_pixels is False


def test_policy_component_runner_kwargs_are_runner_specific() -> None:
    assert _runner_kwargs(
        {
            "threshold_method": "min",
            "iou_distance_threshold": 12.0,
            "cell_probability_threshold": 0.5,
            "apply_splits": True,
            "split_risk_threshold": 1.5,
            "min_side_observations": 2,
        },
        "track2p-policy-component-audit",
    ) == {
        "threshold_method": "min",
        "iou_distance_threshold": 12.0,
        "apply_splits": True,
        "split_risk_threshold": 1.5,
        "min_side_observations": 2,
    }
