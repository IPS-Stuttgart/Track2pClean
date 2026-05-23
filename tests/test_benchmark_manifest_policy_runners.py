from pathlib import Path

from bayescatrack.experiments.benchmark_manifest import (
    _run_config,
    _runner_kwargs,
    _runner_name,
)


def test_policy_runner_aliases_are_supported() -> None:
    assert _runner_name("track2p-policy") == "track2p-policy"
    assert _runner_name("track2p-policy-dp") == "track2p-policy-dp"


def test_policy_runner_config_defaults_method() -> None:
    config = _run_config(
        "track2p-policy",
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


def test_policy_runner_kwargs_are_separate_from_track2p_config() -> None:
    assert _runner_kwargs(
        {
            "threshold_method": "min",
            "iou_distance_threshold": 12.0,
            "cell_probability_threshold": 0.5,
        },
        "track2p-policy",
    ) == {
        "threshold_method": "min",
        "iou_distance_threshold": 12.0,
    }


def test_policy_dp_runner_kwargs_include_only_dp_specific_options() -> None:
    assert _runner_kwargs(
        {
            "threshold_method": "min",
            "iou_distance_threshold": 12.0,
            "row_top_k": 2,
            "rescue_min_iou": 0.1,
            "max_gap": 2,
        },
        "track2p-policy-dp",
    ) == {
        "threshold_method": "min",
        "iou_distance_threshold": 12.0,
        "row_top_k": 2,
        "rescue_min_iou": 0.1,
    }
