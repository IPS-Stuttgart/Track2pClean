from pathlib import Path

import pytest
from bayescatrack.experiments.benchmark_manifest import (
    _run_config,
    _runner_kwargs,
    _runner_name,
)
from bayescatrack.experiments.track2p_benchmark import Track2pBenchmarkConfig
from bayescatrack.experiments.track2p_policy_benchmark import track2p_policy_config


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


def test_policy_config_inherits_configured_max_gap() -> None:
    config = _run_config(
        "track2p-policy",
        {
            "data": "data-root",
            "reference": "gt-root",
            "reference_kind": "manual-gt",
            "max_gap": 3,
        },
        base_dir=Path("/tmp/benchmark"),
    )

    policy_config = track2p_policy_config(config)

    assert config.max_gap == 3
    assert policy_config.max_gap == 3


def test_policy_config_allows_explicit_max_gap_override() -> None:
    config = _run_config(
        "track2p-policy",
        {
            "data": "data-root",
            "reference": "gt-root",
            "reference_kind": "manual-gt",
            "max_gap": 3,
        },
        base_dir=Path("/tmp/benchmark"),
    )

    policy_config = track2p_policy_config(config, max_gap=1)

    assert policy_config.max_gap == 1


@pytest.mark.parametrize(
    "max_gap",
    [True, False, 0, -1, 1.5, "2", float("nan"), float("inf")],
)
def test_policy_config_rejects_invalid_inherited_max_gap(max_gap: object) -> None:
    config = Track2pBenchmarkConfig(
        data=Path("data-root"),
        method="global-assignment",
        max_gap=max_gap,  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="max_gap"):
        track2p_policy_config(config)


@pytest.mark.parametrize(
    "max_gap",
    [True, False, 0, -1, 1.5, "2", float("nan"), float("inf")],
)
def test_policy_config_rejects_invalid_explicit_max_gap(max_gap: object) -> None:
    config = Track2pBenchmarkConfig(data=Path("data-root"), method="global-assignment")

    with pytest.raises(ValueError, match="max_gap"):
        track2p_policy_config(config, max_gap=max_gap)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "cell_probability_threshold",
    [True, False, float("nan"), float("inf"), -0.1, 1.1],
)
def test_policy_config_rejects_invalid_cell_probability_threshold(
    cell_probability_threshold: float | bool,
) -> None:
    config = Track2pBenchmarkConfig(data=Path("data-root"), method="global-assignment")

    with pytest.raises(ValueError, match="cell_probability_threshold"):
        track2p_policy_config(
            config,
            cell_probability_threshold=cell_probability_threshold,
        )


def test_policy_runner_kwargs_are_separate_from_track2p_config() -> None:
    assert _runner_kwargs(
        {
            "threshold_method": "min",
            "iou_distance_threshold": 12.0,
            "cell_probability_threshold": 0.5,
            "max_gap": 3,
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
