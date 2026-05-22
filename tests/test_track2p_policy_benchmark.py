from pathlib import Path

from bayescatrack.experiments.track2p_benchmark import Track2pBenchmarkConfig
from bayescatrack.experiments.track2p_policy_benchmark import (
    TRACK2P_POLICY_DEFAULT_CELL_PROBABILITY_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_MAX_GAP,
    TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE,
    track2p_policy_config,
)


def test_track2p_policy_defaults_are_tuned_grid_winner() -> None:
    assert TRACK2P_POLICY_DEFAULT_TRANSFORM_TYPE == "affine"
    assert TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD == "min"
    assert TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD == 12.0
    assert TRACK2P_POLICY_DEFAULT_CELL_PROBABILITY_THRESHOLD == 0.5
    assert TRACK2P_POLICY_DEFAULT_MAX_GAP == 1


def test_track2p_policy_config_applies_hard_filtering_defaults() -> None:
    base = Track2pBenchmarkConfig(
        data=Path("data"),
        method="global-assignment",
        include_non_cells=True,
        weighted_masks=True,
        weighted_centroids=True,
        exclude_overlapping_pixels=True,
        max_gap=2,
        transform_type="fov-affine",
        cell_probability_threshold=0.0,
    )

    policy = track2p_policy_config(base)

    assert policy.method == "global-assignment"
    assert policy.transform_type == "affine"
    assert policy.max_gap == 1
    assert policy.include_non_cells is False
    assert policy.cell_probability_threshold == 0.5
    assert policy.weighted_masks is False
    assert policy.weighted_centroids is False
    assert policy.exclude_overlapping_pixels is False
