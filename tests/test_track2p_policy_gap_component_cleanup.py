from __future__ import annotations

from bayescatrack.experiments.track2p_policy_gap_component_cleanup import (
    TRACK2P_POLICY_GAP_COMPONENT_DEFAULT_MAX_GAP,
    TRACK2P_POLICY_GAP_COMPONENT_DEFAULT_MIN_SIDE_OBSERVATIONS,
    TRACK2P_POLICY_GAP_COMPONENT_DEFAULT_REQUIRE_COMPLETE_TRACK,
    _default_gap_cleanup_config,
    _no_prune_config,
    build_arg_parser,
)
from bayescatrack.experiments.track2p_policy_pruned_benchmark import (
    should_prune_policy_edge,
)


def test_gap_component_cleanup_parser_defaults_to_gap_rescue() -> None:
    args = build_arg_parser().parse_args(["--data", "track2p-root"])

    assert args.max_gap == TRACK2P_POLICY_GAP_COMPONENT_DEFAULT_MAX_GAP
    assert args.apply_splits is True
    assert args.require_complete_track is (
        TRACK2P_POLICY_GAP_COMPONENT_DEFAULT_REQUIRE_COMPLETE_TRACK
    )
    assert args.require_complete_track is False
    assert args.min_side_observations == (
        TRACK2P_POLICY_GAP_COMPONENT_DEFAULT_MIN_SIDE_OBSERVATIONS
    )
    assert args.min_side_observations == 1


def test_default_gap_cleanup_config_allows_incomplete_rescued_splits() -> None:
    config = _default_gap_cleanup_config()

    assert config.require_complete_track is False
    assert config.min_side_observations == 1


def test_gap_component_no_prune_config_keeps_weak_geometry_edges() -> None:
    config = _no_prune_config()

    assert not should_prune_policy_edge(
        assigned_iou=0.01,
        threshold=0.0,
        row_margin=0.0,
        column_margin=0.0,
        area_ratio=0.0,
        centroid_distance=1.0e9,
        config=config,
    )
