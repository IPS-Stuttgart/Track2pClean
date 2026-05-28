from __future__ import annotations

import numpy as np
from bayescatrack.experiments.track2p_policy_supported_gap_component_cleanup import (
    TRACK2P_POLICY_SUPPORTED_GAP_DEFAULT_MIN_BRIDGE_SUPPORT,
    build_arg_parser,
    filter_gap_links_by_bridge_support,
)


def test_supported_gap_filter_keeps_adjacent_links_and_supported_skip() -> None:
    links_by_gap = {
        (0, 1): np.asarray([[0, 2]], dtype=int),
        (1, 1): np.asarray([[7, 3]], dtype=int),
        (0, 2): np.asarray([[0, 3], [1, 4]], dtype=int),
    }

    filtered = filter_gap_links_by_bridge_support(
        links_by_gap, max_gap=2, min_bridge_support=1
    )

    np.testing.assert_array_equal(filtered[(0, 1)], [[0, 2]])
    np.testing.assert_array_equal(filtered[(1, 1)], [[7, 3]])
    np.testing.assert_array_equal(filtered[(0, 2)], [[0, 3]])


def test_supported_gap_filter_can_require_both_endpoint_support() -> None:
    links_by_gap = {
        (0, 1): np.asarray([[0, 2], [1, 5]], dtype=int),
        (1, 1): np.asarray([[8, 3]], dtype=int),
        (0, 2): np.asarray([[0, 3], [1, 4]], dtype=int),
    }

    filtered = filter_gap_links_by_bridge_support(
        links_by_gap, max_gap=2, min_bridge_support=2
    )

    np.testing.assert_array_equal(filtered[(0, 2)], [[0, 3]])


def test_supported_gap_filter_zero_support_recovers_raw_gap_rescue() -> None:
    links_by_gap = {
        (0, 1): np.zeros((0, 2), dtype=int),
        (1, 1): np.zeros((0, 2), dtype=int),
        (0, 2): np.asarray([[1, 4]], dtype=int),
    }

    filtered = filter_gap_links_by_bridge_support(
        links_by_gap, max_gap=2, min_bridge_support=0
    )

    np.testing.assert_array_equal(filtered[(0, 2)], [[1, 4]])


def test_supported_gap_parser_defaults_to_one_bridge_support_vote() -> None:
    args = build_arg_parser().parse_args(["--data", "track2p-root"])

    assert (
        args.min_bridge_support
        == TRACK2P_POLICY_SUPPORTED_GAP_DEFAULT_MIN_BRIDGE_SUPPORT
    )
    assert args.min_bridge_support == 1
