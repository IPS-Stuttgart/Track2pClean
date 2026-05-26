from __future__ import annotations

from bayescatrack.experiments.track2p_policy_gap_consensus_cleanup import (
    TRACK2P_POLICY_GAP_CONSENSUS_DEFAULT_MAX_GAP,
    build_arg_parser,
)
from bayescatrack.experiments.track2p_policy_benchmark import (
    TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
    TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
)


def test_gap_consensus_cleanup_parser_defaults_to_gap_rescue_consensus() -> None:
    args = build_arg_parser().parse_args(["--data", "track2p-root"])

    assert args.max_gap == TRACK2P_POLICY_GAP_CONSENSUS_DEFAULT_MAX_GAP
    assert args.threshold_method == TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD
    assert args.base_iou_distance_threshold == TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD
    assert args.apply_splits is True
    assert args.require_complete_track is True
    assert args.consensus_mode == "risk-and-stability"


def test_gap_consensus_cleanup_parser_accepts_support_threshold_tuple() -> None:
    args = build_arg_parser().parse_args(
        [
            "--data",
            "track2p-root",
            "--stability-iou-distance-thresholds",
            "8,12,16",
            "--min-support-votes",
            "2",
            "--no-apply-splits",
        ]
    )

    assert args.stability_iou_distance_thresholds == (8.0, 12.0, 16.0)
    assert args.min_support_votes == 2
    assert args.apply_splits is False
