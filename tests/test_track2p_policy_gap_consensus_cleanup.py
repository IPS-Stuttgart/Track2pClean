from __future__ import annotations

from bayescatrack.experiments.track2p_policy_gap_consensus_cleanup import (
    TRACK2P_POLICY_GAP_CONSENSUS_DEFAULT_MAX_GAP,
    build_arg_parser,
)


def test_gap_consensus_cleanup_parser_defaults_to_gap_rescue() -> None:
    args = build_arg_parser().parse_args(["--data", "track2p-root"])

    assert args.max_gap == TRACK2P_POLICY_GAP_CONSENSUS_DEFAULT_MAX_GAP
    assert args.apply_splits is True
    assert args.require_complete_track is True
    assert args.consensus_mode == "risk-and-stability"


def test_gap_consensus_cleanup_parser_accepts_audit_mode_and_gap_override() -> None:
    args = build_arg_parser().parse_args(
        [
            "--data",
            "track2p-root",
            "--max-gap",
            "3",
            "--no-apply-splits",
            "--no-require-complete-track",
            "--stability-iou-distance-thresholds",
            "8,12,16",
        ]
    )

    assert args.max_gap == 3
    assert args.apply_splits is False
    assert args.require_complete_track is False
    assert args.stability_iou_distance_thresholds == (8.0, 12.0, 16.0)
