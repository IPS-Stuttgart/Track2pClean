from __future__ import annotations

from bayescatrack.experiments.track2p_policy_gap_component_cleanup import (
    TRACK2P_POLICY_GAP_COMPONENT_DEFAULT_MAX_GAP,
    build_arg_parser,
)


def test_gap_component_cleanup_parser_defaults_to_gap_rescue() -> None:
    args = build_arg_parser().parse_args(["--data", "track2p-root"])

    assert args.max_gap == TRACK2P_POLICY_GAP_COMPONENT_DEFAULT_MAX_GAP
    assert args.apply_splits is True
    assert args.require_complete_track is True
