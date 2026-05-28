from __future__ import annotations

import numpy as np

from bayescatrack.experiments.track2p_policy_component_audit import (
    ComponentCleanupConfig,
    _mark_applied_splits,
    apply_weakest_bridge_splits,
)
from bayescatrack.experiments.track2p_policy_gap_bridge_cleanup import (
    TRACK2P_POLICY_GAP_BRIDGE_CLEANUP_METHOD,
    build_arg_parser,
    gap_bridge_component_audit_rows,
)
from bayescatrack.experiments.track2p_policy_pruned_benchmark import (
    Track2pPolicyLinkDiagnostic,
)


def test_gap_bridge_cleanup_parser_defaults_to_gap_rescue() -> None:
    args = build_arg_parser().parse_args(["--data", "track2p-root"])

    assert args.max_gap == 2
    assert args.apply_splits is True
    assert args.require_complete_track is False
    assert args.min_side_observations == 1


def test_gap_bridge_cleanup_detects_rescued_skip_bridge() -> None:
    predicted = np.asarray([[4, -1, 9]], dtype=int)
    diagnostic = Track2pPolicyLinkDiagnostic(
        session_index=0,
        local_roi_a=0,
        local_roi_b=0,
        assigned_iou=0.51,
        threshold=0.50,
        threshold_margin=0.01,
        row_margin=0.0,
        column_margin=0.0,
        centroid_distance=12.0,
        area_ratio=0.30,
        pruned=False,
        prune_reason="",
    )
    config = ComponentCleanupConfig(
        min_side_observations=1,
        require_complete_track=False,
        split_risk_threshold=1.5,
    )

    rows = gap_bridge_component_audit_rows(
        predicted,
        {(0, 2, 4, 9): diagnostic},
        config=config,
    )

    assert rows[0]["n_edges"] == 1
    assert rows[0]["weakest_bridge_session_a"] == 0
    assert rows[0]["weakest_bridge_session_b"] == 2
    assert rows[0]["weakest_bridge_gap"] == 2
    assert rows[0]["would_split_at_weakest_edge"] == 1

    cleaned = apply_weakest_bridge_splits(
        predicted,
        _mark_applied_splits(rows, apply_splits=True),
    )

    assert cleaned.tolist() == [[4, -1, -1], [-1, -1, 9]]


def test_gap_bridge_cleanup_keeps_gap_when_no_diagnostic_support() -> None:
    predicted = np.asarray([[4, -1, 9]], dtype=int)

    rows = gap_bridge_component_audit_rows(
        predicted,
        {},
        config=ComponentCleanupConfig(
            min_side_observations=1,
            require_complete_track=False,
        ),
    )

    assert rows[0]["n_edges"] == 1
    assert rows[0]["weakest_bridge_risk"] == 0.0
    assert rows[0]["would_split_at_weakest_edge"] == 0
    assert TRACK2P_POLICY_GAP_BRIDGE_CLEANUP_METHOD in (
        "track2p-policy-gap-bridge-cleanup",
    )
