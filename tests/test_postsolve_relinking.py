from __future__ import annotations

import numpy as np
from bayescatrack.association.postsolve_relinking import (
    PostSolveRelinkingConfig,
    relink_tracks_at_geometry_issues,
)
from bayescatrack.association.track_refinement import TrackGeometryIssue


def _issue() -> TrackGeometryIssue:
    return TrackGeometryIssue(
        track_index=0,
        session_index=1,
        roi_index=21,
        residual=9.0,
        robust_z=4.0,
        suggested_action="split_or_relink",
    )


def test_relink_uses_outgoing_edge_when_available():
    rows = np.array([[10, 21, 30]], dtype=int)
    roi_indices_by_session = ([10], [20, 21, 22], [30])
    pairwise_costs = {
        (0, 1): np.array([[0.1, 0.2, 0.4]], dtype=float),
        (1, 2): np.array([[10.0], [8.0], [0.1]], dtype=float),
    }

    relinked = relink_tracks_at_geometry_issues(
        rows,
        [_issue()],
        pairwise_costs,
        roi_indices_by_session=roi_indices_by_session,
        config=PostSolveRelinkingConfig(
            max_edge_cost=None,
            min_cost_improvement=0.0,
        ),
    )

    assert relinked.tolist() == [[10, 22, 30]]


def test_relink_can_disable_outgoing_edge_evidence_for_legacy_behavior():
    rows = np.array([[10, 21, 30]], dtype=int)
    roi_indices_by_session = ([10], [20, 21, 22], [30])
    pairwise_costs = {
        (0, 1): np.array([[0.1, 0.2, 0.4]], dtype=float),
        (1, 2): np.array([[10.0], [8.0], [0.1]], dtype=float),
    }

    relinked = relink_tracks_at_geometry_issues(
        rows,
        [_issue()],
        pairwise_costs,
        roi_indices_by_session=roi_indices_by_session,
        config=PostSolveRelinkingConfig(
            max_edge_cost=None,
            min_cost_improvement=0.0,
            bidirectional_next_weight=0.0,
        ),
    )

    assert relinked.tolist() == [[10, 20, 30]]
