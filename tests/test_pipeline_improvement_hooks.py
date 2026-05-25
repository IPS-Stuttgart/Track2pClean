from __future__ import annotations

import numpy as np
from bayescatrack.association.absence_model import apply_absence_adjustment
from bayescatrack.association.consensus_priors import (
    apply_consensus_edge_priors,
    edge_votes_from_tracks,
)
from bayescatrack.association.joint_registration_assignment import (
    apply_joint_anchor_relief_to_pairwise_costs,
)
from bayescatrack.association.postsolve_relinking import (
    relink_tracks_at_geometry_issues,
)
from bayescatrack.association.track_refinement import TrackGeometryIssue


class _Plane:
    def __init__(self, n_rois: int):
        self.n_rois = n_rois
        self.cell_probabilities = None
        self.traces = np.ones((n_rois, 3), dtype=float)
        self.spike_traces = None


def test_absence_adjustment_uses_registered_empty_mask_discount():
    costs = np.zeros((2, 2), dtype=float)
    adjusted = apply_absence_adjustment(
        costs,
        _Plane(2),
        _Plane(2),
        session_gap=2,
        registered_empty_mask=np.array([False, True]),
        config={
            "base_absence_cost": 1.0,
            "empty_registered_mask_discount": 0.5,
            "trace_missing_discount": 0.0,
        },
    )

    assert adjusted[0, 1] < adjusted[0, 0]


def test_joint_anchor_relief_reduces_mutual_low_cost_edges():
    pairwise = {(0, 1): np.array([[0.1, 4.0], [4.0, 0.2]], dtype=float)}

    adjusted = apply_joint_anchor_relief_to_pairwise_costs(
        pairwise,
        config={
            "min_anchor_edges": 1,
            "high_confidence_quantile": 0.5,
            "cost_relief": 0.05,
        },
    )

    assert adjusted[(0, 1)][0, 0] < pairwise[(0, 1)][0, 0]


def test_consensus_prior_relieves_edges_with_enough_votes():
    votes = edge_votes_from_tracks(
        [
            [{0: 0, 1: 1}],
            [{0: 0, 1: 1}],
            [{0: 0, 1: 0}],
        ],
        session_edges=((0, 1),),
    )
    pairwise = {(0, 1): np.array([[1.0, 2.0]], dtype=float)}

    adjusted = apply_consensus_edge_priors(
        pairwise,
        votes,
        config={"min_votes": 2, "relief": 0.25},
    )

    assert adjusted[(0, 1)][0, 1] == 1.5
    assert adjusted[(0, 1)][0, 0] == 1.0


def test_consensus_votes_ignore_missing_negative_roi_entries():
    votes = edge_votes_from_tracks(
        [
            [{0: 0, 1: -1, 2: 5}],
            [{0: 0, 1: -1, 2: 5}],
        ],
        session_edges=((0, 1), (1, 2), (0, 2)),
    )

    assert (0, 1, 0, -1) not in votes
    assert (1, 2, -1, 5) not in votes
    assert votes == {
        (0, 2, 0, 5): 2,
    }


def test_postsolve_relinking_replaces_flagged_detection_with_better_candidate():
    tracks = np.array([[0, 5], [1, 6]], dtype=int)
    issues = [
        TrackGeometryIssue(
            track_index=0,
            session_index=1,
            roi_index=5,
            residual=10.0,
            robust_z=4.0,
            suggested_action="split_or_relink",
        )
    ]
    pairwise = {(0, 1): np.array([[5.0, 0.5, 3.0], [4.0, 4.0, 4.0]], dtype=float)}

    relinked = relink_tracks_at_geometry_issues(
        tracks,
        issues,
        pairwise,
        roi_indices_by_session=(np.array([0, 1]), np.array([5, 7, 6])),
        config={"max_edge_cost": 6.0, "min_cost_improvement": 0.25},
    )

    assert relinked[0, 1] == 7
    assert relinked[1, 1] == 6
