from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association.absence_model import apply_absence_adjustment
from bayescatrack.association.consensus_priors import (
    ConsensusPriorConfig,
    apply_consensus_edge_priors,
    edge_votes_from_tracks,
)
from bayescatrack.association.joint_registration_assignment import (
    JointRefinementConfig,
    JointRegistrationAssignmentConfig,
    anchor_relief_cost_matrix,
    apply_joint_anchor_relief_to_pairwise_costs,
    high_confidence_anchor_edges,
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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_iterations", True),
        ("max_iterations", 1.5),
        ("max_iterations", 0),
        ("high_confidence_quantile", True),
        ("high_confidence_quantile", float("nan")),
        ("high_confidence_quantile", float("inf")),
        ("high_confidence_quantile", -0.1),
        ("high_confidence_quantile", 1.1),
        ("min_anchor_edges", False),
        ("min_anchor_edges", 1.5),
        ("min_anchor_edges", 0),
        ("cost_relief", True),
        ("cost_relief", float("nan")),
        ("cost_relief", float("inf")),
        ("cost_relief", -0.1),
        ("convergence_tolerance", False),
        ("convergence_tolerance", float("nan")),
        ("convergence_tolerance", float("inf")),
        ("convergence_tolerance", -0.1),
    ],
)
def test_joint_refinement_config_rejects_invalid_controls(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError, match=field):
        JointRefinementConfig(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_anchor_probability", True),
        ("min_anchor_probability", float("nan")),
        ("min_anchor_probability", float("inf")),
        ("min_anchor_probability", -0.1),
        ("min_anchor_probability", 1.1),
        ("min_anchor_margin", False),
        ("min_anchor_margin", float("nan")),
        ("min_anchor_margin", float("inf")),
        ("min_anchor_margin", -0.1),
        ("min_anchors", True),
        ("min_anchors", 1.5),
        ("min_anchors", 0),
    ],
)
def test_joint_registration_assignment_config_rejects_invalid_controls(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError, match=field):
        JointRegistrationAssignmentConfig(**{field: value})


@pytest.mark.parametrize("quantile", [True, float("nan"), float("inf"), -0.1, 1.1])
def test_high_confidence_anchor_edges_rejects_invalid_quantile(
    quantile: object,
) -> None:
    with pytest.raises(ValueError, match="quantile"):
        high_confidence_anchor_edges([[1.0]], quantile=quantile)


@pytest.mark.parametrize("min_anchor_edges", [True, 1.5, 0])
def test_high_confidence_anchor_edges_rejects_invalid_anchor_budget(
    min_anchor_edges: object,
) -> None:
    with pytest.raises(ValueError, match="min_anchor_edges"):
        high_confidence_anchor_edges([[1.0]], min_anchor_edges=min_anchor_edges)


@pytest.mark.parametrize("relief", [True, float("nan"), float("inf"), -0.1])
def test_anchor_relief_cost_matrix_rejects_invalid_relief(relief: object) -> None:
    with pytest.raises(ValueError, match="relief"):
        anchor_relief_cost_matrix([[1.0]], ((0, 0),), relief=relief)


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


def test_consensus_prior_direct_string_variant_cost_is_not_split_into_characters():
    config = ConsensusPriorConfig(variant_costs="registered-iou, roi-aware-shifted")

    assert config.variant_costs == ("registered-iou", "roi-aware-shifted")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("variant_costs", ()),
        ("variant_costs", ("registered-iou", "")),
        ("variant_costs", ("registered-iou", True)),
        ("min_votes", True),
        ("min_votes", 1.5),
        ("min_votes", 0),
        ("relief", True),
        ("relief", float("nan")),
        ("relief", float("inf")),
        ("relief", -0.1),
        ("max_relief", False),
        ("max_relief", float("nan")),
        ("max_relief", float("inf")),
        ("max_relief", -0.1),
        ("large_cost", True),
        ("large_cost", float("nan")),
        ("large_cost", float("inf")),
        ("large_cost", 0.0),
        ("ignore_variant_failures", "true"),
    ],
)
def test_consensus_prior_config_rejects_invalid_controls(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError, match=field):
        ConsensusPriorConfig(**{field: value})


@pytest.mark.parametrize("vote_count", [True, 1.5, -1])
def test_consensus_prior_rejects_invalid_vote_counts(vote_count: object) -> None:
    with pytest.raises(ValueError, match="vote_count"):
        apply_consensus_edge_priors(
            {(0, 1): np.array([[1.0]], dtype=float)},
            {(0, 1, 0, 0): vote_count},
        )


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
