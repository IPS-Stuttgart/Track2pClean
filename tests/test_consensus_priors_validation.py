from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association.consensus_priors import (
    apply_consensus_edge_priors,
    edge_votes_from_tracks,
)


@pytest.mark.parametrize(
    "session_edges",
    [
        [(False, 1)],
        [(0, True)],
        [(0.25, 1)],
        [(0, 1.5)],
        [(0, 0)],
        "01",
    ],
)
def test_edge_votes_from_tracks_rejects_malformed_session_edges(session_edges: object) -> None:
    with pytest.raises(ValueError, match="session_edges"):
        edge_votes_from_tracks([[{0: 10, 1: 20}]], session_edges=session_edges)  # type: ignore[arg-type]


def test_edge_votes_from_tracks_rejects_malformed_track_indices() -> None:
    with pytest.raises(ValueError, match="track ROI index"):
        edge_votes_from_tracks([[{0: "10", 1: 20}]], session_edges=[(0, 1)])


def test_edge_votes_from_tracks_still_ignores_negative_missing_rois() -> None:
    votes = edge_votes_from_tracks([[{0: 10, 1: -1}]], session_edges=[(0, 1)])

    assert votes == {}


@pytest.mark.parametrize(
    "votes",
    [
        {(0, 1, True, 2): 2},
        {(0, 1, 1.5, 2): 2},
        {(0, 0, 1, 2): 2},
        {(0, 1, 1, 2): True},
        {(0, 1, 1, 2): "2"},
    ],
)
def test_apply_consensus_edge_priors_rejects_malformed_votes(votes: object) -> None:
    with pytest.raises(ValueError):
        apply_consensus_edge_priors({(0, 1): np.ones((3, 3))}, votes)  # type: ignore[arg-type]


def test_apply_consensus_edge_priors_rejects_malformed_cost_edges() -> None:
    with pytest.raises(ValueError, match="pairwise_costs"):
        apply_consensus_edge_priors({(False, 1): np.ones((3, 3))}, {})  # type: ignore[dict-item]
