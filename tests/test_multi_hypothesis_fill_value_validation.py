"""Regression tests for multi-hypothesis missing-value sentinels."""

from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association.multi_hypothesis import (
    HypothesisConfig,
    TrackHypothesis,
    candidate_edge_map,
    consensus_edges,
    edge_union_costs,
    enumerate_track_hypotheses,
)


@pytest.mark.parametrize("bad_fill_value", [0, 1])
def test_hypothesis_config_rejects_non_negative_fill_value(bad_fill_value) -> None:
    with pytest.raises(ValueError, match="fill_value"):
        HypothesisConfig(fill_value=bad_fill_value)


@pytest.mark.parametrize("bad_fill_value", [0, 1])
def test_consensus_edges_rejects_non_negative_fill_value(bad_fill_value) -> None:
    with pytest.raises(ValueError, match="fill_value"):
        consensus_edges([[[0, 2]]], min_votes=1, fill_value=bad_fill_value)


def test_enumerate_track_hypotheses_preserves_roi_zero() -> None:
    hypotheses = enumerate_track_hypotheses(
        ("s0", "s1"),
        {(0, 1): [(0, 2, 0.25)]},
        start_roi_indices=[0],
        config=HypothesisConfig(fill_value=-1),
    )

    assert hypotheses == [TrackHypothesis(row=(0, 2), cost=0.25)]


def test_consensus_edges_preserves_roi_zero_with_negative_fill_value() -> None:
    assert consensus_edges([[[0, 2]]], min_votes=1, fill_value=-1) == {(0, 1, 0, 2): 1}


def test_candidate_edge_map_preserves_large_exact_roi_indices() -> None:
    source_roi = 2**53 + 1
    target_roi = 2**53 + 3

    edge_map = candidate_edge_map(
        {(0, 1): np.array([[0.25]], dtype=float)},
        ([source_roi], [target_roi]),
        config=HypothesisConfig(edge_top_k=1),
    )

    assert edge_map == {(0, 1): [(source_roi, target_roi, 0.25)]}


def test_edge_union_costs_preserves_large_exact_roi_indices() -> None:
    source_roi = 2**53 + 1
    target_roi = 2**53 + 3

    costs = edge_union_costs([{(0, 1, source_roi, target_roi): 2}])

    assert costs == {(0, 1, source_roi, target_roi): 0.5}
