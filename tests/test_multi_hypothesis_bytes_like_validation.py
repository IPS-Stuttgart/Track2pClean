from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.association.multi_hypothesis import (
    edge_union_costs,
    enumerate_track_hypotheses,
    top_k_edge_candidates,
)

_TWO_ITEM_EDGE_ERROR = "edge must be a two-item session edge"
_EDGE_CANDIDATE_ERROR = "must be a three-item edge candidate"
_CONSENSUS_EDGE_ERROR = "edge must be a four-item consensus edge"


@pytest.mark.parametrize("edge", [bytearray(b"ab"), memoryview(b"ab")])
def test_top_k_edge_candidates_rejects_bytes_like_session_edges(edge: object) -> None:
    with pytest.raises(ValueError, match=_TWO_ITEM_EDGE_ERROR):
        top_k_edge_candidates(np.zeros((1, 1), dtype=float), edge=edge)  # type: ignore[arg-type]


def test_enumerate_track_hypotheses_rejects_bytes_like_edge_keys() -> None:
    with pytest.raises(ValueError, match=_TWO_ITEM_EDGE_ERROR):
        enumerate_track_hypotheses(
            ("s0", "s1"),
            {memoryview(b"ab"): [(0, 0, 0.0)]},  # type: ignore[dict-item]
            start_roi_indices=[0],
        )


@pytest.mark.parametrize(
    "candidate",
    [bytearray(b"abc"), memoryview(b"abc")],
)
def test_enumerate_track_hypotheses_rejects_bytes_like_edge_candidates(candidate: object) -> None:
    with pytest.raises(ValueError, match=_EDGE_CANDIDATE_ERROR):
        enumerate_track_hypotheses(
            ("s0", "s1"),
            {(0, 1): [candidate]},  # type: ignore[list-item]
            start_roi_indices=[0],
        )


def test_edge_union_costs_rejects_bytes_like_consensus_edges() -> None:
    with pytest.raises(ValueError, match=_CONSENSUS_EDGE_ERROR):
        edge_union_costs([{memoryview(b"abcd"): 1}])  # type: ignore[dict-item]
