"""Regression tests for multi-hypothesis session-edge validation."""

from __future__ import annotations

import pytest
from bayescatrack.association.multi_hypothesis import (
    candidate_edge_map,
    enumerate_track_hypotheses,
    top_k_edge_candidates,
)


def test_top_k_edge_candidates_rejects_negative_session_edge() -> None:
    with pytest.raises(ValueError, match="edge session indices must be non-negative"):
        top_k_edge_candidates([[0.1]], edge=(-1, 0))


def test_candidate_edge_map_rejects_negative_session_edge() -> None:
    with pytest.raises(ValueError, match="edge session indices must be non-negative"):
        candidate_edge_map({(-1, 0): [[0.1]]}, [[10], [20]])


def test_candidate_edge_map_rejects_out_of_range_session_edge() -> None:
    with pytest.raises(
        ValueError,
        match="edge session indices must refer to existing sessions",
    ):
        candidate_edge_map({(0, 2): [[0.1]]}, [[10], [20]])


def test_enumerate_track_hypotheses_rejects_invalid_candidate_edge() -> None:
    with pytest.raises(ValueError, match="edge session indices must be non-negative"):
        enumerate_track_hypotheses(
            ("s0", "s1"),
            {(-1, 0): [(10, 20, 0.1)]},
            start_roi_indices=[10],
        )
