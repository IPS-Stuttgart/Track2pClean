"""Regression tests for multi-hypothesis ROI candidate validation."""

from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association.multi_hypothesis import (
    candidate_edge_map,
    enumerate_track_hypotheses,
)


@pytest.mark.parametrize(
    "roi_value", [True, np.bool_(True), np.array(True), 1.5, np.nan, -1]
)
def test_candidate_edge_map_rejects_invalid_roi_indices(roi_value) -> None:
    with pytest.raises(ValueError, match="roi_indices_by_session"):
        candidate_edge_map({(0, 1): [[0.1]]}, [[roi_value], [20]])


@pytest.mark.parametrize(
    "candidate",
    [(True, 20, 0.1), (10, np.array(True), 0.1), (10, 20, np.nan)],
)
def test_enumerate_track_hypotheses_rejects_invalid_candidates(candidate) -> None:
    with pytest.raises(ValueError, match="edge_candidates"):
        enumerate_track_hypotheses(
            ("s0", "s1"),
            {(0, 1): [candidate]},
            start_roi_indices=[10],
        )


def test_enumerate_track_hypotheses_rejects_invalid_start_roi() -> None:
    with pytest.raises(ValueError, match="start_roi_indices"):
        enumerate_track_hypotheses(
            ("s0", "s1"),
            {(0, 1): [(10, 20, 0.1)]},
            start_roi_indices=[np.array(True)],
        )
