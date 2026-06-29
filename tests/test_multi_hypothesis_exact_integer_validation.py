"""Regression tests for exact ROI-index preservation in multi-hypothesis helpers."""

from __future__ import annotations

import numpy as np
from bayescatrack.association.multi_hypothesis import (
    HypothesisConfig,
    candidate_edge_map,
    edge_union_costs,
)


def test_candidate_edge_map_preserves_large_exact_roi_indices() -> None:
    source_roi = 9007199254740993
    target_roi = 9007199254740995

    edge_map = candidate_edge_map(
        {(0, 1): np.array([[0.25]], dtype=float)},
        ([source_roi], [target_roi]),
        config=HypothesisConfig(edge_top_k=1),
    )

    assert edge_map == {(0, 1): [(source_roi, target_roi, 0.25)]}


def test_edge_union_costs_preserves_large_exact_roi_indices() -> None:
    source_roi = 9007199254740993
    target_roi = 9007199254740995

    costs = edge_union_costs([{(0, 1, source_roi, target_roi): 2}])

    assert costs == {(0, 1, source_roi, target_roi): 0.5}
