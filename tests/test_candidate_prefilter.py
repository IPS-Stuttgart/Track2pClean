from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.association.candidate_prefilter import (
    CentroidCandidatePrefilterConfig,
    apply_candidate_mask,
    candidate_edges_from_mask,
    centroid_candidate_mask,
)


def test_centroid_candidate_mask_supports_distance_and_row_top_k():
    reference = np.array([[0.0, 10.0], [0.0, 0.0]])
    measurement = np.array([[0.1, 9.8, 100.0], [0.0, 0.0, 0.0]])

    mask = centroid_candidate_mask(
        reference,
        measurement,
        config=CentroidCandidatePrefilterConfig(max_distance=1.0, row_top_k=1),
    )

    assert mask.tolist() == [[True, False, False], [False, True, False]]


def test_centroid_candidate_mask_can_require_column_top_k_intersection():
    reference = np.array([[0.0, 0.2, 10.0], [0.0, 0.0, 0.0]])
    measurement = np.array([[0.1, 9.9], [0.0, 0.0]])

    mask = centroid_candidate_mask(
        reference,
        measurement,
        config=CentroidCandidatePrefilterConfig(row_top_k=1, column_top_k=1),
    )

    assert mask.tolist() == [[True, False], [False, False], [False, True]]


def test_apply_candidate_mask_replaces_non_candidates_with_large_cost():
    costs = np.array([[1.0, 2.0], [3.0, 4.0]])
    mask = np.array([[True, False], [False, True]])

    gated = apply_candidate_mask(costs, mask, large_cost=99.0)

    assert gated.tolist() == [[1.0, 99.0], [99.0, 4.0]]


def test_candidate_edges_from_mask_returns_sparse_coordinates():
    mask = np.array([[True, False, True], [False, True, False]])

    assert candidate_edges_from_mask(mask) == ((0, 0), (0, 2), (1, 1))


def test_candidate_prefilter_rejects_invalid_shapes():
    with pytest.raises(ValueError, match="reference_centroids"):
        centroid_candidate_mask(np.zeros((3, 3)), np.zeros((2, 2)))

    with pytest.raises(ValueError, match="candidate_mask shape"):
        apply_candidate_mask(np.zeros((2, 2)), np.zeros((2, 3), dtype=bool))
