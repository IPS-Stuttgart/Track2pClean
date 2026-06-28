"""Regression tests for candidate prefilter scalar controls."""

from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association.candidate_prefilter import (
    CentroidCandidatePrefilterConfig,
    apply_candidate_mask,
)


@pytest.mark.parametrize("bad_value", [np.asarray(1), np.asarray([1])])
def test_centroid_candidate_top_k_rejects_array_values(bad_value):
    with pytest.raises(ValueError, match="row_top_k must be an integer"):
        CentroidCandidatePrefilterConfig(row_top_k=bad_value)

    with pytest.raises(ValueError, match="column_top_k must be an integer"):
        CentroidCandidatePrefilterConfig(column_top_k=bad_value)


@pytest.mark.parametrize(
    "bad_value",
    [np.asarray(1.0), np.asarray([1.0]), "1.0", b"1.0"],
)
def test_centroid_candidate_float_controls_reject_array_and_text_values(bad_value):
    with pytest.raises(
        ValueError,
        match="max_distance must be a finite non-negative value",
    ):
        CentroidCandidatePrefilterConfig(max_distance=bad_value)

    with pytest.raises(ValueError, match="large_cost must be a finite positive value"):
        CentroidCandidatePrefilterConfig(large_cost=bad_value)

    with pytest.raises(ValueError, match="large_cost must be a finite positive value"):
        apply_candidate_mask(
            np.zeros((1, 1), dtype=float),
            np.ones((1, 1), dtype=bool),
            large_cost=bad_value,
        )


@pytest.mark.parametrize("value", [1, 1.0, np.float64(1.0)])
def test_centroid_candidate_float_controls_accept_numeric_scalars(value):
    config = CentroidCandidatePrefilterConfig(max_distance=value, large_cost=value)
    masked = apply_candidate_mask(
        np.zeros((1, 1), dtype=float),
        np.zeros((1, 1), dtype=bool),
        large_cost=value,
    )

    assert config.max_distance == 1.0
    assert config.large_cost == 1.0
    assert masked.tolist() == [[1.0]]
