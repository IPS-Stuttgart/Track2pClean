"""Regression tests for strict numeric configuration validation."""

import numpy as np
import pytest
from bayescatrack.advanced_roi_components import (
    CandidatePruningConfig,
    candidate_mask_from_cost_matrix,
    mask_shape_descriptors,
)
from bayescatrack.association.candidate_prefilter import (
    CentroidCandidatePrefilterConfig,
    apply_candidate_mask,
)


@pytest.mark.parametrize("bad_value", [True, False, 1.5, "1.5", np.nan, np.inf])
def test_centroid_candidate_top_k_rejects_bad_values(bad_value):
    with pytest.raises(ValueError, match="row_top_k must be an integer"):
        CentroidCandidatePrefilterConfig(row_top_k=bad_value)

    with pytest.raises(ValueError, match="column_top_k must be an integer"):
        CentroidCandidatePrefilterConfig(column_top_k=bad_value)


@pytest.mark.parametrize("integer_like", [1, np.int64(1), 1.0, "1"])
def test_centroid_candidate_top_k_accepts_integer_like_values(integer_like):
    config = CentroidCandidatePrefilterConfig(
        row_top_k=integer_like,
        column_top_k=integer_like,
    )

    assert config.row_top_k == 1
    assert config.column_top_k == 1


@pytest.mark.parametrize("bad_value", [True, False, 1.5, "1.5", np.nan, np.inf])
def test_advanced_candidate_top_k_rejects_bad_values(bad_value):
    with pytest.raises(ValueError, match="top_k_per_roi must be an integer"):
        CandidatePruningConfig(top_k_per_roi=bad_value)

    with pytest.raises(ValueError, match="top_k must be an integer"):
        candidate_mask_from_cost_matrix(
            np.zeros((2, 2), dtype=float),
            top_k=bad_value,
        )


@pytest.mark.parametrize("integer_like", [1, np.int64(1), 1.0, "1"])
def test_advanced_candidate_top_k_accepts_integer_like_values(integer_like):
    config = CandidatePruningConfig(top_k_per_roi=integer_like)
    candidate_mask = candidate_mask_from_cost_matrix(
        np.asarray([[0.0, 10.0], [10.0, 0.0]], dtype=float),
        top_k=integer_like,
    )

    assert config.top_k_per_roi == 1
    assert candidate_mask.tolist() == [[True, False], [False, True]]


@pytest.mark.parametrize(
    ("factory", "kwargs", "message"),
    [
        (
            CentroidCandidatePrefilterConfig,
            {"max_distance": np.nan},
            "max_distance must be a finite non-negative value",
        ),
        (
            CentroidCandidatePrefilterConfig,
            {"large_cost": np.inf},
            "large_cost must be a finite positive value",
        ),
        (
            CandidatePruningConfig,
            {"gate_margin": np.nan},
            "gate_margin must be a finite non-negative value",
        ),
        (
            CandidatePruningConfig,
            {"large_cost": np.inf},
            "large_cost must be a finite positive value",
        ),
    ],
)
def test_candidate_configs_reject_nonfinite_numeric_values(factory, kwargs, message):
    with pytest.raises(ValueError, match=message):
        factory(**kwargs)


@pytest.mark.parametrize("bad_large_cost", [np.nan, np.inf, 0.0, -1.0])
def test_apply_candidate_mask_rejects_invalid_large_cost(bad_large_cost):
    with pytest.raises(ValueError, match="large_cost must be a finite positive value"):
        apply_candidate_mask(
            np.zeros((1, 1), dtype=float),
            np.ones((1, 1), dtype=bool),
            large_cost=bad_large_cost,
        )


@pytest.mark.parametrize("bad_radial_bins", [True, False, 1.5, "1.5", np.nan, np.inf])
def test_shape_descriptors_reject_invalid_radial_bins(bad_radial_bins):
    masks = np.zeros((1, 4, 4), dtype=bool)
    masks[0, 1:3, 1:3] = True

    with pytest.raises(ValueError, match="radial_bins must be an integer"):
        mask_shape_descriptors(masks, radial_bins=bad_radial_bins)
