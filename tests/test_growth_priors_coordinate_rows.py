from __future__ import annotations

import numpy as np
from bayescatrack.association.growth_priors import (
    affine_growth_residuals,
    estimate_affine_growth_field,
    growth_penalty_matrix,
)


def test_growth_priors_accept_coordinate_row_centroid_matrices() -> None:
    source = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    target = source + np.asarray([2.0, 3.0])
    source_centroids = source.T
    target_centroids = target.T

    affine = estimate_affine_growth_field(source_centroids, target_centroids)
    residuals = affine_growth_residuals(
        source_centroids, target_centroids, affine=affine
    )
    penalties = growth_penalty_matrix(source_centroids, target_centroids, affine=affine)

    np.testing.assert_allclose(residuals, np.zeros(3), atol=1.0e-10)
    assert penalties.shape == (3, 3)
    assert np.argmin(penalties, axis=1).tolist() == [0, 1, 2]


def test_growth_priors_default_ambiguous_two_roi_centroids_to_coordinate_rows() -> None:
    source_centroids = np.asarray([[0.0, 1.0], [10.0, 10.0]])
    target_centroids = np.asarray([[5.0, 6.0], [8.0, 8.0]])
    affine = np.asarray([[1.0, 0.0, 5.0], [0.0, 1.0, -2.0]])

    penalties = growth_penalty_matrix(source_centroids, target_centroids, affine=affine)

    assert penalties.shape == (2, 2)
    np.testing.assert_allclose(np.diag(penalties), np.zeros(2), atol=1.0e-12)
    assert np.argmin(penalties, axis=1).tolist() == [0, 1]
