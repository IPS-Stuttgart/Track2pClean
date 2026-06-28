from __future__ import annotations

import numpy as np
import pytest
from bayescatrack import CalciumPlaneData


def _single_roi_plane() -> CalciumPlaneData:
    masks = np.zeros((1, 4, 4), dtype=bool)
    masks[0, 1:3, 1:3] = True
    return CalciumPlaneData(masks)


def test_position_covariances_rejects_nan_regularization() -> None:
    with pytest.raises(
        ValueError, match="regularization must be a finite non-negative value"
    ):
        _single_roi_plane().position_covariances(regularization=np.nan)


def test_position_covariances_rejects_vector_regularization() -> None:
    with pytest.raises(
        ValueError, match="regularization must be a finite non-negative value"
    ):
        _single_roi_plane().position_covariances(regularization=np.array([1.0]))


def test_position_covariances_accepts_zero_dimensional_regularization_array() -> None:
    covariances = _single_roi_plane().position_covariances(
        regularization=np.array(1.0e-6)
    )

    assert covariances.shape == (2, 2, 1)


def test_constant_velocity_state_moments_rejects_vector_velocity_variance() -> None:
    with pytest.raises(
        ValueError, match="velocity_variance must be a finite non-negative value"
    ):
        _single_roi_plane().to_constant_velocity_state_moments(
            velocity_variance=np.array([25.0])
        )


def test_pairwise_cost_matrix_rejects_vector_scalar_weight() -> None:
    plane = _single_roi_plane()

    with pytest.raises(
        ValueError, match="centroid_weight must be a finite non-negative value"
    ):
        plane.build_pairwise_cost_matrix(plane, centroid_weight=np.array([1.0]))
