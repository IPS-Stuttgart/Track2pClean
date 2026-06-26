from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from bayescatrack.association.calibrated_costs import (
    pairwise_components_from_bundle,
    supervised_pairwise_mask_from_reference,
)
from bayescatrack.reference import Track2pReference


def test_supervised_pairwise_mask_from_reference_uses_only_annotated_endpoints():
    reference = Track2pReference(
        session_names=("s0", "s1"),
        suite2p_indices=np.array(
            [
                [0, 10],
                [2, None],
                [None, 12],
            ],
            dtype=object,
        ),
    )

    supervised = supervised_pairwise_mask_from_reference(
        reference,
        0,
        1,
        reference_roi_indices=np.array([0, 1, 2], dtype=int),
        measurement_roi_indices=np.array([10, 11, 12], dtype=int),
    )

    expected = np.array(
        [
            [True, False, True],
            [False, False, False],
            [True, False, True],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(supervised, expected)


def _minimal_covariance_bundle() -> SimpleNamespace:
    return SimpleNamespace(
        pairwise_components={"pairwise_cost_matrix": np.zeros((1, 1), dtype=float)},
        reference_state_means=np.zeros((4, 1), dtype=float),
        reference_state_covariances=np.eye(4, dtype=float)[:, :, None],
        measurements=np.zeros((2, 1), dtype=float),
        measurement_covariances=np.eye(2, dtype=float)[:, :, None],
    )


@pytest.mark.parametrize("covariance_epsilon", [True, 0.0, -1.0, np.nan, np.inf])
def test_pairwise_components_from_bundle_rejects_invalid_covariance_epsilon(
    covariance_epsilon,
):
    with pytest.raises(ValueError, match="covariance_epsilon"):
        pairwise_components_from_bundle(
            _minimal_covariance_bundle(),
            covariance_epsilon=covariance_epsilon,
        )


def test_pairwise_components_from_bundle_accepts_numpy_scalar_covariance_epsilon():
    components = pairwise_components_from_bundle(
        _minimal_covariance_bundle(),
        covariance_epsilon=np.float64(1.0e-6),
    )

    assert components["covariance_shape_cost"].shape == (1, 1)
    assert components["mahalanobis_centroid_distance"].shape == (1, 1)
