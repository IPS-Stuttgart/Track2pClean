from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from bayescatrack import _pyrecest_pairwise_features as pairwise_features


def _means() -> np.ndarray:
    return np.zeros((2, 1), dtype=float)


def _covariances() -> np.ndarray:
    return np.eye(2, dtype=float)[:, :, None]


@pytest.mark.parametrize(
    "regularization",
    [True, np.nan, np.inf, -1.0, "invalid", np.array([0.0])],
)
def test_fallback_mahalanobis_rejects_invalid_regularization(
    monkeypatch: pytest.MonkeyPatch,
    regularization: Any,
) -> None:
    monkeypatch.setattr(
        pairwise_features, "_pyrecest_pairwise_mahalanobis_distances", None
    )

    with pytest.raises(
        ValueError,
        match="regularization must be a finite non-negative scalar",
    ):
        pairwise_features.pairwise_mahalanobis_distances(
            _means(),
            _covariances(),
            _means(),
            _covariances(),
            regularization=regularization,
        )


def test_fallback_mahalanobis_accepts_valid_regularization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pairwise_features, "_pyrecest_pairwise_mahalanobis_distances", None
    )

    distances = pairwise_features.pairwise_mahalanobis_distances(
        _means(),
        _covariances(),
        _means(),
        _covariances(),
        regularization=0.0,
    )

    np.testing.assert_allclose(distances, np.zeros((1, 1)))


@pytest.mark.parametrize(
    "epsilon",
    [True, np.nan, np.inf, 0.0, -1.0, "invalid", np.array([1.0e-6])],
)
def test_fallback_covariance_shape_components_reject_invalid_epsilon(
    monkeypatch: pytest.MonkeyPatch,
    epsilon: Any,
) -> None:
    monkeypatch.setattr(
        pairwise_features,
        "_pyrecest_pairwise_covariance_shape_components",
        None,
    )

    with pytest.raises(
        ValueError,
        match="epsilon must be a finite positive scalar",
    ):
        pairwise_features.pairwise_covariance_shape_components(
            _covariances(),
            _covariances(),
            epsilon=epsilon,
        )


def test_fallback_covariance_shape_components_accept_valid_epsilon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pairwise_features,
        "_pyrecest_pairwise_covariance_shape_components",
        None,
    )

    shape_cost, logdet_cost, shape_similarity = (
        pairwise_features.pairwise_covariance_shape_components(
            _covariances(),
            _covariances(),
            epsilon=1.0e-6,
        )
    )

    np.testing.assert_allclose(shape_cost, np.zeros((1, 1)))
    np.testing.assert_allclose(logdet_cost, np.zeros((1, 1)))
    np.testing.assert_allclose(shape_similarity, np.ones((1, 1)))
