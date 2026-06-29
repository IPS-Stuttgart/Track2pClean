from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.association.advanced_uncertainty import (
    posterior_probability_matrix,
    uncertainty_aware_cost_matrix,
)


def test_posterior_probabilities_accept_empty_candidate_columns() -> None:
    probabilities = posterior_probability_matrix(np.zeros((3, 0), dtype=float))

    assert probabilities.shape == (3, 0)
    assert np.issubdtype(probabilities.dtype, np.floating)


def test_posterior_probabilities_validate_empty_candidate_reliability_shape() -> None:
    with pytest.raises(ValueError, match="reliability_matrix must match cost_matrix shape"):
        posterior_probability_matrix(
            np.zeros((3, 0), dtype=float),
            reliability_matrix=np.zeros((3, 1), dtype=float),
        )


def test_uncertainty_aware_cost_matrix_accepts_empty_candidate_columns() -> None:
    result = uncertainty_aware_cost_matrix(np.zeros((3, 0), dtype=float))

    assert result.adjusted_cost_matrix.shape == (3, 0)
    assert result.posterior_probability_matrix.shape == (3, 0)
    assert result.reliability_matrix.shape == (3, 0)
