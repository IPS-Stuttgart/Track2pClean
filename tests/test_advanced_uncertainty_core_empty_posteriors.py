from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association import advanced_uncertainty

_ORIGINAL_ATTRS = (
    "_bayescatrack_advanced_uncertainty_empty_probability_original",
    "_bayescatrack_original",
)


def _unwrapped_posterior_probability_matrix():
    function = advanced_uncertainty.posterior_probability_matrix
    seen = set()
    while True:
        function_id = id(function)
        if function_id in seen:
            raise RuntimeError("wrapper chain cycle")
        seen.add(function_id)
        for attr in _ORIGINAL_ATTRS:
            wrapped = getattr(function, attr, None)
            if wrapped is not None:
                function = wrapped
                break
        else:
            return function


def test_core_posterior_probabilities_accept_empty_candidate_rows() -> None:
    posterior_probability_matrix = _unwrapped_posterior_probability_matrix()

    probabilities = posterior_probability_matrix(np.zeros((0, 3), dtype=float))

    assert probabilities.shape == (0, 3)
    assert np.issubdtype(probabilities.dtype, np.floating)


def test_core_posterior_probabilities_accept_empty_candidate_columns() -> None:
    posterior_probability_matrix = _unwrapped_posterior_probability_matrix()

    probabilities = posterior_probability_matrix(np.zeros((3, 0), dtype=float))

    assert probabilities.shape == (3, 0)
    assert np.issubdtype(probabilities.dtype, np.floating)


def test_core_posterior_probabilities_validate_empty_reliability_shape() -> None:
    posterior_probability_matrix = _unwrapped_posterior_probability_matrix()

    with pytest.raises(
        ValueError, match="reliability_matrix must match cost_matrix shape"
    ):
        posterior_probability_matrix(
            np.zeros((3, 0), dtype=float),
            reliability_matrix=np.zeros((3, 1), dtype=float),
        )
