from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.matching import SessionMatchResult


def _valid_match_result(**overrides):
    kwargs = {
        "reference_session_name": "day0",
        "measurement_session_name": "day1",
        "reference_positions": np.asarray([0], dtype=int),
        "measurement_positions": np.asarray([0], dtype=int),
        "reference_roi_indices": np.asarray([10], dtype=int),
        "measurement_roi_indices": np.asarray([20], dtype=int),
        "costs": np.asarray([1.0], dtype=float),
    }
    kwargs.update(overrides)
    return SessionMatchResult(**kwargs)


def test_session_match_result_rejects_fractional_roi_indices():
    with pytest.raises(ValueError, match="reference_roi_indices"):
        _valid_match_result(reference_roi_indices=np.asarray([10.5], dtype=float))


def test_session_match_result_rejects_boolean_positions():
    with pytest.raises(ValueError, match="reference_positions"):
        _valid_match_result(reference_positions=np.asarray([True], dtype=object))


def test_session_match_result_rejects_nonfinite_costs():
    with pytest.raises(ValueError, match="costs"):
        _valid_match_result(costs=np.asarray([np.nan], dtype=float))


def test_session_match_result_normalizes_integer_like_arrays():
    result = _valid_match_result(
        reference_roi_indices=np.asarray([10.0], dtype=float),
        measurement_roi_indices=np.asarray([20], dtype=np.int64),
        costs=np.asarray([2], dtype=int),
    )

    np.testing.assert_array_equal(result.as_pair_array(), np.asarray([[10, 20]]))
    assert result.costs.dtype.kind == "f"
