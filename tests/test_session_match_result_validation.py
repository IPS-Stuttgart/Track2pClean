from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.matching import SessionMatchResult


class _BadIndex:
    def __index__(self) -> int:
        raise OverflowError("index conversion failed")


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


def test_session_match_result_rejects_duplicate_reference_roi_indices():
    with pytest.raises(ValueError, match="reference_roi_indices.*unique ROI indices"):
        _valid_match_result(
            reference_positions=np.asarray([0, 1], dtype=int),
            measurement_positions=np.asarray([0, 1], dtype=int),
            reference_roi_indices=np.asarray([10, 10], dtype=int),
            measurement_roi_indices=np.asarray([20, 21], dtype=int),
            costs=np.asarray([1.0, 1.25], dtype=float),
        )


def test_session_match_result_rejects_duplicate_measurement_roi_indices():
    with pytest.raises(ValueError, match="measurement_roi_indices.*unique ROI indices"):
        _valid_match_result(
            reference_positions=np.asarray([0, 1], dtype=int),
            measurement_positions=np.asarray([0, 1], dtype=int),
            reference_roi_indices=np.asarray([10, 11], dtype=int),
            measurement_roi_indices=np.asarray([20, 20], dtype=int),
            costs=np.asarray([1.0, 1.25], dtype=float),
        )


def test_session_match_result_rejects_boolean_positions():
    with pytest.raises(ValueError, match="reference_positions"):
        _valid_match_result(reference_positions=np.asarray([True], dtype=object))


def test_session_match_result_rejects_nonfinite_costs():
    with pytest.raises(ValueError, match="costs"):
        _valid_match_result(costs=np.asarray([np.nan], dtype=float))


@pytest.mark.parametrize(
    "field_name",
    [
        "reference_positions",
        "measurement_positions",
        "reference_roi_indices",
        "measurement_roi_indices",
    ],
)
def test_session_match_result_rejects_bad_index_values(field_name):
    with pytest.raises(ValueError, match=field_name):
        _valid_match_result(**{field_name: np.asarray([_BadIndex()], dtype=object)})


@pytest.mark.parametrize(
    "field_name",
    [
        "reference_positions",
        "measurement_positions",
        "reference_roi_indices",
        "measurement_roi_indices",
    ],
)
def test_session_match_result_rejects_oversized_integer_values(field_name):
    with pytest.raises(ValueError, match=field_name):
        _valid_match_result(**{field_name: np.asarray([10**100], dtype=object)})


@pytest.mark.parametrize(
    "costs",
    [
        [True],
        [np.bool_(False)],
        np.asarray([True], dtype=bool),
        np.asarray([np.bool_(False)], dtype=object),
    ],
)
def test_session_match_result_rejects_boolean_costs(costs):
    with pytest.raises(ValueError, match="finite numeric assignment costs"):
        _valid_match_result(costs=costs)


def test_session_match_result_accepts_numeric_cost_strings():
    result = _valid_match_result(costs=["1.25"])
    np.testing.assert_allclose(result.costs, np.asarray([1.25]))


def test_session_match_result_normalizes_integer_like_arrays():
    result = _valid_match_result(
        reference_roi_indices=np.asarray([10.0], dtype=float),
        measurement_roi_indices=np.asarray([20], dtype=np.int64),
        costs=np.asarray([2], dtype=int),
    )

    np.testing.assert_array_equal(result.as_pair_array(), np.asarray([[10, 20]]))
    assert result.costs.dtype.kind == "f"
