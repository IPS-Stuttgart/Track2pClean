from __future__ import annotations

import pytest
from bayescatrack.matching import SessionMatchResult


def _valid_match_result(**overrides):
    kwargs = {
        "reference_session_name": "day0",
        "measurement_session_name": "day1",
        "reference_positions": [0, 1],
        "measurement_positions": [0, 1],
        "reference_roi_indices": [10, 11],
        "measurement_roi_indices": [20, 21],
        "costs": [1.0, 1.25],
    }
    kwargs.update(overrides)
    return SessionMatchResult(**kwargs)


def test_session_match_result_rejects_duplicate_reference_positions() -> None:
    with pytest.raises(
        ValueError, match="reference_positions.*unique assignment positions"
    ):
        _valid_match_result(reference_positions=[0, 0])


def test_session_match_result_rejects_duplicate_measurement_positions() -> None:
    with pytest.raises(
        ValueError,
        match="measurement_positions.*unique assignment positions",
    ):
        _valid_match_result(measurement_positions=[0, 0])
