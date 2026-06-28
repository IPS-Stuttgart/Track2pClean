import numpy as np
import pytest
from bayescatrack.matching import (
    build_track_rows_from_bundles,
    build_track_rows_from_matches,
    solve_bundle_linear_assignment,
)


class _Bundle:
    def __init__(self, costs):
        self.pairwise_cost_matrix = np.asarray(costs, dtype=float)
        self.reference_session_name = "s1"
        self.measurement_session_name = "s2"
        self.reference_roi_indices = np.array([10, 20], dtype=int)
        self.measurement_roi_indices = np.array([100, 200], dtype=int)


def test_solve_bundle_linear_assignment_rejects_boolean_max_cost():
    with pytest.raises(ValueError, match="max_cost"):
        solve_bundle_linear_assignment(
            _Bundle([[0.0, 1.0], [1.0, 0.0]]),
            max_cost=True,
        )


def test_build_track_rows_from_matches_rejects_boolean_start_session_index():
    with pytest.raises(ValueError, match="start_session_index"):
        build_track_rows_from_matches(
            ("s1", "s2"),
            [np.array([[0, 1]], dtype=int)],
            start_roi_indices=np.array([0], dtype=int),
            start_session_index=True,
        )


def test_build_track_rows_from_bundles_rejects_fractional_start_session_index():
    with pytest.raises(ValueError, match="start_session_index"):
        build_track_rows_from_bundles(
            [_Bundle([[0.0, 1.0], [1.0, 0.0]])],
            start_session_index=0.5,
        )


def test_build_track_rows_from_matches_rejects_boolean_fill_value():
    with pytest.raises(ValueError, match="fill_value"):
        build_track_rows_from_matches(
            ("s1", "s2"),
            [np.array([[0, 1]], dtype=int)],
            start_roi_indices=np.array([0], dtype=int),
            fill_value=False,
        )


def test_build_track_rows_from_matches_rejects_bytearray_session_names():
    with pytest.raises(ValueError, match="session_names"):
        build_track_rows_from_matches(
            bytearray(b"ab"),
            [np.empty((0, 2), dtype=int)],
            start_roi_indices=np.array([0], dtype=int),
        )
