import numpy as np
import pytest

from bayescatrack.matching import (
    build_track_rows_from_bundles,
    build_track_rows_from_matches,
    solve_bundle_linear_assignment,
)


class _Bundle:
    def __init__(
        self,
        costs,
        *,
        reference_roi_indices=None,
        measurement_roi_indices=None,
    ):
        self.pairwise_cost_matrix = np.asarray(costs, dtype=float)
        self.reference_session_name = "s1"
        self.measurement_session_name = "s2"
        self.reference_roi_indices = (
            [10] if reference_roi_indices is None else reference_roi_indices
        )
        self.measurement_roi_indices = (
            [100] if measurement_roi_indices is None else measurement_roi_indices
        )


@pytest.mark.parametrize(
    "bad_start_session_index",
    [np.array(0), np.array(0, dtype=object)],
)
def test_build_track_rows_from_matches_rejects_array_start_session_index(
    bad_start_session_index,
):
    with pytest.raises(ValueError, match="start_session_index must be an integer"):
        build_track_rows_from_matches(
            ("s1", "s2"),
            [np.array([[0, 1]], dtype=int)],
            start_roi_indices=[0],
            start_session_index=bad_start_session_index,
        )


@pytest.mark.parametrize(
    "bad_fill_value",
    [np.array(-1), np.array(-1, dtype=object)],
)
def test_build_track_rows_from_matches_rejects_array_fill_value(bad_fill_value):
    with pytest.raises(ValueError, match="fill_value must be an integer"):
        build_track_rows_from_matches(
            ("s1", "s2"),
            [np.array([[0, 1]], dtype=int)],
            start_roi_indices=[0],
            fill_value=bad_fill_value,
        )


def test_build_track_rows_from_bundles_rejects_array_start_session_index():
    with pytest.raises(
        ValueError,
        match="start_session_index must be an integer session index",
    ):
        build_track_rows_from_bundles(
            [_Bundle([[0.0]])],
            start_session_index=np.array(0),
        )


def test_solve_bundle_linear_assignment_rejects_array_reference_roi_index():
    with pytest.raises(
        ValueError,
        match="reference_roi_indices must contain integer ROI indices",
    ):
        solve_bundle_linear_assignment(
            _Bundle(
                [[0.0, 1.0], [1.0, 0.0]],
                reference_roi_indices=[np.array(10), 20],
                measurement_roi_indices=[100, 200],
            )
        )


def test_build_track_rows_from_matches_rejects_array_match_roi_index():
    with pytest.raises(
        ValueError,
        match="reference_roi_indices must contain integer ROI indices",
    ):
        build_track_rows_from_matches(
            ("s1", "s2"),
            [([np.array(0)], [1])],
            start_roi_indices=[0],
        )
