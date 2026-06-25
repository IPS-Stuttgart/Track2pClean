import numpy as np
import numpy.testing as npt
import pytest
from bayescatrack.matching import (
    build_track_rows_from_bundles,
    build_track_rows_from_matches,
    solve_bundle_linear_assignment,
)


class _Bundle:
    def __init__(self, costs, *, reference_roi_indices=None, measurement_roi_indices=None):
        self.pairwise_cost_matrix = np.asarray(costs, dtype=float)
        self.reference_session_name = "s1"
        self.measurement_session_name = "s2"
        self.reference_roi_indices = np.asarray(
            [10, 20] if reference_roi_indices is None else reference_roi_indices
        )
        self.measurement_roi_indices = np.asarray(
            [100, 200] if measurement_roi_indices is None else measurement_roi_indices
        )


def test_build_track_rows_from_consecutive_matches():
    rows = build_track_rows_from_matches(
        ("s1", "s2", "s3"),
        [np.array([[0, 1], [2, 3]]), np.array([[1, 5], [3, 6]])],
        start_roi_indices=np.array([0, 2]),
    )

    npt.assert_array_equal(rows, np.array([[0, 1, 5], [2, 3, 6]]))


def test_build_track_rows_from_matches_requires_explicit_start_roi_indices():
    with pytest.raises(ValueError, match="start_roi_indices must be provided"):
        build_track_rows_from_matches(
            ("s1", "s2"),
            [np.array([[0, 1]], dtype=int)],
        )


def test_build_track_rows_from_matches_rejects_duplicate_reference_rois():
    with pytest.raises(ValueError, match="duplicate reference ROI 0"):
        build_track_rows_from_matches(
            ("s1", "s2"),
            [np.array([[0, 1], [0, 2]], dtype=int)],
            start_roi_indices=np.array([0]),
        )


def test_build_track_rows_from_matches_rejects_duplicate_measurement_rois():
    with pytest.raises(ValueError, match="duplicate measurement ROI 1"):
        build_track_rows_from_matches(
            ("s1", "s2"),
            [np.array([[0, 1], [2, 1]], dtype=int)],
            start_roi_indices=np.array([0, 2]),
        )


def test_build_track_rows_from_later_seed_session_stitches_both_directions():
    rows = build_track_rows_from_matches(
        ("s1", "s2", "s3"),
        [np.array([[0, 10], [2, 20]]), np.array([[10, 5], [20, 6]])],
        start_roi_indices=np.array([10, 20, 30]),
        start_session_index=1,
    )

    npt.assert_array_equal(
        rows,
        np.array([[0, 10, 5], [2, 20, 6], [-1, 30, -1]]),
    )


@pytest.mark.parametrize(
    "bad_start_session_index",
    [True, np.bool_(False), 0.5, np.nan, "1"],
)
def test_build_track_rows_from_matches_rejects_malformed_start_session_index(
    bad_start_session_index,
):
    with pytest.raises(
        ValueError,
        match="start_session_index must be an integer session index",
    ):
        build_track_rows_from_matches(
            ("s1", "s2"),
            [np.array([[0, 1]], dtype=int)],
            start_roi_indices=np.array([0]),
            start_session_index=bad_start_session_index,
        )


@pytest.mark.parametrize(
    "bad_start_session_index",
    [True, np.bool_(False), 1.5, np.inf, "1"],
)
def test_build_track_rows_from_bundles_rejects_malformed_start_session_index(
    bad_start_session_index,
):
    with pytest.raises(
        ValueError,
        match="start_session_index must be an integer session index",
    ):
        build_track_rows_from_bundles(
            [_Bundle([[0.0]])],
            start_session_index=bad_start_session_index,
        )


def test_solve_bundle_linear_assignment_uses_default_cost_gate():
    result = solve_bundle_linear_assignment(_Bundle([[0.0, 100.0], [100.0, 100.0]]))

    npt.assert_array_equal(result.reference_roi_indices, np.array([10]))
    npt.assert_array_equal(result.measurement_roi_indices, np.array([100]))
    npt.assert_array_equal(result.costs, np.array([0.0]))


def test_solve_bundle_linear_assignment_can_disable_cost_gate():
    result = solve_bundle_linear_assignment(
        _Bundle([[0.0, 100.0], [100.0, 100.0]]),
        max_cost=None,
    )

    npt.assert_array_equal(result.reference_roi_indices, np.array([10, 20]))
    npt.assert_array_equal(result.measurement_roi_indices, np.array([100, 200]))
    npt.assert_array_equal(result.costs, np.array([0.0, 100.0]))


def test_solve_bundle_linear_assignment_with_disabled_gate_filters_invalid_costs():
    result = solve_bundle_linear_assignment(
        _Bundle([[0.0, np.nan], [np.inf, 1.0]]),
        max_cost=None,
    )

    npt.assert_array_equal(result.reference_roi_indices, np.array([10, 20]))
    npt.assert_array_equal(result.measurement_roi_indices, np.array([100, 200]))
    npt.assert_array_equal(result.costs, np.array([0.0, 1.0]))


def test_solve_bundle_linear_assignment_rejects_short_reference_roi_indices():
    with pytest.raises(ValueError, match="reference_roi_indices length"):
        solve_bundle_linear_assignment(
            _Bundle(
                [[0.0, 1.0], [1.0, 0.0]],
                reference_roi_indices=[10],
            )
        )


def test_solve_bundle_linear_assignment_rejects_extra_measurement_roi_indices():
    with pytest.raises(ValueError, match="measurement_roi_indices length"):
        solve_bundle_linear_assignment(
            _Bundle(
                [[0.0, 1.0], [1.0, 0.0]],
                measurement_roi_indices=[100, 200, 300],
            )
        )


def test_solve_bundle_linear_assignment_rejects_fractional_bundle_roi_indices():
    with pytest.raises(ValueError, match="reference_roi_indices"):
        solve_bundle_linear_assignment(
            _Bundle(
                [[0.0, 1.0], [1.0, 0.0]],
                reference_roi_indices=[10.5, 20],
            )
        )


def test_solve_bundle_linear_assignment_gates_before_hungarian():
    result = solve_bundle_linear_assignment(_Bundle([[6.0, 7.0], [0.0, 6.0]]))

    npt.assert_array_equal(result.reference_roi_indices, np.array([10, 20]))
    npt.assert_array_equal(result.measurement_roi_indices, np.array([100, 200]))
    npt.assert_array_equal(result.costs, np.array([6.0, 6.0]))


def test_build_track_rows_from_bundles_uses_default_cost_gate():
    session_names, rows, match_results = build_track_rows_from_bundles(
        [_Bundle([[0.0, 100.0], [100.0, 100.0]])]
    )

    assert session_names == ("s1", "s2")
    npt.assert_array_equal(rows, np.array([[10, 100], [20, -1]]))
    assert match_results[0].n_matches == 1
    npt.assert_array_equal(
        match_results[0].reference_roi_indices,
        np.array([10]),
    )
    npt.assert_array_equal(
        match_results[0].measurement_roi_indices,
        np.array([100]),
    )
    npt.assert_array_equal(match_results[0].costs, np.array([0.0]))


def test_build_track_rows_from_bundles_can_disable_cost_gate():
    session_names, rows, match_results = build_track_rows_from_bundles(
        [_Bundle([[0.0, 100.0], [100.0, 100.0]])],
        max_cost=None,
    )

    assert session_names == ("s1", "s2")
    npt.assert_array_equal(rows, np.array([[10, 100], [20, 200]]))
    assert match_results[0].n_matches == 2
    npt.assert_array_equal(match_results[0].costs, np.array([0.0, 100.0]))
