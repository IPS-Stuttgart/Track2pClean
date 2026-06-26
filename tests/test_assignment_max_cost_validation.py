import numpy as np
import numpy.testing as npt
import pytest

from bayescatrack.matching import (
    build_track_rows_from_bundles,
    solve_bundle_linear_assignment,
)


class _Bundle:
    def __init__(self, costs):
        self.pairwise_cost_matrix = np.asarray(costs, dtype=float)
        self.reference_session_name = "s1"
        self.measurement_session_name = "s2"
        self.reference_roi_indices = np.arange(
            self.pairwise_cost_matrix.shape[0], dtype=int
        )
        self.measurement_roi_indices = 100 + np.arange(
            self.pairwise_cost_matrix.shape[1], dtype=int
        )


@pytest.mark.parametrize("bad_max_cost", [True, False, np.bool_(True), np.bool_(False)])
def test_solve_bundle_linear_assignment_rejects_boolean_max_cost(bad_max_cost):
    with pytest.raises(
        ValueError,
        match=r"max_cost must.*finite non-negative",
    ):
        solve_bundle_linear_assignment(_Bundle([[0.0]]), max_cost=bad_max_cost)


@pytest.mark.parametrize("bad_max_cost", [True, np.bool_(False)])
def test_build_track_rows_from_bundles_rejects_boolean_max_cost(bad_max_cost):
    with pytest.raises(
        ValueError,
        match=r"max_cost must.*finite non-negative",
    ):
        build_track_rows_from_bundles([_Bundle([[0.0]])], max_cost=bad_max_cost)


def test_assignment_max_cost_validation_preserves_none_gate():
    result = solve_bundle_linear_assignment(
        _Bundle([[0.0, 100.0], [100.0, 100.0]]),
        max_cost=None,
    )

    npt.assert_array_equal(result.reference_roi_indices, np.array([0, 1]))
    npt.assert_array_equal(result.measurement_roi_indices, np.array([100, 101]))
    npt.assert_array_equal(result.costs, np.array([0.0, 100.0]))
