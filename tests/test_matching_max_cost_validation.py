from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
from bayescatrack.matching import (
    build_track_rows_from_bundles,
    solve_bundle_linear_assignment,
)


class _Bundle:
    def __init__(self, costs: object):
        cost_matrix = np.asarray(costs, dtype=float)
        self.pairwise_cost_matrix = cost_matrix
        self.reference_session_name = "s1"
        self.measurement_session_name = "s2"
        self.reference_roi_indices = np.arange(cost_matrix.shape[0], dtype=int)
        self.measurement_roi_indices = np.arange(cost_matrix.shape[1], dtype=int) + 100


@pytest.mark.parametrize(
    "bad_max_cost",
    [
        True,
        np.bool_(False),
        np.array(True),
        np.array([1.0]),
        "1.0",
        -0.1,
        np.nan,
        np.inf,
    ],
)
def test_solve_bundle_linear_assignment_rejects_ambiguous_max_cost(
    bad_max_cost: object,
) -> None:
    with pytest.raises(ValueError, match="max_cost"):
        solve_bundle_linear_assignment(
            _Bundle([[0.0, 10.0], [10.0, 1.0]]),
            max_cost=bad_max_cost,
        )


def test_build_track_rows_from_bundles_rejects_ambiguous_max_cost() -> None:
    with pytest.raises(ValueError, match="max_cost"):
        build_track_rows_from_bundles(
            [_Bundle([[0.0, 10.0], [10.0, 1.0]])],
            max_cost=True,
        )


def test_solve_bundle_linear_assignment_accepts_numpy_scalar_max_cost() -> None:
    result = solve_bundle_linear_assignment(
        _Bundle([[0.0, 10.0], [10.0, 1.0]]),
        max_cost=np.array(1.0),
    )

    npt.assert_array_equal(result.reference_roi_indices, np.array([0, 1]))
    npt.assert_array_equal(result.measurement_roi_indices, np.array([100, 101]))
    npt.assert_array_equal(result.costs, np.array([0.0, 1.0]))
