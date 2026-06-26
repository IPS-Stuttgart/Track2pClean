import numpy as np
import pytest

from bayescatrack.matching import solve_bundle_linear_assignment


class _Bundle:
    def __init__(self, reference_roi_indices, measurement_roi_indices):
        self.pairwise_cost_matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        self.reference_session_name = "s1"
        self.measurement_session_name = "s2"
        self.reference_roi_indices = np.asarray(reference_roi_indices, dtype=object)
        self.measurement_roi_indices = np.asarray(measurement_roi_indices, dtype=object)


@pytest.mark.parametrize(
    ("reference_roi_indices", "measurement_roi_indices", "match"),
    [
        ([10, 10], [100, 200], r"bundle\.reference_roi_indices must contain unique ROI indices"),
        ([10, 20], [100, 100], r"bundle\.measurement_roi_indices must contain unique ROI indices"),
    ],
)
def test_solve_bundle_linear_assignment_rejects_duplicate_bundle_roi_indices(
    reference_roi_indices,
    measurement_roi_indices,
    match,
):
    with pytest.raises(ValueError, match=match):
        solve_bundle_linear_assignment(
            _Bundle(reference_roi_indices, measurement_roi_indices),
            max_cost=None,
        )
