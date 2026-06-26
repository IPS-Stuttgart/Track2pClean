from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.matching import (
    build_track_rows_from_bundles,
    solve_bundle_linear_assignment,
)


class _Bundle:
    def __init__(self) -> None:
        self.pairwise_cost_matrix = np.asarray([[0.0]], dtype=float)
        self.reference_session_name = "s1"
        self.measurement_session_name = "s2"
        self.reference_roi_indices = np.asarray([0])
        self.measurement_roi_indices = np.asarray([1])


@pytest.mark.parametrize("bad_max_cost", [True, False, np.bool_(True), np.bool_(False)])
def test_solve_bundle_linear_assignment_rejects_boolean_max_cost(bad_max_cost) -> None:
    with pytest.raises(ValueError, match="max_cost must be None or a finite non-negative scalar"):
        solve_bundle_linear_assignment(_Bundle(), max_cost=bad_max_cost)


@pytest.mark.parametrize("bad_max_cost", [True, False, np.bool_(True), np.bool_(False)])
def test_build_track_rows_from_bundles_rejects_boolean_max_cost(bad_max_cost) -> None:
    with pytest.raises(ValueError, match="max_cost must be None or a finite non-negative scalar"):
        build_track_rows_from_bundles([_Bundle()], max_cost=bad_max_cost)
