from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.association import absence_model


class Plane:
    def __init__(self, n_rois: object) -> None:
        self.n_rois = n_rois
        self.cell_probabilities = None


@pytest.mark.parametrize("value", [np.array([2]), np.array([2.0]), np.array([[2]])])
def test_roi_count_must_be_scalar(value: object) -> None:
    with pytest.raises(ValueError, match=r"plane\.n_rois"):
        absence_model.absence_cost_vector(Plane(value))
