from __future__ import annotations

import numpy as np
import pytest
from bayescatrack import CalciumPlaneData


def _single_roi_plane() -> CalciumPlaneData:
    masks = np.zeros((1, 4, 4), dtype=bool)
    masks[0, 1:3, 1:3] = True
    return CalciumPlaneData(masks)


def test_pairwise_cost_rejects_boolean_weight() -> None:
    plane = _single_roi_plane()

    with pytest.raises(
        ValueError, match="centroid_weight must be a finite non-negative value"
    ):
        plane.build_pairwise_cost_matrix(plane, centroid_weight=True)


def test_pairwise_cost_rejects_nonfinite_large_cost() -> None:
    plane = _single_roi_plane()

    with pytest.raises(ValueError, match="large_cost must be a finite positive value"):
        plane.build_pairwise_cost_matrix(plane, large_cost=np.inf)


def test_pairwise_cost_rejects_zero_centroid_scale() -> None:
    plane = _single_roi_plane()

    with pytest.raises(
        ValueError, match="centroid_scale must be a finite positive value"
    ):
        plane.build_pairwise_cost_matrix(plane, centroid_scale=0.0)
