from __future__ import annotations

import numpy as np
import pytest
from bayescatrack import CalciumPlaneData


def _masks(n_rois: int = 2) -> np.ndarray:
    masks = np.zeros((n_rois, 4, 4), dtype=bool)
    for roi in range(n_rois):
        masks[roi, roi, roi] = True
    return masks


@pytest.mark.parametrize(
    ("roi_indices", "match"),
    [
        ([False, True], "boolean"),
        (["0", "1"], "integer ROI indices"),
        ([0.0, 1.5], "integer ROI indices"),
        ([0, -1], "non-negative"),
        ([1, 1], "unique"),
    ],
)
def test_calcium_plane_data_rejects_invalid_roi_indices(roi_indices, match):
    with pytest.raises(ValueError, match=match):
        CalciumPlaneData(_masks(), roi_indices=np.asarray(roi_indices, dtype=object))


def test_calcium_plane_data_keeps_integer_roi_indices():
    plane = CalciumPlaneData(_masks(), roi_indices=np.asarray([5, 8], dtype=np.int64))

    np.testing.assert_array_equal(plane.roi_indices, np.asarray([5, 8]))
