from __future__ import annotations

import numpy as np
import numpy.testing as npt

from bayescatrack import CalciumPlaneData


def _single_roi_plane_without_fov() -> CalciumPlaneData:
    masks = np.zeros((1, 5, 5), dtype=bool)
    masks[0, 2:4, 2:4] = True
    return CalciumPlaneData(masks, source="synthetic")


def test_invalid_image_patch_evidence_is_neutral_not_zero_cost():
    reference = _single_roi_plane_without_fov()
    measurement = _single_roi_plane_without_fov()

    cost_matrix, components = reference.build_pairwise_cost_matrix(
        measurement,
        centroid_weight=0.0,
        mahalanobis_weight=0.0,
        iou_weight=0.0,
        mask_cosine_weight=0.0,
        area_weight=0.0,
        roi_feature_weight=0.0,
        cell_probability_weight=0.0,
        image_patch_weight=1.0,
        return_components=True,
    )

    npt.assert_array_equal(components["image_patch_valid"], np.array([[0.0]]))
    npt.assert_allclose(components["image_patch_cost"], np.array([[0.5]]))
    npt.assert_allclose(cost_matrix, np.array([[0.5]]))
