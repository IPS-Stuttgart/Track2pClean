import numpy as np
import numpy.testing as npt
from bayescatrack import CalciumPlaneData


def test_bare_string_feature_name_is_treated_as_single_roi_feature():
    roi_masks = np.zeros((1, 3, 3), dtype=bool)
    roi_masks[0, 1, 1] = True
    reference = CalciumPlaneData(
        roi_masks=roi_masks,
        roi_features={"radius": np.array([1.0], dtype=float)},
    )
    measurement = CalciumPlaneData(
        roi_masks=roi_masks.copy(),
        roi_features={"radius": np.array([3.0], dtype=float)},
    )
    kwargs = {
        "centroid_weight": 0.0,
        "iou_weight": 0.0,
        "mask_cosine_weight": 0.0,
        "area_weight": 0.0,
        "roi_feature_weight": 1.0,
        "cell_probability_weight": 0.0,
    }

    string_cost, string_components = reference.build_pairwise_cost_matrix(
        measurement,
        feature_names="radius",
        return_components=True,
        **kwargs,
    )
    tuple_cost, tuple_components = reference.build_pairwise_cost_matrix(
        measurement,
        feature_names=("radius",),
        return_components=True,
        **kwargs,
    )

    assert string_cost[0, 0] > 0.0
    npt.assert_allclose(string_cost, tuple_cost)
    npt.assert_allclose(
        string_components["roi_feature_cost"],
        tuple_components["roi_feature_cost"],
    )
