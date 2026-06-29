import numpy as np
import pytest
from bayescatrack import CalciumPlaneData


def _single_roi_mask() -> np.ndarray:
    masks = np.zeros((1, 5, 5), dtype=bool)
    masks[0, 2:4, 2:4] = True
    return masks


def test_zero_roi_feature_weight_skips_incompatible_unused_features():
    reference = CalciumPlaneData(
        roi_masks=_single_roi_mask(),
        roi_features={"embedding": np.zeros((1, 2), dtype=float)},
    )
    measurement = CalciumPlaneData(
        roi_masks=_single_roi_mask(),
        roi_features={"embedding": np.zeros((1, 3), dtype=float)},
    )

    cost = reference.build_pairwise_cost_matrix(
        measurement,
        centroid_weight=0.0,
        iou_weight=0.0,
        mask_cosine_weight=0.0,
        area_weight=0.0,
        roi_feature_weight=0.0,
        cell_probability_weight=0.0,
    )

    assert cost.shape == (1, 1)
    assert np.all(np.isfinite(cost))
    assert float(cost[0, 0]) == 0.0


def test_return_components_keeps_incompatible_feature_diagnostic_validation():
    reference = CalciumPlaneData(
        roi_masks=_single_roi_mask(),
        roi_features={"embedding": np.zeros((1, 2), dtype=float)},
    )
    measurement = CalciumPlaneData(
        roi_masks=_single_roi_mask(),
        roi_features={"embedding": np.zeros((1, 3), dtype=float)},
    )

    with pytest.raises(ValueError, match="incompatible trailing dimensions"):
        reference.build_pairwise_cost_matrix(
            measurement,
            centroid_weight=0.0,
            iou_weight=0.0,
            mask_cosine_weight=0.0,
            area_weight=0.0,
            roi_feature_weight=0.0,
            cell_probability_weight=0.0,
            return_components=True,
        )
