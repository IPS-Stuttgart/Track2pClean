"""Regression tests for ROI feature-name normalization."""

from __future__ import annotations

import numpy as np

from bayescatrack import CalciumPlaneData


def _two_roi_plane(feature_values: tuple[float, float]) -> CalciumPlaneData:
    roi_masks = np.zeros((2, 3, 3), dtype=bool)
    roi_masks[0, 0, 0] = True
    roi_masks[1, 2, 2] = True
    return CalciumPlaneData(
        roi_masks=roi_masks,
        roi_features={"radius": np.asarray(feature_values, dtype=float)},
    )


def _feature_only_cost(
    reference: CalciumPlaneData,
    measurement: CalciumPlaneData,
    feature_names: object,
) -> np.ndarray:
    return reference.build_pairwise_cost_matrix(
        measurement,
        centroid_weight=0.0,
        iou_weight=0.0,
        mask_cosine_weight=0.0,
        area_weight=0.0,
        roi_feature_weight=1.0,
        feature_names=feature_names,
        cell_probability_weight=0.0,
    )


def test_single_roi_feature_name_string_matches_one_element_sequence() -> None:
    reference = _two_roi_plane((1.0, 2.0))
    measurement = _two_roi_plane((2.0, 1.0))

    string_cost = _feature_only_cost(reference, measurement, "radius")
    sequence_cost = _feature_only_cost(reference, measurement, ["radius"])

    np.testing.assert_allclose(string_cost, sequence_cost)
    assert string_cost[0, 0] > 0.0
    assert string_cost[0, 1] == 0.0
