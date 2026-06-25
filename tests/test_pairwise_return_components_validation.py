"""Regression tests for strict pairwise-cost return-component controls."""

from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.core.bridge import CalciumPlaneData


def _single_roi_plane(mask: np.ndarray) -> CalciumPlaneData:
    roi_masks = np.asarray(mask, dtype=float).reshape(1, 1, -1)
    return CalciumPlaneData(roi_masks=roi_masks)


def _soft_overlap_kwargs() -> dict[str, float | int]:
    return {
        "centroid_weight": 0.0,
        "iou_weight": 0.0,
        "mask_cosine_weight": 0.0,
        "area_weight": 0.0,
        "roi_feature_weight": 0.0,
        "cell_probability_weight": 0.0,
        "soft_iou_weight": 1.0,
        "soft_iou_radius": 1,
    }


@pytest.mark.parametrize(
    "value",
    ["false", "true", 0, 1, None, np.array(True), np.array([True])],
)
def test_pairwise_cost_rejects_ambiguous_return_components(value: object) -> None:
    reference = _single_roi_plane(np.array([1.0, 0.0, 0.0]))
    measurement = _single_roi_plane(np.array([1.0, 0.0, 0.0]))

    with pytest.raises(ValueError, match="return_components must be a boolean"):
        reference.build_pairwise_cost_matrix(
            measurement,
            return_components=value,
            **_soft_overlap_kwargs(),
        )


def test_pairwise_cost_accepts_numpy_bool_return_components() -> None:
    reference = _single_roi_plane(np.array([1.0, 0.0, 0.0]))
    measurement = _single_roi_plane(np.array([1.0, 0.0, 0.0]))

    cost, components = reference.build_pairwise_cost_matrix(
        measurement,
        return_components=np.bool_(True),
        **_soft_overlap_kwargs(),
    )
    only_cost = reference.build_pairwise_cost_matrix(
        measurement,
        return_components=np.bool_(False),
        **_soft_overlap_kwargs(),
    )

    assert cost.shape == (1, 1)
    assert components["pairwise_cost_matrix"].shape == (1, 1)
    assert isinstance(only_cost, np.ndarray)
