from __future__ import annotations

import numpy as np
import pytest

from bayescatrack import CalciumPlaneData


def _single_roi_plane() -> CalciumPlaneData:
    roi_masks = np.zeros((1, 5, 5), dtype=bool)
    roi_masks[0, 2:4, 2:4] = True
    return CalciumPlaneData(roi_masks=roi_masks)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"radial_profile_weight": np.nan}, "radial_profile_weight must be a finite non-negative value"),
        ({"orientation_weight": np.inf}, "orientation_weight must be a finite non-negative value"),
        ({"eccentricity_weight": True}, "eccentricity_weight must be a finite non-negative value"),
        ({"compactness_weight": -1.0}, "compactness_weight must be a finite non-negative value"),
        ({"border_proximity_weight": np.nan}, "border_proximity_weight must be a finite non-negative value"),
        ({"ambiguity_margin_weight": np.inf}, "ambiguity_margin_weight must be a finite non-negative value"),
        ({"shape_descriptor_components": 1}, "shape_descriptor_components must be a boolean"),
        ({"ambiguity_margin_components": "true"}, "ambiguity_margin_components must be a boolean"),
        ({"candidate_include_column_top_k": 1}, "candidate_include_column_top_k must be a boolean"),
        ({"large_cost": np.nan}, "large_cost must be a finite positive value"),
    ],
)
def test_advanced_pairwise_runtime_knobs_reject_invalid_scalars(
    kwargs: dict[str, object], message: str
) -> None:
    reference = _single_roi_plane()
    measurement = _single_roi_plane()
    base_kwargs = {
        "centroid_weight": 0.0,
        "iou_weight": 0.0,
        "mask_cosine_weight": 0.0,
        "area_weight": 0.0,
        "roi_feature_weight": 0.0,
        "cell_probability_weight": 0.0,
    }
    base_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        reference.build_pairwise_cost_matrix(measurement, **base_kwargs)


def test_advanced_pairwise_runtime_knobs_accept_finite_values() -> None:
    reference = _single_roi_plane()
    measurement = _single_roi_plane()

    cost, components = reference.build_pairwise_cost_matrix(
        measurement,
        centroid_weight=0.0,
        iou_weight=0.0,
        mask_cosine_weight=0.0,
        area_weight=0.0,
        roi_feature_weight=0.0,
        cell_probability_weight=0.0,
        shape_descriptor_components=True,
        radial_profile_weight=0.0,
        orientation_weight=0.0,
        eccentricity_weight=0.0,
        compactness_weight=0.0,
        border_proximity_weight=0.0,
        ambiguity_margin_components=True,
        ambiguity_margin_weight=0.0,
        candidate_include_column_top_k=True,
        large_cost=1.0e6,
        return_components=True,
    )

    assert cost.shape == (1, 1)
    assert np.all(np.isfinite(cost))
    assert "radial_profile_cost" in components
    assert "ambiguity_margin_cost" in components
