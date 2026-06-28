from __future__ import annotations

import numpy as np
import pytest
from bayescatrack import CalciumPlaneData


def _plane() -> CalciumPlaneData:
    roi_masks = np.zeros((1, 3, 3), dtype=bool)
    roi_masks[0, 1, 1] = True
    return CalciumPlaneData(roi_masks)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"weighted_dice_weight": np.asarray([1.0])},
            "weighted_dice_weight must be a finite non-negative value",
        ),
        (
            {"overlap_fraction_weight": np.asarray([1.0])},
            "overlap_fraction_weight must be a finite non-negative value",
        ),
        (
            {"containment_weight": np.asarray([1.0])},
            "containment_weight must be a finite non-negative value",
        ),
        (
            {"distance_transform_weight": np.asarray([1.0])},
            "distance_transform_weight must be a finite non-negative value",
        ),
        (
            {"image_patch_weight": np.asarray([1.0])},
            "image_patch_weight must be a finite non-negative value",
        ),
        (
            {"neighbor_constellation_weight": np.asarray([1.0])},
            "neighbor_constellation_weight must be a finite non-negative value",
        ),
        (
            {"centroid_rank_weight": np.asarray([1.0])},
            "centroid_rank_weight must be a finite non-negative value",
        ),
        ({"patch_radius": np.asarray([2])}, "patch_radius must be an integer"),
        ({"neighbor_k": np.asarray([2])}, "neighbor_k must be an integer"),
        (
            {"local_evidence_components": np.asarray(True)},
            "local_evidence_components must be a boolean",
        ),
        (
            {"normalize_weighted_overlap": np.asarray(True)},
            "normalize_weighted_overlap must be a boolean",
        ),
        (
            {"return_components": np.asarray(True)},
            "return_components must be a boolean",
        ),
    ],
)
def test_pairwise_local_evidence_rejects_array_control_values(kwargs, message):
    plane = _plane()
    with pytest.raises(ValueError, match=message):
        plane.build_pairwise_cost_matrix(plane, **kwargs)


def test_pairwise_local_evidence_accepts_zero_dimensional_numeric_controls():
    plane = _plane()

    cost, components = plane.build_pairwise_cost_matrix(
        plane,
        weighted_dice_weight=np.asarray(0.0),
        overlap_fraction_weight=np.asarray(0.0),
        containment_weight=np.asarray(0.0),
        distance_transform_weight=np.asarray(0.0),
        image_patch_weight=np.asarray(0.0),
        neighbor_constellation_weight=np.asarray(0.0),
        centroid_rank_weight=np.asarray(0.0),
        patch_radius=np.asarray(0),
        neighbor_k=np.asarray(1),
        local_evidence_components=True,
        return_components=True,
    )

    assert cost.shape == (1, 1)
    assert "weighted_dice_cost" in components
