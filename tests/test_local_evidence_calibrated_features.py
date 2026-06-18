"""Tests for calibrated local-evidence feature presets."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
from bayescatrack.association.calibrated_costs import (
    DEFAULT_ASSOCIATION_FEATURES,
    LOCAL_EVIDENCE_ASSOCIATION_FEATURES,
    pairwise_feature_tensor,
)
from bayescatrack.core.bridge import CalciumPlaneData


def _masks(*, shift_second_roi: bool = False) -> np.ndarray:
    masks = np.zeros((2, 9, 9), dtype=float)
    masks[0, 1:4, 1:4] = 1.0
    row_offset = 1 if shift_second_roi else 0
    masks[1, 5 + row_offset : 7 + row_offset, 5:7] = 1.0
    return masks


def _plane(
    *, shift_second_roi: bool = False, with_fov: bool = True
) -> CalciumPlaneData:
    fov = None
    if with_fov:
        y_grid, x_grid = np.mgrid[:9, :9]
        fov = x_grid.astype(float) + 0.25 * y_grid.astype(float)
    return CalciumPlaneData(
        roi_masks=_masks(shift_second_roi=shift_second_roi),
        fov=fov,
    )


def test_local_evidence_feature_preset_builds_named_tensor() -> None:
    _, components = _plane().build_pairwise_cost_matrix(
        _plane(shift_second_roi=True),
        local_evidence_components=True,
        return_components=True,
    )

    for feature_name in LOCAL_EVIDENCE_ASSOCIATION_FEATURES:
        assert feature_name in components
        assert components[feature_name].shape == (2, 2)
        assert np.all(np.isfinite(components[feature_name]))

    features = pairwise_feature_tensor(
        components,
        feature_names=LOCAL_EVIDENCE_ASSOCIATION_FEATURES,
    )

    assert features.shape == (2, 2, len(LOCAL_EVIDENCE_ASSOCIATION_FEATURES))
    assert np.all(np.isfinite(features))

    weighted_dice_index = LOCAL_EVIDENCE_ASSOCIATION_FEATURES.index(
        "weighted_dice_cost"
    )
    assert features[0, 0, weighted_dice_index] < features[0, 1, weighted_dice_index]


def test_local_evidence_features_are_finite_without_fov() -> None:
    _, components = _plane(with_fov=False).build_pairwise_cost_matrix(
        _plane(with_fov=False),
        local_evidence_components=True,
        return_components=True,
    )

    features = pairwise_feature_tensor(
        components,
        feature_names=LOCAL_EVIDENCE_ASSOCIATION_FEATURES,
    )

    assert np.all(np.isfinite(features))
    npt.assert_allclose(components["image_patch_valid"], np.zeros((2, 2)))
    npt.assert_allclose(components["image_patch_cost"], np.zeros((2, 2)))


def test_default_features_keep_local_evidence_opt_in() -> None:
    assert not (
        set(DEFAULT_ASSOCIATION_FEATURES) & set(LOCAL_EVIDENCE_ASSOCIATION_FEATURES)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weighted_dice_weight": True},
        {"overlap_fraction_weight": float("nan")},
        {"containment_weight": float("inf")},
        {"distance_transform_weight": -1.0},
        {"image_patch_weight": True},
        {"neighbor_constellation_weight": float("nan")},
        {"centroid_rank_weight": float("inf")},
        {"local_evidence_components": "true"},
        {"normalize_weighted_overlap": 1},
        {"patch_radius": True},
        {"patch_radius": 1.5},
        {"patch_radius": -1},
        {"neighbor_k": False},
        {"neighbor_k": 1.5},
        {"neighbor_k": 0},
        {"similarity_epsilon": True},
        {"similarity_epsilon": float("nan")},
        {"similarity_epsilon": 0.0},
        {"large_cost": False},
        {"large_cost": float("inf")},
        {"large_cost": 0.0},
        {"weighted_centroids": "false"},
    ],
)
def test_local_evidence_pairwise_wrapper_rejects_invalid_controls(kwargs) -> None:
    field_name = next(iter(kwargs))

    with pytest.raises(ValueError, match=field_name):
        _plane().build_pairwise_cost_matrix(_plane(), **kwargs)
