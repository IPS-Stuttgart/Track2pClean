"""Regression tests for split ROI-stat and local patch evidence components."""

from __future__ import annotations

import numpy as np

from bayescatrack.association.calibrated_costs import pairwise_feature_tensor
from bayescatrack.core.bridge import CalciumPlaneData


def _toy_plane(*, radius_offset: float = 0.0) -> CalciumPlaneData:
    masks = np.zeros((2, 9, 9), dtype=bool)
    masks[0, 2:4, 2:4] = True
    masks[1, 5:7, 5:7] = True

    yy, xx = np.mgrid[:9, :9]
    fov = np.asarray(yy + 2.0 * xx, dtype=float)
    fov[2:4, 2:4] += 5.0
    fov[5:7, 5:7] -= 4.0

    return CalciumPlaneData(
        roi_masks=masks,
        fov=fov,
        roi_indices=np.arange(2, dtype=int),
        roi_features={
            "radius": np.array([2.0 + radius_offset, 5.0 + radius_offset]),
            "aspect_ratio": np.array([1.0, 1.8]),
            "compact": np.array([0.9, 0.4]),
            "footprint": np.array([4.0, 4.0]),
            "skew": np.array([0.1, 0.3]),
            "std": np.array([1.0, 2.0]),
            "npix": np.array([4.0, 4.0]),
            "npix_norm": np.array([0.5, 0.5]),
        },
    )


def test_pairwise_components_include_split_roi_stats_and_local_patch_evidence() -> None:
    reference_plane = _toy_plane()
    measurement_plane = _toy_plane(radius_offset=0.5)

    _, components = reference_plane.build_pairwise_cost_matrix(
        measurement_plane,
        return_components=True,
        roi_feature_weight=0.0,
        local_patch_radius=1,
    )

    expected_feature_components = {
        "roi_stat_radius_cost",
        "roi_stat_aspect_ratio_cost",
        "roi_stat_compact_cost",
        "roi_stat_footprint_cost",
        "roi_stat_skew_cost",
        "roi_stat_std_cost",
        "roi_stat_npix_cost",
        "roi_stat_npix_norm_cost",
        "local_patch_correlation",
        "local_patch_cost",
        "local_patch_available",
    }
    assert expected_feature_components.issubset(components)
    for component_name in expected_feature_components:
        assert components[component_name].shape == (2, 2)
        assert np.all(np.isfinite(components[component_name]))

    assert np.all(components["local_patch_available"] == 1.0)
    assert components["roi_stat_radius_cost"][0, 0] < components["roi_stat_radius_cost"][0, 1]


def test_split_roi_and_local_patch_features_are_schema_addressable() -> None:
    reference_plane = _toy_plane()
    measurement_plane = _toy_plane(radius_offset=0.25)
    _, components = reference_plane.build_pairwise_cost_matrix(
        measurement_plane,
        return_components=True,
        roi_feature_weight=0.0,
        local_patch_radius=1,
    )

    features = pairwise_feature_tensor(
        components,
        feature_names=(
            "roi_stat_radius_cost",
            "roi_stat_aspect_ratio_cost",
            "local_patch_cost",
            "local_patch_available",
        ),
    )
    assert features.shape == (2, 2, 4)
    assert np.all(np.isfinite(features))
