"""Tests for split Suite2p ROI-stat pairwise features."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
from bayescatrack.association.calibrated_costs import (
    DEFAULT_ASSOCIATION_FEATURES,
    SPLIT_ROI_STAT_FEATURES,
    pairwise_feature_tensor,
)
from bayescatrack.core.bridge import CalciumPlaneData
from bayescatrack.core._roi_stat_features import _normalize_roi_feature_names


def _masks(n_rois: int) -> np.ndarray:
    masks = np.zeros((n_rois, 8, 8), dtype=bool)
    for roi_index in range(n_rois):
        row_slice = slice(roi_index, roi_index + 2)
        col_slice = slice(roi_index, roi_index + 2)
        masks[roi_index, row_slice, col_slice] = True
    return masks


def _plane(roi_features: dict[str, list[float]]) -> CalciumPlaneData:
    n_rois = len(next(iter(roi_features.values())))
    return CalciumPlaneData(
        roi_masks=_masks(n_rois),
        roi_features={
            feature_name: np.asarray(values, dtype=float)
            for feature_name, values in roi_features.items()
        },
    )


def _reference_plane() -> CalciumPlaneData:
    return _plane(
        {
            "radius": [2.0, np.nan],
            "aspect_ratio": [1.0, 2.0],
            "compact": [0.5, 0.6],
            "footprint": [20.0, 24.0],
            "skew": [0.1, -0.2],
            "std": [3.0, 4.0],
            "npix": [100.0, 200.0],
            "npix_norm": [0.1, 0.2],
        }
    )


def _measurement_plane() -> CalciumPlaneData:
    return _plane(
        {
            "radius": [2.0, 4.0],
            "aspect_ratio": [3.0, 2.0],
            "compact": [0.5, 0.9],
            "footprint": [20.0, 40.0],
            "skew": [0.1, 0.0],
            "std": [3.0, 8.0],
            "npix": [100.0, 400.0],
            "npix_norm": [0.1, 0.3],
        }
    )


def test_pairwise_components_expose_split_suite2p_roi_stats() -> None:
    _, components = _reference_plane().build_pairwise_cost_matrix(
        _measurement_plane(),
        return_components=True,
    )

    for feature_name in SPLIT_ROI_STAT_FEATURES:
        assert feature_name in components
        assert components[feature_name].shape == (2, 2)
        assert np.all(np.isfinite(components[feature_name]))

    npt.assert_allclose(
        components["abs_log_radius_ratio"][0, :],
        np.array([0.0, np.log(2.0)]),
    )
    npt.assert_allclose(
        components["abs_log_npix_ratio"][0, :],
        np.array([0.0, np.log(4.0)]),
    )

    npt.assert_allclose(components["missing_radius_indicator"][0, :], [0.0, 0.0])
    npt.assert_allclose(components["missing_radius_indicator"][1, :], [1.0, 1.0])
    npt.assert_allclose(components["missing_stat_indicator"][1, :], [1.0, 1.0])

    assert components["abs_aspect_ratio_diff"][1, 1] == 0.0
    assert (
        components["abs_aspect_ratio_diff"][0, 0]
        > components["abs_aspect_ratio_diff"][0, 1]
        > 0.0
    )


def test_split_roi_stat_features_are_usable_by_named_feature_schema() -> None:
    _, components = _reference_plane().build_pairwise_cost_matrix(
        _measurement_plane(),
        return_components=True,
    )

    features = pairwise_feature_tensor(
        components,
        feature_names=SPLIT_ROI_STAT_FEATURES,
    )

    assert features.shape == (2, 2, len(SPLIT_ROI_STAT_FEATURES))
    assert np.all(np.isfinite(features))


def test_default_calibrated_features_use_split_roi_stats_not_scalar_summary() -> None:
    assert "roi_feature_cost" not in DEFAULT_ASSOCIATION_FEATURES
    assert "cell_probability_cost" in DEFAULT_ASSOCIATION_FEATURES
    for feature_name in SPLIT_ROI_STAT_FEATURES:
        assert feature_name in DEFAULT_ASSOCIATION_FEATURES
    assert len(DEFAULT_ASSOCIATION_FEATURES) == len(set(DEFAULT_ASSOCIATION_FEATURES))


def test_missing_indicators_cover_planes_without_suite2p_stats() -> None:
    reference = CalciumPlaneData(roi_masks=_masks(1))
    measurement = CalciumPlaneData(roi_masks=_masks(1))

    _, components = reference.build_pairwise_cost_matrix(
        measurement,
        return_components=True,
    )

    npt.assert_allclose(components["missing_radius_indicator"], [[1.0]])
    npt.assert_allclose(components["missing_stat_indicator"], [[1.0]])
    assert np.all(
        np.isfinite(
            pairwise_feature_tensor(components, feature_names=SPLIT_ROI_STAT_FEATURES)
        )
    )


def test_split_roi_stat_feature_names_accept_direct_comma_string() -> None:
    assert _normalize_roi_feature_names(
        "abs_log_radius_ratio, abs_skew_diff"
    ) == ["radius", "skew"]


@pytest.mark.parametrize(
    "feature_names",
    [
        ("abs_log_radius_ratio", True),
        ("abs_log_radius_ratio", ""),
    ],
)
def test_split_roi_stat_feature_names_reject_invalid_entries(feature_names) -> None:
    with pytest.raises(ValueError, match="feature_names"):
        _normalize_roi_feature_names(feature_names)


@pytest.mark.parametrize("similarity_epsilon", [True, float("nan"), float("inf"), 0.0])
def test_split_roi_stat_wrapper_rejects_invalid_similarity_epsilon(
    similarity_epsilon,
) -> None:
    with pytest.raises(ValueError, match="similarity_epsilon"):
        _reference_plane().build_pairwise_cost_matrix(
            _measurement_plane(), similarity_epsilon=similarity_epsilon
        )
