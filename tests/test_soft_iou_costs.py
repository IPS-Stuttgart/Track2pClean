"""Regression tests for registered soft-IoU association costs."""

from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association.pyrecest_global_assignment import (
    _cost_kwargs_for_method,
    registered_soft_iou_cost_kwargs,
)
from bayescatrack.core.bridge import CalciumPlaneData


class _BrokenIndex:
    def __index__(self) -> int:
        raise ValueError("broken index")


def _single_roi_plane(mask: np.ndarray) -> CalciumPlaneData:
    roi_masks = np.asarray(mask, dtype=float).reshape(1, 1, -1)
    return CalciumPlaneData(roi_masks=roi_masks)


def test_registered_soft_iou_cost_preset_is_supported() -> None:
    kwargs = _cost_kwargs_for_method("registered-soft-iou")

    assert kwargs["soft_iou_weight"] == pytest.approx(1.0)
    assert kwargs["soft_iou_radius"] > 0
    assert kwargs["iou_weight"] == pytest.approx(0.0)
    assert kwargs["centroid_weight"] == pytest.approx(0.0)
    assert kwargs["mask_cosine_weight"] == pytest.approx(0.0)


def test_registered_soft_iou_kwargs_do_not_mutate_registered_iou_kwargs() -> None:
    soft_kwargs = registered_soft_iou_cost_kwargs()
    registered_kwargs = _cost_kwargs_for_method("registered-iou")

    assert soft_kwargs["soft_iou_weight"] == pytest.approx(1.0)
    assert "soft_iou_weight" not in registered_kwargs


def test_soft_iou_uses_mask_weights_not_only_binary_support() -> None:
    reference = _single_roi_plane(np.array([1.0, 1.0, 0.0]))
    measurement = _single_roi_plane(np.array([1.0, 0.25, 0.0]))
    common_kwargs = {
        "centroid_weight": 0.0,
        "iou_weight": 1.0,
        "mask_cosine_weight": 0.0,
        "area_weight": 0.0,
        "roi_feature_weight": 0.0,
        "cell_probability_weight": 0.0,
        "similarity_epsilon": 1.0e-9,
    }

    binary_cost = reference.build_pairwise_cost_matrix(measurement, **common_kwargs)
    soft_cost = reference.build_pairwise_cost_matrix(
        measurement, soft_iou=True, **common_kwargs
    )

    assert binary_cost[0, 0] == pytest.approx(0.0)
    assert soft_cost[0, 0] == pytest.approx(-np.log(1.25 / 2.0))
    assert soft_cost[0, 0] > binary_cost[0, 0]


def test_soft_iou_matches_binary_iou_for_boolean_masks() -> None:
    reference = _single_roi_plane(np.array([1.0, 1.0, 0.0]))
    measurement = _single_roi_plane(np.array([1.0, 0.0, 1.0]))
    common_kwargs = {
        "centroid_weight": 0.0,
        "iou_weight": 1.0,
        "mask_cosine_weight": 0.0,
        "area_weight": 0.0,
        "roi_feature_weight": 0.0,
        "cell_probability_weight": 0.0,
    }

    binary_cost = reference.build_pairwise_cost_matrix(measurement, **common_kwargs)
    soft_cost = reference.build_pairwise_cost_matrix(
        measurement, soft_iou=True, **common_kwargs
    )

    assert soft_cost[0, 0] == pytest.approx(binary_cost[0, 0])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"soft_iou_radius": True}, "soft_iou_radius must be an integer"),
        ({"soft_iou_radius": 1.5}, "soft_iou_radius must be an integer"),
        ({"soft_iou_radius": -1}, "soft_iou_radius must be non-negative"),
        (
            {"soft_iou_radius": _BrokenIndex()},
            "soft_iou_radius must be an integer",
        ),
        (
            {"distance_transform_overlap_radius": False},
            "distance_transform_overlap_radius must be an integer",
        ),
        (
            {"distance_transform_overlap_radius": "1.5"},
            "distance_transform_overlap_radius must be an integer",
        ),
        (
            {"distance_transform_overlap_radius": -1},
            "distance_transform_overlap_radius must be non-negative",
        ),
        (
            {"distance_transform_overlap_radius": _BrokenIndex()},
            "distance_transform_overlap_radius must be an integer",
        ),
        (
            {"distance_transform_overlap_weight": np.nan},
            "distance_transform_overlap_weight must be a finite non-negative value",
        ),
        (
            {"distance_transform_overlap_weight": True},
            "distance_transform_overlap_weight must be a finite non-negative value",
        ),
        (
            {"distance_transform_overlap_scale": np.nan},
            "distance_transform_overlap_scale must be a finite positive value",
        ),
        (
            {"distance_transform_overlap_scale": False},
            "distance_transform_overlap_scale must be a finite positive value",
        ),
        (
            {"similarity_epsilon": np.inf},
            "similarity_epsilon must be a finite positive value",
        ),
        (
            {"similarity_epsilon": True},
            "similarity_epsilon must be a finite positive value",
        ),
    ],
)
def test_registered_soft_iou_kwargs_reject_invalid_numeric_values(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        registered_soft_iou_cost_kwargs(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"soft_iou_weight": np.nan},
            "soft_iou_weight must be a finite non-negative value",
        ),
        (
            {"soft_iou_weight": True},
            "soft_iou_weight must be a finite non-negative value",
        ),
        ({"soft_iou_radius": True}, "soft_iou_radius must be an integer"),
        ({"soft_iou_radius": 1.5}, "soft_iou_radius must be an integer"),
        ({"soft_iou_radius": -1}, "soft_iou_radius must be non-negative"),
        ({"soft_iou_radius": _BrokenIndex()}, "soft_iou_radius must be an integer"),
        (
            {"distance_transform_overlap_weight": np.nan},
            "distance_transform_overlap_weight must be a finite non-negative value",
        ),
        (
            {"distance_transform_overlap_radius": "1.5"},
            "distance_transform_overlap_radius must be an integer",
        ),
        (
            {"distance_transform_overlap_radius": _BrokenIndex()},
            "distance_transform_overlap_radius must be an integer",
        ),
        (
            {"distance_transform_overlap_scale": np.inf},
            "distance_transform_overlap_scale must be a finite positive value",
        ),
        (
            {"similarity_epsilon": 0.0},
            "similarity_epsilon must be a finite positive value",
        ),
        ({"large_cost": np.nan}, "large_cost must be a finite positive value"),
    ],
)
def test_soft_overlap_runtime_rejects_invalid_numeric_values(
    kwargs: dict[str, object], message: str
) -> None:
    reference = _single_roi_plane(np.array([1.0, 0.0, 0.0]))
    measurement = _single_roi_plane(np.array([1.0, 0.0, 0.0]))
    common_kwargs = {
        "centroid_weight": 0.0,
        "iou_weight": 0.0,
        "mask_cosine_weight": 0.0,
        "area_weight": 0.0,
        "roi_feature_weight": 0.0,
        "cell_probability_weight": 0.0,
        "soft_iou_weight": 1.0,
        "soft_iou_radius": 1,
        "distance_transform_overlap_weight": 0.5,
        "distance_transform_overlap_radius": 1,
    }
    common_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        reference.build_pairwise_cost_matrix(measurement, **common_kwargs)
