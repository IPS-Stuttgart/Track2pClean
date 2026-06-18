"""Tests for the FOV-affine Track2p benchmark wrapper."""

# pylint: disable=protected-access

import numpy as np
import numpy.testing as npt
import pytest
from bayescatrack import CalciumPlaneData
from bayescatrack.association.pyrecest_global_assignment import (
    registered_iou_cost_kwargs,
)
from bayescatrack.core import _bridge_impl
from bayescatrack.experiments.track2p_fov_affine_benchmark import (
    _dilate_mask_stack,
    _pairwise_dilated_iou_matrix,
    _soft_iou_pairwise_cost_matrix,
)


def test_pairwise_dilated_iou_matches_reference_with_overlapping_pixels():
    reference = np.zeros((2, 8, 8), dtype=bool)
    measurement = np.zeros((2, 8, 8), dtype=bool)
    reference[0, 3, 3] = True
    reference[1, 3, 4] = True
    measurement[0, 3, 5] = True
    measurement[1, 6, 6] = True

    radius = 2
    actual = _pairwise_dilated_iou_matrix(reference, measurement, radius=radius)
    expected = _bridge_impl._pairwise_iou_matrix(
        _dilate_mask_stack(reference, radius=radius),
        _dilate_mask_stack(measurement, radius=radius),
    )

    npt.assert_allclose(actual, expected)


def test_radius_zero_iou_only_cost_matches_reference_implementation():
    reference = np.zeros((2, 8, 8), dtype=bool)
    measurement = np.zeros((2, 8, 8), dtype=bool)
    reference[0, 2:4, 2:4] = True
    reference[1, 4:6, 4:6] = True
    measurement[0, 2:4, 3:5] = True
    measurement[1, 0:2, 0:2] = True
    reference_plane = CalciumPlaneData(reference)
    measurement_plane = CalciumPlaneData(measurement)
    kwargs = registered_iou_cost_kwargs()

    actual = _soft_iou_pairwise_cost_matrix(
        CalciumPlaneData.build_pairwise_cost_matrix,
        reference_plane,
        measurement_plane,
        **kwargs,
    )
    expected = CalciumPlaneData.build_pairwise_cost_matrix(
        reference_plane,
        measurement_plane,
        **kwargs,
    )

    npt.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"soft_iou_radius": True}, "soft_iou_radius"),
        ({"soft_iou_radius": 1.5}, "soft_iou_radius"),
        ({"soft_iou_radius": -1}, "soft_iou_radius"),
        ({"return_components": 1}, "return_components"),
        ({"similarity_epsilon": 0.0}, "similarity_epsilon"),
        ({"similarity_epsilon": np.nan}, "similarity_epsilon"),
        ({"large_cost": np.inf}, "large_cost"),
        ({"iou_weight": True}, "iou_weight"),
        ({"iou_weight": -0.1}, "iou_weight"),
        ({"centroid_weight": np.nan, "soft_iou_radius": 0}, "centroid_weight"),
    ],
)
def test_soft_iou_benchmark_cost_rejects_invalid_runtime_controls(
    kwargs: dict[str, object], message: str
) -> None:
    reference = np.zeros((1, 8, 8), dtype=bool)
    measurement = np.zeros((1, 8, 8), dtype=bool)
    reference[0, 2:4, 2:4] = True
    measurement[0, 2:4, 3:5] = True
    reference_plane = CalciumPlaneData(reference)
    measurement_plane = CalciumPlaneData(measurement)

    common_kwargs = registered_iou_cost_kwargs()
    common_kwargs["soft_iou_radius"] = 1
    common_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        _soft_iou_pairwise_cost_matrix(
            CalciumPlaneData.build_pairwise_cost_matrix,
            reference_plane,
            measurement_plane,
            **common_kwargs,
        )


@pytest.mark.parametrize("radius", [True, 1.5, np.nan])
def test_benchmark_soft_iou_dilation_rejects_non_integer_radius(
    radius: object,
) -> None:
    masks = np.zeros((1, 5, 5), dtype=bool)
    masks[0, 2, 2] = True

    with pytest.raises(ValueError, match="soft_iou_radius"):
        _dilate_mask_stack(masks, radius=radius)
