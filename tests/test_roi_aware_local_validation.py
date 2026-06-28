from __future__ import annotations

import importlib

import numpy as np
import pytest
from bayescatrack.association import pyrecest_global_assignment as global_assignment


@pytest.mark.parametrize(
    "weight_name",
    [
        "weighted_dice_weight",
        "overlap_fraction_weight",
        "distance_transform_weight",
        "image_patch_weight",
        "neighbor_constellation_weight",
        "centroid_rank_weight",
    ],
)
@pytest.mark.parametrize(
    "bad_value",
    [
        True,
        False,
        np.asarray(True),
        np.asarray([1.0]),
        np.nan,
        np.inf,
        -1.0,
    ],
)
def test_roi_aware_local_cost_kwargs_reject_invalid_weights(weight_name, bad_value):
    with pytest.raises(
        ValueError,
        match=f"{weight_name} must be a finite non-negative value",
    ):
        global_assignment.roi_aware_local_cost_kwargs(**{weight_name: bad_value})


@pytest.mark.parametrize(
    "bad_value",
    [True, False, 1.5, np.asarray([1]), np.nan, np.inf, -1],
)
def test_roi_aware_local_cost_kwargs_reject_invalid_patch_radius(bad_value):
    with pytest.raises(ValueError, match="patch_radius"):
        global_assignment.roi_aware_local_cost_kwargs(patch_radius=bad_value)


@pytest.mark.parametrize(
    "bad_value",
    [True, False, 1.5, np.asarray([1]), np.nan, np.inf, 0],
)
def test_roi_aware_local_cost_kwargs_reject_invalid_neighbor_k(bad_value):
    with pytest.raises(ValueError, match="neighbor_k"):
        global_assignment.roi_aware_local_cost_kwargs(neighbor_k=bad_value)


@pytest.mark.parametrize(
    "integer_like",
    [4, np.int64(4), np.asarray(4), 4.0, np.asarray(4.0), "4"],
)
def test_roi_aware_local_cost_kwargs_accept_integer_like_radius_controls(integer_like):
    kwargs = global_assignment.roi_aware_local_cost_kwargs(
        patch_radius=integer_like,
        neighbor_k=integer_like,
    )

    assert kwargs["patch_radius"] == 4
    assert kwargs["neighbor_k"] == 4


def test_roi_aware_local_validation_survives_association_reload():
    import bayescatrack.association as association

    importlib.reload(association)
    kwargs = global_assignment.roi_aware_local_cost_kwargs(
        patch_radius=np.int64(3),
        neighbor_k=np.int64(5),
    )

    assert kwargs["patch_radius"] == 3
    assert kwargs["neighbor_k"] == 5
