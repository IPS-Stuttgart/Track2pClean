from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.fov_affine_registration import estimate_fov_affine_transform


@pytest.mark.parametrize(
    "subtract_mean",
    [
        0,
        1,
        "false",
        "true",
        np.array(True),
        None,
    ],
)
def test_estimator_rejects_invalid_subtract_mean(subtract_mean):
    image = np.eye(8, dtype=float)

    with pytest.raises(ValueError, match="subtract_mean"):
        estimate_fov_affine_transform(image, image, subtract_mean=subtract_mean)


def test_estimator_accepts_numpy_bool_subtract_mean():
    image = np.eye(8, dtype=float)

    estimate = estimate_fov_affine_transform(
        image,
        image,
        subtract_mean=np.bool_(True),
    )

    assert estimate.matrix_xy.shape == (2, 3)


@pytest.mark.parametrize(
    "grid_shape",
    [
        (True, 3),
        (0, 3),
        (2.5, 3),
        (3,),
        "3,3",
    ],
)
def test_estimator_rejects_invalid_grid_shape(grid_shape):
    image = np.eye(8, dtype=float)

    with pytest.raises(ValueError, match="grid_shape"):
        estimate_fov_affine_transform(image, image, grid_shape=grid_shape)


@pytest.mark.parametrize("min_tile_size", [True, 0, -1, 2.5, "2"])
def test_estimator_rejects_invalid_min_tile_size(min_tile_size):
    image = np.eye(8, dtype=float)

    with pytest.raises(ValueError, match="min_tile_size"):
        estimate_fov_affine_transform(image, image, min_tile_size=min_tile_size)


@pytest.mark.parametrize(
    "max_shift_fraction",
    [
        True,
        -0.1,
        float("nan"),
        float("inf"),
        "0.5",
        [0.5],
    ],
)
def test_estimator_rejects_invalid_max_shift_fraction(max_shift_fraction):
    image = np.eye(8, dtype=float)

    with pytest.raises(ValueError, match="max_shift_fraction"):
        estimate_fov_affine_transform(
            image,
            image,
            max_shift_fraction=max_shift_fraction,
        )
