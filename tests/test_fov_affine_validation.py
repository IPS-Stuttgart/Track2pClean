from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.fov_affine_registration import (
    apply_affine_image_warp,
    apply_affine_roi_mask_warp,
    estimate_fov_affine_transform,
)

_AFFINE_IDENTITY_XY = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=float,
)


def test_apply_affine_image_warp_rejects_nonfinite_matrix_before_all_fill_output():
    image = np.ones((4, 4), dtype=float)
    bad_matrix = _AFFINE_IDENTITY_XY.copy()
    bad_matrix[0, 2] = np.nan

    with pytest.raises(ValueError, match="finite 2-by-3 affine matrix"):
        apply_affine_image_warp(image, bad_matrix, output_shape=(4, 4))


def test_apply_affine_roi_mask_warp_rejects_nonfinite_bilinear_matrix():
    masks = np.zeros((1, 4, 4), dtype=bool)
    masks[0, 1, 1] = True
    bad_matrix = _AFFINE_IDENTITY_XY.copy()
    bad_matrix[1, 2] = np.inf

    with pytest.raises(ValueError, match="finite 2-by-3 affine matrix"):
        apply_affine_roi_mask_warp(
            masks,
            bad_matrix,
            output_shape=(4, 4),
            mode="bilinear",
        )


@pytest.mark.parametrize(
    "bad_matrix_entry",
    [
        b"1",
        bytearray(b"1"),
        memoryview(b"1"),
    ],
)
def test_apply_affine_image_warp_rejects_bytes_like_matrix_entries(
    bad_matrix_entry,
):
    image = np.ones((4, 4), dtype=float)
    bad_matrix = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, bad_matrix_entry]],
        dtype=object,
    )

    with pytest.raises(ValueError, match="finite 2-by-3 affine matrix"):
        apply_affine_image_warp(image, bad_matrix, output_shape=(4, 4))


@pytest.mark.parametrize(
    "bad_output_shape",
    [
        (4.5, 4),
        (True, 4),
        (-1, 4),
        (0, 4),
        (4, 0),
        (4,),
        bytearray([4, 4]),
        memoryview(bytes([4, 4])),
    ],
)
def test_apply_affine_image_warp_rejects_malformed_output_shape(bad_output_shape):
    image = np.ones((4, 4), dtype=float)

    with pytest.raises(ValueError, match="output_shape"):
        apply_affine_image_warp(
            image,
            _AFFINE_IDENTITY_XY,
            output_shape=bad_output_shape,
        )


@pytest.mark.parametrize(
    "bad_output_shape",
    [
        (4.5, 4),
        (np.bool_(True), 4),
        (-1, 4),
        (0, 4),
        (4, 0),
        (4,),
        bytearray([4, 4]),
        memoryview(bytes([4, 4])),
    ],
)
def test_apply_affine_roi_mask_warp_rejects_malformed_output_shape(bad_output_shape):
    masks = np.zeros((1, 4, 4), dtype=bool)

    with pytest.raises(ValueError, match="output_shape"):
        apply_affine_roi_mask_warp(
            masks,
            _AFFINE_IDENTITY_XY,
            output_shape=bad_output_shape,
        )


def test_estimate_fov_affine_transform_returns_identity_for_constant_fovs():
    image = np.ones((8, 8), dtype=float)

    estimate = estimate_fov_affine_transform(image, image, min_tile_size=2)

    np.testing.assert_allclose(estimate.matrix_xy, _AFFINE_IDENTITY_XY)
    np.testing.assert_allclose(estimate.inverse_matrix_xy, _AFFINE_IDENTITY_XY)
    np.testing.assert_allclose(
        estimate.tile_shift_yx,
        np.zeros((1, 2), dtype=float),
    )
    np.testing.assert_allclose(
        estimate.tile_peak_correlation,
        np.zeros((1,), dtype=float),
    )
    assert estimate.tile_reference_xy.shape == (0, 2)
    assert estimate.tile_measurement_xy.shape == (0, 2)
    assert estimate.fit_rmse == 0.0
    assert estimate.fallback_translation


@pytest.mark.parametrize("bad_subtract_mean", ["false", 1, 0, None])
def test_estimate_fov_affine_transform_rejects_malformed_subtract_mean(
    bad_subtract_mean,
):
    image = np.eye(8, dtype=float)

    with pytest.raises(ValueError, match="subtract_mean"):
        estimate_fov_affine_transform(
            image,
            image,
            subtract_mean=bad_subtract_mean,
        )


@pytest.mark.parametrize(
    "bad_grid_shape",
    [
        (True, 3),
        (0, 3),
        (2.5, 3),
        (3,),
        "3,3",
        bytearray([3, 3]),
        memoryview(bytes([3, 3])),
    ],
)
def test_estimate_fov_affine_transform_rejects_malformed_grid_shape(bad_grid_shape):
    image = np.eye(8, dtype=float)

    with pytest.raises(ValueError, match="grid_shape"):
        estimate_fov_affine_transform(
            image,
            image,
            grid_shape=bad_grid_shape,
        )


@pytest.mark.parametrize("bad_min_tile_size", [True, 0, -1, 2.5, "2"])
def test_estimate_fov_affine_transform_rejects_malformed_min_tile_size(
    bad_min_tile_size,
):
    image = np.eye(8, dtype=float)

    with pytest.raises(ValueError, match="min_tile_size"):
        estimate_fov_affine_transform(
            image,
            image,
            min_tile_size=bad_min_tile_size,
        )


@pytest.mark.parametrize(
    "bad_max_shift_fraction",
    [
        True,
        -0.1,
        np.nan,
        np.inf,
        "0.5",
        [0.5],
    ],
)
def test_estimate_fov_affine_transform_rejects_malformed_max_shift_fraction(
    bad_max_shift_fraction,
):
    image = np.eye(8, dtype=float)

    with pytest.raises(ValueError, match="max_shift_fraction"):
        estimate_fov_affine_transform(
            image,
            image,
            max_shift_fraction=bad_max_shift_fraction,
        )
