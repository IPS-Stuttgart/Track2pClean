"""Regression tests for FOV-affine index-protocol validation errors."""

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


class _IndexRaises:
    def __init__(self, exception_type: type[Exception]) -> None:
        self.exception_type = exception_type

    def __index__(self) -> int:
        raise self.exception_type("broken index conversion")


@pytest.mark.parametrize("exception_type", [ValueError, OverflowError, FloatingPointError])
def test_apply_affine_image_warp_normalizes_bad_output_shape_index_errors(
    exception_type: type[Exception],
) -> None:
    image = np.ones((4, 4), dtype=float)

    with pytest.raises(ValueError, match="output_shape must contain"):
        apply_affine_image_warp(
            image,
            _AFFINE_IDENTITY_XY,
            output_shape=(_IndexRaises(exception_type), 4),
        )


@pytest.mark.parametrize("exception_type", [ValueError, OverflowError, FloatingPointError])
def test_apply_affine_roi_mask_warp_normalizes_bad_output_shape_index_errors(
    exception_type: type[Exception],
) -> None:
    masks = np.ones((1, 4, 4), dtype=bool)

    with pytest.raises(ValueError, match="output_shape must contain"):
        apply_affine_roi_mask_warp(
            masks,
            _AFFINE_IDENTITY_XY,
            output_shape=(4, _IndexRaises(exception_type)),
        )


@pytest.mark.parametrize("exception_type", [ValueError, OverflowError, FloatingPointError])
def test_estimate_fov_affine_transform_normalizes_bad_grid_shape_index_errors(
    exception_type: type[Exception],
) -> None:
    image = np.eye(8, dtype=float)

    with pytest.raises(ValueError, match="grid_shape must contain"):
        estimate_fov_affine_transform(
            image,
            image,
            grid_shape=(_IndexRaises(exception_type), 3),
        )


@pytest.mark.parametrize("exception_type", [ValueError, OverflowError, FloatingPointError])
def test_estimate_fov_affine_transform_normalizes_bad_min_tile_size_index_errors(
    exception_type: type[Exception],
) -> None:
    image = np.eye(8, dtype=float)

    with pytest.raises(ValueError, match="min_tile_size must be a positive integer"):
        estimate_fov_affine_transform(
            image,
            image,
            min_tile_size=_IndexRaises(exception_type),
        )
