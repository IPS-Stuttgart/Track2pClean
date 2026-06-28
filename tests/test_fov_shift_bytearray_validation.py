from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from bayescatrack.fov_registration import (
    apply_integer_image_translation,
    apply_integer_roi_mask_translation,
    apply_subpixel_image_translation,
    apply_subpixel_roi_mask_translation,
)


@pytest.mark.parametrize(
    ("translation_function", "source"),
    [
        (apply_integer_image_translation, np.zeros((4, 4), dtype=float)),
        (apply_integer_roi_mask_translation, np.zeros((1, 4, 4), dtype=bool)),
    ],
)
def test_integer_fov_translation_rejects_bytearray_shift(
    translation_function: Callable[..., np.ndarray],
    source: np.ndarray,
):
    with pytest.raises(ValueError, match="shift_yx"):
        translation_function(source, bytearray(b"12"))


@pytest.mark.parametrize(
    ("translation_function", "source"),
    [
        (apply_subpixel_image_translation, np.zeros((4, 4), dtype=float)),
        (apply_subpixel_roi_mask_translation, np.zeros((1, 4, 4), dtype=bool)),
    ],
)
def test_subpixel_fov_translation_rejects_bytearray_shift(
    translation_function: Callable[..., np.ndarray],
    source: np.ndarray,
):
    with pytest.raises(ValueError, match="shift_yx"):
        translation_function(source, bytearray(b"12"))


def test_subpixel_fov_translation_rejects_bytearray_shift_component():
    with pytest.raises(ValueError, match="shift_yx"):
        apply_subpixel_image_translation(
            np.zeros((4, 4), dtype=float),
            [bytearray(b"1"), 0.0],
        )
