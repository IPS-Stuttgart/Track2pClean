"""Strict validation for nonrigid-registration FOV image inputs.

The dense nonrigid registration path used to pass FOV images through
``np.nan_to_num`` before fitting the warp.  That silently turned NaN/Inf pixels
into zeros and could bias the estimated registration instead of surfacing
corrupt image data.  This package-level hook preserves the existing dimensional
validation while rejecting non-finite FOV values before the estimator runs.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_nonrigid_fov_image_validation_patch"
_ERROR_MESSAGE = "FOV images must contain only finite values"


def install_nonrigid_fov_image_validation() -> None:
    """Install idempotent validation around nonrigid FOV image normalization."""

    from . import (
        nonrigid_registration as _nonrigid_registration,  # pylint: disable=import-outside-toplevel
    )

    original = _nonrigid_registration._finite_image  # pylint: disable=protected-access
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def finite_image_with_nonfinite_validation(image: Any) -> np.ndarray:
        image_array = np.asarray(image, dtype=float)
        if image_array.ndim != 2:
            return original(image)
        if not np.all(np.isfinite(image_array)):
            raise ValueError(_ERROR_MESSAGE)
        return original(image_array)

    setattr(finite_image_with_nonfinite_validation, _PATCH_MARKER, True)
    setattr(finite_image_with_nonfinite_validation, "_bayescatrack_original", original)
    _nonrigid_registration._finite_image = (
        finite_image_with_nonfinite_validation  # pylint: disable=protected-access
    )


__all__ = ["install_nonrigid_fov_image_validation"]
