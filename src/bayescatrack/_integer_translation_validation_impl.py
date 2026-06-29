"""Implementation for strict integer FOV translation shift validation."""

from __future__ import annotations

import operator
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_integer_translation_validation_patch"
_ROI_PATCH_MARKER = "_bayescatrack_integer_roi_translation_validation_patch"
_SHIFT_ERROR = "shift_yx must contain exactly two integer values"


def install_integer_image_translation_validation() -> None:
    """Install idempotent validation around integer image/ROI translations."""

    from . import (
        fov_registration as _fov_registration,  # pylint: disable=import-outside-toplevel
    )

    image_original = _fov_registration.apply_integer_image_translation
    if not getattr(image_original, _PATCH_MARKER, False):

        @wraps(image_original)
        def apply_integer_image_translation_with_shift_validation(
            image: Any,
            shift_yx: Any,
            *args: Any,
            **kwargs: Any,
        ) -> np.ndarray:
            return image_original(
                image,
                _normalize_integer_shift_yx(shift_yx),
                *args,
                **kwargs,
            )

        _mark_patch(
            apply_integer_image_translation_with_shift_validation,
            image_original,
            _PATCH_MARKER,
        )
        _fov_registration.apply_integer_image_translation = (
            apply_integer_image_translation_with_shift_validation
        )

    roi_original = _fov_registration.apply_integer_roi_mask_translation
    if not getattr(roi_original, _ROI_PATCH_MARKER, False):

        @wraps(roi_original)
        def apply_integer_roi_mask_translation_with_shift_validation(
            roi_masks: Any,
            shift_yx: Any,
            *args: Any,
            **kwargs: Any,
        ) -> np.ndarray:
            return roi_original(
                roi_masks,
                _normalize_integer_shift_yx(shift_yx),
                *args,
                **kwargs,
            )

        _mark_patch(
            apply_integer_roi_mask_translation_with_shift_validation,
            roi_original,
            _ROI_PATCH_MARKER,
        )
        _fov_registration.apply_integer_roi_mask_translation = (
            apply_integer_roi_mask_translation_with_shift_validation
        )


def _mark_patch(wrapper: Any, original: Any, marker: str) -> None:
    setattr(wrapper, marker, True)
    setattr(wrapper, "_bayescatrack_original", original)


def _normalize_integer_shift_yx(shift_yx: Any) -> tuple[int, int]:
    if isinstance(shift_yx, (str, bytes, bytearray)):
        raise ValueError(_SHIFT_ERROR)
    try:
        shift_array = np.asarray(shift_yx, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(_SHIFT_ERROR) from exc

    if shift_array.shape != (2,):
        raise ValueError(_SHIFT_ERROR)

    shift_y, shift_x = (
        _normalize_integer_shift_component(value) for value in shift_array.tolist()
    )
    return shift_y, shift_x


def _normalize_integer_shift_component(value: Any) -> int:
    if isinstance(value, (bool, np.bool_, bytes, bytearray)):
        raise ValueError(_SHIFT_ERROR)

    try:
        return int(operator.index(value))
    except TypeError:
        pass
    except (ValueError, OverflowError) as exc:
        raise ValueError(_SHIFT_ERROR) from exc

    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        if np.isfinite(numeric_value) and numeric_value.is_integer():
            return int(numeric_value)
        raise ValueError(_SHIFT_ERROR)

    if isinstance(value, str):
        try:
            return int(value.strip(), 10)
        except ValueError as exc:
            raise ValueError(_SHIFT_ERROR) from exc

    raise ValueError(_SHIFT_ERROR)


__all__ = ["install_integer_image_translation_validation"]
