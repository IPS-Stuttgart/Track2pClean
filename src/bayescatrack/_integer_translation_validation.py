"""Strict validation for integer FOV translation shifts.

The low-level integer translation helper previously coerced ``shift_yx`` through a
NumPy integer array before unpacking it.  Fractional values such as ``1.9`` were
therefore truncated to ``1`` instead of being rejected, which could silently apply
the wrong integer displacement to FOV images and ROI masks.  This package-level
hook keeps integer-like inputs working while failing fast for fractional,
non-finite, boolean, or malformed shifts.
"""

from __future__ import annotations

import operator
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_integer_translation_validation_patch"
_SHIFT_ERROR = "shift_yx must contain exactly two integer values"


def install_integer_image_translation_validation() -> None:
    """Install idempotent validation around integer image translations."""

    from . import fov_registration as _fov_registration  # pylint: disable=import-outside-toplevel

    original = _fov_registration.apply_integer_image_translation
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def apply_integer_image_translation_with_shift_validation(
        image: Any,
        shift_yx: Any,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        return original(
            image,
            _normalize_integer_shift_yx(shift_yx),
            *args,
            **kwargs,
        )

    _mark_patch(apply_integer_image_translation_with_shift_validation, original)
    _fov_registration.apply_integer_image_translation = (
        apply_integer_image_translation_with_shift_validation
    )


def _mark_patch(wrapper: Any, original: Any) -> None:
    setattr(wrapper, _PATCH_MARKER, True)
    setattr(wrapper, "_bayescatrack_original", original)


def _normalize_integer_shift_yx(shift_yx: Any) -> tuple[int, int]:
    try:
        shift_array = np.asarray(shift_yx, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(_SHIFT_ERROR) from exc

    flattened_shift = shift_array.reshape(-1)
    if flattened_shift.size != 2:
        raise ValueError(_SHIFT_ERROR)

    return tuple(
        _normalize_integer_shift_component(value)
        for value in flattened_shift.tolist()
    )  # type: ignore[return-value]


def _normalize_integer_shift_component(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(_SHIFT_ERROR)

    try:
        return int(operator.index(value))
    except TypeError:
        pass

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
