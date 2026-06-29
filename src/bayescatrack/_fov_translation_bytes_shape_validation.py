"""Reject bytes-like FOV-translation output shapes before NumPy expansion."""

from __future__ import annotations

from functools import wraps
from typing import Any

_PATCH_MARKER = "_bayescatrack_fov_translation_bytes_shape_validation_patch"
_BYTES_LIKE_SHAPE_TYPES = (bytearray, memoryview)
_OUTPUT_SHAPE_ERROR = "output_shape must contain exactly two positive integer dimensions"


def install_fov_translation_bytes_shape_validation() -> None:
    """Install idempotent bytes-like output-shape validation."""

    from . import fov_registration as _fov_registration  # pylint: disable=import-outside-toplevel

    for function_name in (
        "apply_integer_image_translation",
        "apply_subpixel_image_translation",
        "apply_integer_roi_mask_translation",
        "apply_subpixel_roi_mask_translation",
    ):
        _wrap_output_shape_kwarg(_fov_registration, function_name)


def _wrap_output_shape_kwarg(module: Any, function_name: str) -> None:
    original = getattr(module, function_name)
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        output_shape = kwargs.get("output_shape")
        if isinstance(output_shape, _BYTES_LIKE_SHAPE_TYPES):
            raise ValueError(_OUTPUT_SHAPE_ERROR)
        return original(*args, **kwargs)

    setattr(wrapper, _PATCH_MARKER, True)
    setattr(wrapper, "_bayescatrack_original", original)
    setattr(module, function_name, wrapper)


__all__ = ["install_fov_translation_bytes_shape_validation"]
