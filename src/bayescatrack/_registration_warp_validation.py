"""Strict validation for registration image and ROI-mask warp controls.

Registration mask warping accepts a ``binarize`` flag and threshold that are often
threaded through experiment configs.  Python truthiness would otherwise turn
malformed values such as ``"false"`` or ``1`` into an enabled binarization gate,
and boolean thresholds would be reinterpreted as numeric ``0.0``/``1.0``.

The registration warps also thread ``output_shape`` into NumPy allocation and
index-grid helpers.  Values such as ``True`` or fractional floats can otherwise be
silently reinterpreted, changing the registered image/mask geometry rather than
failing fast at the call boundary.  The hook below preserves valid integer-like
shape components while rejecting ambiguous runtime controls before registered
images or masks are changed.
"""

from __future__ import annotations

import operator
from functools import wraps
from typing import Any

import numpy as np

_IMAGE_PATCH_MARKER = "_bayescatrack_registration_image_warp_validation_patch"
_ROI_MASK_PATCH_MARKER = "_bayescatrack_registration_roi_mask_warp_validation_patch"
_OUTPUT_SHAPE_ERROR = (
    "output_shape must contain exactly two non-negative integer values"
)
_STRING_LIKE_SCALAR_TYPES = (str, bytes, bytearray, np.str_, np.bytes_)


def install_registration_warp_validation() -> None:
    """Install idempotent validation around registration image and ROI-mask warping."""

    from . import (
        registration as _registration,  # pylint: disable=import-outside-toplevel
    )

    original_image_warp = _registration.warp_image_into_reference_frame
    if not getattr(original_image_warp, _IMAGE_PATCH_MARKER, False):

        @wraps(original_image_warp)
        def warp_image_into_reference_frame_with_validation(
            image: Any,
            reference_to_measurement_matrix: Any,
            reference_to_measurement_offset: Any,
            *args: Any,
            **kwargs: Any,
        ) -> np.ndarray:
            return original_image_warp(
                image,
                reference_to_measurement_matrix,
                reference_to_measurement_offset,
                *args,
                **_normalize_common_warp_kwargs(kwargs),
            )

        _mark_patch(
            warp_image_into_reference_frame_with_validation,
            original_image_warp,
            _IMAGE_PATCH_MARKER,
        )
        _registration.warp_image_into_reference_frame = (  # type: ignore[assignment]
            warp_image_into_reference_frame_with_validation
        )

    original_roi_mask_warp = _registration.warp_roi_masks_into_reference_frame
    if not getattr(original_roi_mask_warp, _ROI_MASK_PATCH_MARKER, False):

        @wraps(original_roi_mask_warp)
        def warp_roi_masks_into_reference_frame_with_validation(
            roi_masks: Any,
            reference_to_measurement_matrix: Any,
            reference_to_measurement_offset: Any,
            *args: Any,
            **kwargs: Any,
        ) -> np.ndarray:
            normalized_kwargs = _normalize_registration_mask_warp_kwargs(kwargs)
            return original_roi_mask_warp(
                roi_masks,
                reference_to_measurement_matrix,
                reference_to_measurement_offset,
                *args,
                **normalized_kwargs,
            )

        _mark_patch(
            warp_roi_masks_into_reference_frame_with_validation,
            original_roi_mask_warp,
            _ROI_MASK_PATCH_MARKER,
        )
        _registration.warp_roi_masks_into_reference_frame = (  # type: ignore[assignment]
            warp_roi_masks_into_reference_frame_with_validation
        )


def _mark_patch(wrapper: Any, original: Any, marker: str) -> None:
    setattr(wrapper, marker, True)
    setattr(wrapper, "_bayescatrack_original", original)


def _normalize_common_warp_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    normalized_kwargs = dict(kwargs)
    if "output_shape" in normalized_kwargs:
        normalized_kwargs["output_shape"] = _normalize_output_shape(
            normalized_kwargs["output_shape"]
        )
    return normalized_kwargs


def _normalize_registration_mask_warp_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    normalized_kwargs = _normalize_common_warp_kwargs(kwargs)
    if "binarize" in normalized_kwargs:
        normalized_kwargs["binarize"] = _strict_bool(
            normalized_kwargs["binarize"],
            name="binarize",
        )
    if "threshold" in normalized_kwargs:
        normalized_kwargs["threshold"] = _finite_unit_interval_float(
            normalized_kwargs["threshold"],
            name="threshold",
        )
    return normalized_kwargs


def _normalize_output_shape(output_shape: Any) -> tuple[int, int]:
    if isinstance(output_shape, (str, bytes)):
        raise ValueError(_OUTPUT_SHAPE_ERROR)
    try:
        shape_array = np.asarray(output_shape, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(_OUTPUT_SHAPE_ERROR) from exc

    flat_shape = shape_array.reshape(-1)
    if flat_shape.size != 2:
        raise ValueError(_OUTPUT_SHAPE_ERROR)

    height, width = (
        _normalize_output_shape_component(value) for value in flat_shape.tolist()
    )
    return height, width


def _normalize_output_shape_component(value: Any) -> int:
    if isinstance(value, (bool, np.bool_, str, bytes)):
        raise ValueError(_OUTPUT_SHAPE_ERROR)

    try:
        integer_value = int(operator.index(value))
    except TypeError:
        integer_value = None

    if integer_value is None:
        if isinstance(value, (float, np.floating)):
            numeric_value = float(value)
            if not np.isfinite(numeric_value) or not numeric_value.is_integer():
                raise ValueError(_OUTPUT_SHAPE_ERROR)
            integer_value = int(numeric_value)
        else:
            raise ValueError(_OUTPUT_SHAPE_ERROR)

    if integer_value < 0:
        raise ValueError(_OUTPUT_SHAPE_ERROR)
    return int(integer_value)


def _strict_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def _finite_unit_interval_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_, *_STRING_LIKE_SCALAR_TYPES)):
        raise ValueError(f"{name} must be a finite scalar value in [0, 1]")
    try:
        value_array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar value in [0, 1]") from exc
    if value_array.shape != ():
        raise ValueError(f"{name} must be a finite scalar value in [0, 1]")

    scalar_value = value_array.item()
    if isinstance(scalar_value, (bool, np.bool_, *_STRING_LIKE_SCALAR_TYPES)):
        raise ValueError(f"{name} must be a finite scalar value in [0, 1]")

    try:
        numeric_value = float(scalar_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar value in [0, 1]") from exc
    if not np.isfinite(numeric_value) or numeric_value < 0.0 or numeric_value > 1.0:
        raise ValueError(f"{name} must be a finite scalar value in [0, 1]")
    return numeric_value


__all__ = ["install_registration_warp_validation"]
