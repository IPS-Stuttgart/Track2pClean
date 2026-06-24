"""Strict validation for FOV integer-translation helper inputs."""

from __future__ import annotations

import operator
from functools import wraps
from typing import Any, Sequence

import numpy as np

_PATCH_MARKER = "_bayescatrack_fov_integer_translation_validation_patch"


def install_fov_translation_validation() -> None:
    """Install idempotent validation around integer FOV translation helpers."""

    from . import fov_registration as _fov_registration

    original_image_translation = _fov_registration.apply_integer_image_translation
    if not getattr(original_image_translation, _PATCH_MARKER, False):

        @wraps(original_image_translation)
        def apply_integer_image_translation_with_validation(
            image: Any,
            shift_yx: Sequence[int] | np.ndarray,
            *,
            output_shape: tuple[int, int] | None = None,
            fill_value: float | bool = 0.0,
        ) -> np.ndarray:
            return original_image_translation(
                image,
                _integer_shift_vector(shift_yx, name="shift_yx"),
                output_shape=output_shape,
                fill_value=fill_value,
            )

        setattr(apply_integer_image_translation_with_validation, _PATCH_MARKER, True)
        setattr(
            apply_integer_image_translation_with_validation,
            "_bayescatrack_original",
            original_image_translation,
        )
        _fov_registration.apply_integer_image_translation = (
            apply_integer_image_translation_with_validation
        )

    original_roi_translation = _fov_registration.apply_integer_roi_mask_translation
    if not getattr(original_roi_translation, _PATCH_MARKER, False):

        @wraps(original_roi_translation)
        def apply_integer_roi_mask_translation_with_validation(
            roi_masks: Any,
            shift_yx: Sequence[int] | np.ndarray,
            *,
            output_shape: tuple[int, int] | None = None,
        ) -> np.ndarray:
            return original_roi_translation(
                roi_masks,
                _integer_shift_vector(shift_yx, name="shift_yx"),
                output_shape=output_shape,
            )

        setattr(apply_integer_roi_mask_translation_with_validation, _PATCH_MARKER, True)
        setattr(
            apply_integer_roi_mask_translation_with_validation,
            "_bayescatrack_original",
            original_roi_translation,
        )
        _fov_registration.apply_integer_roi_mask_translation = (
            apply_integer_roi_mask_translation_with_validation
        )


def _integer_shift_vector(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=object)
    if array.shape != (2,):
        raise ValueError(f"{name} must contain exactly two integer values")
    return np.asarray(
        [_integer_shift_component(component, name=name) for component in array],
        dtype=int,
    )


def _integer_shift_component(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must contain integer values, not booleans")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if np.isfinite(numeric) and numeric.is_integer():
            return int(numeric)
        raise ValueError(f"{name} must contain finite integer values")
    if isinstance(value, str):
        try:
            return int(value.strip(), 10)
        except ValueError as exc:
            raise ValueError(f"{name} must contain integer values") from exc
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise ValueError(f"{name} must contain integer values") from exc


__all__ = ["install_fov_translation_validation"]
