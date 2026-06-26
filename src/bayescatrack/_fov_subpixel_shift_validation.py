"""Strict validation for FOV subpixel translation shifts and alias controls.

The low-level subpixel translation helpers previously coerced ``shift_yx`` with
``np.asarray(..., dtype=float).reshape(2)``.  Boolean values therefore became
numeric one-pixel or zero-pixel shifts before image or ROI-mask resampling, and
malformed shapes surfaced as low-level reshape errors.  This package-level hook
keeps ordinary numeric shifts working while rejecting booleans, non-finite
values, and malformed shift vectors at the API boundary.

The public FOV registration entry point also exposes the legacy
``subpixel_refinement`` alias alongside the current ``subpixel`` flag.  The
higher-level option parser rejects contradictory values, so this hook keeps the
direct entry point from silently letting one flag override the other.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_fov_subpixel_shift_validation_patch"
_REGISTRATION_ALIAS_PATCH_MARKER = "_bayescatrack_fov_subpixel_alias_validation_patch"
_SHIFT_ERROR = "shift_yx must contain exactly two finite numeric values"
_UNSET = object()


def install_fov_subpixel_shift_validation() -> None:
    """Install idempotent validation around FOV subpixel translation shifts."""

    from . import fov_registration as _fov_registration  # pylint: disable=import-outside-toplevel

    _wrap_shift_argument(_fov_registration, "apply_subpixel_image_translation")
    _wrap_shift_argument(_fov_registration, "apply_subpixel_roi_mask_translation")
    _wrap_registration_subpixel_aliases(_fov_registration)


def _wrap_shift_argument(module: Any, function_name: str) -> None:
    original = getattr(module, function_name)
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def wrapper(source: Any, shift_yx: Any, *args: Any, **kwargs: Any) -> Any:
        return original(source, _normalize_subpixel_shift_yx(shift_yx), *args, **kwargs)

    setattr(wrapper, _PATCH_MARKER, True)
    setattr(wrapper, "_bayescatrack_original", original)
    setattr(module, function_name, wrapper)


def _wrap_registration_subpixel_aliases(module: Any) -> None:
    original = getattr(module, "register_measurement_plane_by_fov_translation")
    if getattr(original, _REGISTRATION_ALIAS_PATCH_MARKER, False):
        return

    @wraps(original)
    def wrapper(
        reference_plane: Any,
        measurement_plane: Any,
        *args: Any,
        subpixel: Any = _UNSET,
        subpixel_refinement: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        call_kwargs = dict(kwargs)
        subpixel_was_provided = subpixel is not _UNSET

        if subpixel_refinement is not None:
            subpixel_refinement_value = _strict_bool(
                subpixel_refinement,
                name="subpixel_refinement",
            )
            call_kwargs["subpixel_refinement"] = subpixel_refinement_value
            if subpixel_was_provided:
                subpixel_value = _strict_bool(subpixel, name="subpixel")
                if subpixel_value != subpixel_refinement_value:
                    raise ValueError("subpixel and subpixel_refinement disagree")
                subpixel = subpixel_value

        if subpixel_was_provided:
            call_kwargs["subpixel"] = subpixel

        return original(
            reference_plane,
            measurement_plane,
            *args,
            **call_kwargs,
        )

    setattr(wrapper, _REGISTRATION_ALIAS_PATCH_MARKER, True)
    setattr(wrapper, "_bayescatrack_original", original)
    setattr(module, "register_measurement_plane_by_fov_translation", wrapper)


def _normalize_subpixel_shift_yx(shift_yx: Any) -> np.ndarray:
    if isinstance(shift_yx, (str, bytes)):
        raise ValueError(_SHIFT_ERROR)
    try:
        shift_array = np.asarray(shift_yx, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(_SHIFT_ERROR) from exc

    flattened_shift = shift_array.reshape(-1)
    if flattened_shift.size != 2:
        raise ValueError(_SHIFT_ERROR)

    return np.asarray(
        [_normalize_subpixel_shift_component(value) for value in flattened_shift.tolist()],
        dtype=float,
    )


def _normalize_subpixel_shift_component(value: Any) -> float:
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ValueError(_SHIFT_ERROR)
        value = value.item()
    if isinstance(value, (bool, np.bool_, str, bytes)):
        raise ValueError(_SHIFT_ERROR)
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(_SHIFT_ERROR) from exc
    if not np.isfinite(numeric_value):
        raise ValueError(_SHIFT_ERROR)
    return numeric_value


def _strict_bool(value: Any, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


__all__ = ["install_fov_subpixel_shift_validation"]
