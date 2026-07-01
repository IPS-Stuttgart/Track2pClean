from __future__ import annotations

import importlib
from typing import Any

import bayescatrack
import bayescatrack.fov_registration as fov_registration
import track2pclean

_BYTES_SHAPE_MARKER = "_bayescatrack_fov_translation_bytes_shape_validation_patch"
_SUBPIXEL_SHIFT_MARKER = "_bayescatrack_fov_subpixel_shift_validation_patch"


def _marker_count(function: Any, marker: str) -> int:
    seen: set[int] = set()
    current = function
    count = 0
    while current is not None:
        current_id = id(current)
        if current_id in seen:
            raise AssertionError("wrapper chain contains a cycle")
        seen.add(current_id)
        if getattr(current, marker, False):
            count += 1
        current = getattr(current, "_bayescatrack_original", None)
    return count


def test_fov_translation_compatibility_wrappers_are_reload_idempotent() -> None:
    importlib.reload(track2pclean)
    importlib.reload(bayescatrack)
    importlib.reload(track2pclean)

    fov_translation_functions = (
        fov_registration.apply_integer_image_translation,
        fov_registration.apply_subpixel_image_translation,
        fov_registration.apply_integer_roi_mask_translation,
        fov_registration.apply_subpixel_roi_mask_translation,
    )
    for function in fov_translation_functions:
        assert _marker_count(function, _BYTES_SHAPE_MARKER) == 1

    subpixel_translation_functions = (
        fov_registration.apply_subpixel_image_translation,
        fov_registration.apply_subpixel_roi_mask_translation,
    )
    for function in subpixel_translation_functions:
        assert _marker_count(function, _SUBPIXEL_SHIFT_MARKER) == 1
