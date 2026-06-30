from __future__ import annotations

import importlib
from typing import Any

import bayescatrack
import bayescatrack.fov_affine_registration as fov_affine_registration


def _wrapper_count(function: Any, marker: str) -> int:
    count = 0
    seen: set[int] = set()
    current = function
    while current is not None:
        current_id = id(current)
        if current_id in seen:
            raise AssertionError("cycle in FOV-affine validation wrapper chain")
        seen.add(current_id)
        if getattr(current, marker, False):
            count += 1
        current = getattr(current, "_bayescatrack_original", None)
    return count


def test_fov_affine_validation_patch_installers_are_reload_idempotent() -> None:
    importlib.reload(bayescatrack)
    importlib.reload(bayescatrack)

    assert _wrapper_count(
        fov_affine_registration.apply_affine_image_warp,
        "_bayescatrack_fov_affine_image_warp_validation_patch",
    ) == 1
    assert _wrapper_count(
        fov_affine_registration.apply_affine_image_warp,
        "_bayescatrack_fov_affine_image_choice_validation_patch",
    ) == 1
    assert _wrapper_count(
        fov_affine_registration.apply_affine_roi_mask_warp,
        "_bayescatrack_fov_affine_roi_mask_warp_validation_patch",
    ) == 1
    assert _wrapper_count(
        fov_affine_registration.apply_affine_roi_mask_warp,
        "_bayescatrack_fov_affine_mask_choice_validation_patch",
    ) == 1
    assert _wrapper_count(
        fov_affine_registration.estimate_fov_affine_transform,
        "_bayescatrack_fov_affine_estimator_validation",
    ) == 1
    assert _wrapper_count(
        fov_affine_registration.estimate_fov_affine_transform,
        "_bayescatrack_fov_affine_estimate_control_validation_patch",
    ) == 1
