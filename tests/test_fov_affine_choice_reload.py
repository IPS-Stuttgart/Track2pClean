from __future__ import annotations

import importlib

import bayescatrack
from bayescatrack import fov_affine_registration

_IMAGE_CHOICE_MARKER = "_bayescatrack_fov_affine_image_choice_validation_patch"
_MASK_CHOICE_MARKER = "_bayescatrack_fov_affine_mask_choice_validation_patch"


def _count_marker_in_wrapper_chain(function, marker: str) -> int:
    count = 0
    seen: set[int] = set()
    current = function
    while current is not None and id(current) not in seen:
        if getattr(current, marker, False):
            count += 1
        seen.add(id(current))
        current = getattr(current, "_bayescatrack_original", None)
    return count


def test_fov_affine_choice_validation_remains_reload_idempotent():
    for _ in range(2):
        importlib.reload(bayescatrack)

    assert (
        _count_marker_in_wrapper_chain(
            fov_affine_registration.apply_affine_image_warp,
            _IMAGE_CHOICE_MARKER,
        )
        == 1
    )
    assert (
        _count_marker_in_wrapper_chain(
            fov_affine_registration.apply_affine_roi_mask_warp,
            _MASK_CHOICE_MARKER,
        )
        == 1
    )
