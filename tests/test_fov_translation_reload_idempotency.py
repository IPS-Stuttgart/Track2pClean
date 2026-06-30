from __future__ import annotations

from importlib import reload as reload_module

import bayescatrack
import bayescatrack.fov_registration as fov_registration


def test_fov_translation_validation_installers_are_reload_idempotent() -> None:
    reload_module(bayescatrack)
    installed_functions = (
        fov_registration.apply_integer_image_translation,
        fov_registration.apply_subpixel_image_translation,
        fov_registration.apply_integer_roi_mask_translation,
        fov_registration.apply_subpixel_roi_mask_translation,
    )

    reload_module(bayescatrack)

    assert installed_functions == (
        fov_registration.apply_integer_image_translation,
        fov_registration.apply_subpixel_image_translation,
        fov_registration.apply_integer_roi_mask_translation,
        fov_registration.apply_subpixel_roi_mask_translation,
    )
