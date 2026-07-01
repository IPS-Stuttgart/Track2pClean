from __future__ import annotations

from functools import wraps
from types import ModuleType
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_calibration_threshold_validation_patch"
_ERROR_MESSAGE = "thresholds must be finite numeric values in [0, 1]"
_TEXT_TYPES = (str, bytes, np.str_, np.bytes_)


def install_calibration_threshold_validation(module: ModuleType) -> None:
    original = (
        module._validate_probability_threshold
    )  # pylint: disable=protected-access
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def threshold_with_text_rejection(threshold: Any) -> float:
        if isinstance(threshold, _TEXT_TYPES):
            raise ValueError(_ERROR_MESSAGE)
        return original(threshold)

    setattr(threshold_with_text_rejection, _PATCH_MARKER, True)
    setattr(threshold_with_text_rejection, "_bayescatrack_original", original)
    module._validate_probability_threshold = (
        threshold_with_text_rejection  # pylint: disable=protected-access
    )


__all__ = ["install_calibration_threshold_validation"]
