"""Strict validation for track-refinement detection-count controls.

``TrackSmoothingConfig.min_track_detections`` is an integer scalar control.
The base implementation ultimately delegates non-float objects to
``operator.index`` but only normalizes ``TypeError``. Custom index-protocol
implementations can raise other conversion exceptions, which leaks
implementation-level errors instead of the public configuration validation
diagnostic.
"""

from __future__ import annotations

import operator
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_track_refinement_detection_count_validation_patch"
_INTEGER_MESSAGE = "min_track_detections must be an integer"
_MINIMUM_MESSAGE = "min_track_detections must be at least 2"


def install_track_refinement_detection_count_validation() -> None:
    """Install idempotent validation for track-refinement detection counts."""

    from . import track_refinement as module  # pylint: disable=import-outside-toplevel

    original_post_init = module.TrackSmoothingConfig.__post_init__
    if getattr(original_post_init, _PATCH_MARKER, False):
        return

    @wraps(original_post_init)
    def checked_post_init(self: Any) -> None:
        min_track_detections = _normalize_min_track_detections(
            self.min_track_detections
        )
        object.__setattr__(self, "min_track_detections", min_track_detections)
        original_post_init(self)
        object.__setattr__(self, "min_track_detections", min_track_detections)

    setattr(checked_post_init, _PATCH_MARKER, True)
    setattr(checked_post_init, "_bayescatrack_original", original_post_init)
    module.TrackSmoothingConfig.__post_init__ = checked_post_init


def _normalize_min_track_detections(value: Any) -> int:
    integer_value = _normalize_integer(value)
    if integer_value < 2:
        raise ValueError(_MINIMUM_MESSAGE)
    return integer_value


def _normalize_integer(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(_INTEGER_MESSAGE)

    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ValueError(_INTEGER_MESSAGE)
        value = value.item()
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(_INTEGER_MESSAGE)

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, (float, np.floating)):
        return _integer_from_float(float(value))

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(_INTEGER_MESSAGE)
        try:
            return _integer_from_float(float(stripped))
        except (TypeError, ValueError, OverflowError, ArithmeticError) as exc:
            raise ValueError(_INTEGER_MESSAGE) from exc

    if isinstance(value, (bytes, bytearray, np.bytes_)):
        raise ValueError(_INTEGER_MESSAGE)

    try:
        return int(operator.index(value))
    except (TypeError, ValueError, OverflowError, ArithmeticError) as exc:
        raise ValueError(_INTEGER_MESSAGE) from exc


def _integer_from_float(value: float) -> int:
    if not np.isfinite(value) or not value.is_integer():
        raise ValueError(_INTEGER_MESSAGE)
    return int(value)


__all__ = ["install_track_refinement_detection_count_validation"]
