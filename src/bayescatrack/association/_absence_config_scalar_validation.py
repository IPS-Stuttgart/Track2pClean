"""Strict scalar validation for absence-model configuration values.

``AbsenceModelConfig`` normalizes numeric controls with ``float(...)``.  That
keeps numeric scalars convenient, but it can also accept text/bytes or
one-element array-like values that usually indicate malformed benchmark
configuration.  This import-time hook rejects those ambiguous values before the
configuration dataclass performs its ordinary finite/non-negative checks.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np

_PATCH_ATTR = "_track2pclean_absence_config_scalar_validation"
_ORIGINAL_ATTR = "_track2pclean_absence_config_scalar_validation_original"

_NUMERIC_FIELDS = (
    "base_absence_cost",
    "out_of_fov_discount",
    "low_cell_probability_discount",
    "empty_registered_mask_discount",
    "high_local_density_discount",
    "trace_missing_discount",
    "min_cost",
)


_TEXT_TYPES = (str, bytes, bytearray, np.str_, np.bytes_)


def install_absence_config_scalar_validation(absence_model: ModuleType) -> None:
    """Install idempotent strict scalar validation on ``AbsenceModelConfig``."""

    config_cls = absence_model.AbsenceModelConfig
    if getattr(config_cls, _PATCH_ATTR, False):
        return

    original_post_init = config_cls.__post_init__

    def _validated_post_init(self: Any) -> None:
        for field_name in _NUMERIC_FIELDS:
            _validate_numeric_scalar(getattr(self, field_name), field_name)
        original_post_init(self)

    _validated_post_init.__name__ = "__post_init__"
    _validated_post_init.__qualname__ = f"{config_cls.__name__}.__post_init__"
    setattr(_validated_post_init, _ORIGINAL_ATTR, original_post_init)
    config_cls.__post_init__ = _validated_post_init
    setattr(config_cls, _PATCH_ATTR, True)


def _validate_numeric_scalar(value: Any, field_name: str) -> None:
    message = f"{field_name} must be a numeric scalar"
    if isinstance(value, (bool, np.bool_, *_TEXT_TYPES)):
        raise ValueError(message)

    try:
        value_array = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if value_array.shape != ():
        raise ValueError(message)

    scalar_value = value_array.item()
    if isinstance(scalar_value, (bool, np.bool_, *_TEXT_TYPES)):
        raise ValueError(message)
    try:
        float(scalar_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc


__all__ = ["install_absence_config_scalar_validation"]
