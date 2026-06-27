"""Strict scalar validation for absence-model configuration values."""

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
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must be numeric, not boolean")
    if isinstance(value, (str, bytes, bytearray, np.str_, np.bytes_)):
        raise ValueError(f"{field_name} must be numeric, not text")
    try:
        value_array = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a numeric scalar") from exc
    if value_array.shape != ():
        raise ValueError(f"{field_name} must be a numeric scalar")


__all__ = ["install_absence_config_scalar_validation"]
