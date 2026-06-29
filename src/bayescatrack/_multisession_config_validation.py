"""Strict validation for multisession-tracking configuration scalars.

The base dataclass uses plain comparisons for numeric controls.  That leaves
ambiguous values such as booleans and non-finite floats able to pass through
because Python treats ``True`` as ``1`` and comparisons with ``NaN`` are false.
"""

from __future__ import annotations

import operator
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_multisession_config_validation_patch"


def install_multisession_config_validation() -> None:
    """Install idempotent validation for ``MultisessionTrackingConfig``."""

    from . import (
        multisession_tracking as _multisession_tracking,  # pylint: disable=import-outside-toplevel
    )

    config_cls = _multisession_tracking.MultisessionTrackingConfig
    if getattr(config_cls, _PATCH_MARKER, False):
        return

    original_post_init = config_cls.__post_init__

    @wraps(original_post_init)
    def __post_init__(self: Any) -> None:
        object.__setattr__(
            self,
            "max_session_gap",
            _positive_int(self.max_session_gap, name="max_session_gap"),
        )
        object.__setattr__(
            self,
            "weighted_centroids",
            _strict_bool(self.weighted_centroids, name="weighted_centroids"),
        )
        object.__setattr__(
            self,
            "velocity_variance",
            _finite_nonnegative_float(
                self.velocity_variance,
                name="velocity_variance",
            ),
        )
        object.__setattr__(
            self,
            "regularization",
            _finite_nonnegative_float(self.regularization, name="regularization"),
        )
        for field_name in ("start_cost", "end_cost", "gap_penalty"):
            object.__setattr__(
                self,
                field_name,
                _finite_nonnegative_float(
                    getattr(self, field_name),
                    name=field_name,
                ),
            )
        if self.cost_threshold is not None:
            object.__setattr__(
                self,
                "cost_threshold",
                _finite_nonnegative_float(
                    self.cost_threshold,
                    name="cost_threshold",
                ),
            )
        object.__setattr__(
            self,
            "return_pairwise_components",
            _strict_bool(
                self.return_pairwise_components,
                name="return_pairwise_components",
            ),
        )
        original_post_init(self)

    setattr(__post_init__, _PATCH_MARKER, True)
    setattr(__post_init__, "_bayescatrack_original", original_post_init)
    config_cls.__post_init__ = __post_init__
    setattr(config_cls, _PATCH_MARKER, True)


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(f"{name} must be a positive integer")
        integer_value = int(numeric_value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{name} must be a positive integer")
        try:
            numeric_value = float(text)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a positive integer") from exc
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(f"{name} must be a positive integer")
        integer_value = int(numeric_value)
    else:
        try:
            integer_value = operator.index(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a positive integer") from exc
    if integer_value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(integer_value)


def _strict_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def _finite_nonnegative_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite non-negative value")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite non-negative value") from exc
    if not np.isfinite(numeric_value) or numeric_value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative value")
    return numeric_value


__all__ = ["install_multisession_config_validation"]
