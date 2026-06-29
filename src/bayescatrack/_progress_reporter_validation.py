from __future__ import annotations

import operator
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_progress_reporter_validation_patch"


def install_progress_reporter_validation() -> None:
    """Install strict validation for ``ProgressReporter`` constructor inputs."""

    from .experiments import track2p_benchmark as benchmark_module

    original_init = benchmark_module.ProgressReporter.__init__
    if getattr(original_init, _PATCH_MARKER, False):
        return

    @wraps(original_init)
    def progress_reporter_init_with_validation(
        self: Any,
        total: Any,
        *,
        enabled: Any,
        label: Any,
    ) -> None:
        original_init(
            self,
            _positive_integer(total, name="total"),
            enabled=_strict_bool(enabled, name="enabled"),
            label=label,
        )

    setattr(progress_reporter_init_with_validation, _PATCH_MARKER, True)
    setattr(
        progress_reporter_init_with_validation,
        "_bayescatrack_original",
        original_init,
    )
    benchmark_module.ProgressReporter.__init__ = progress_reporter_init_with_validation


def _strict_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f"{name} must be a boolean")


def _positive_integer(value: Any, *, name: str) -> int:
    error_message = f"{name} must be a positive integer"
    if isinstance(value, (bool, np.bool_, str, bytes, bytearray)):
        raise ValueError(error_message)

    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ValueError(error_message)
        value = value.item()

    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(error_message)
        normalized = int(numeric)
    else:
        try:
            normalized = operator.index(value)
        except TypeError as exc:
            raise ValueError(error_message) from exc
        except (ValueError, OverflowError) as exc:
            raise ValueError(error_message) from exc

    normalized = int(normalized)
    if normalized <= 0:
        raise ValueError(error_message)
    return normalized


__all__ = ["install_progress_reporter_validation"]