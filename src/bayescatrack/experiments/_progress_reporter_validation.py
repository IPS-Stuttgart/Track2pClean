from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_progress_reporter_validation_patch"


def install_progress_reporter_validation() -> None:
    """Install strict validation for ``ProgressReporter.enabled``."""

    from . import track2p_benchmark as benchmark_module

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
            total,
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


__all__ = ["install_progress_reporter_validation"]
