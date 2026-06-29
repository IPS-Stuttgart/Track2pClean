from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np


def install_progress_enabled_validation() -> None:
    from .experiments import track2p_benchmark

    original = track2p_benchmark.ProgressReporter.__init__
    if getattr(original, "_progress_enabled_validation", False):
        return

    @wraps(original)
    def checked_init(self: Any, total: Any, *, enabled: Any, label: Any) -> None:
        if not isinstance(enabled, (bool, np.bool_)):
            raise ValueError("enabled must be a boolean")
        original(self, total, enabled=bool(enabled), label=label)

    setattr(checked_init, "_progress_enabled_validation", True)
    setattr(checked_init, "_bayescatrack_original", original)
    track2p_benchmark.ProgressReporter.__init__ = checked_init


__all__ = ["install_progress_enabled_validation"]
