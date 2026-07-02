from __future__ import annotations

import operator
from functools import wraps
from typing import Any

import numpy as np

_MARKER = "_bayescatrack_comparison_limit_validation"
_MESSAGE = "limit must be a positive integer"


def install_comparison_limit_validation() -> None:
    from . import benchmark_comparison as module

    for name in (
        "build_subject_gap_summary_rows",
        "build_subject_deficit_summary_rows",
    ):
        original = getattr(module, name)
        if getattr(original, _MARKER, False):
            continue

        @wraps(original)
        def wrapped(*args: Any, _original: Any = original, **kwargs: Any) -> Any:
            if "limit" in kwargs:
                kwargs = dict(kwargs)
                kwargs["limit"] = _positive_int(kwargs["limit"])
            return _original(*args, **kwargs)

        setattr(wrapped, _MARKER, True)
        setattr(wrapped, "_bayescatrack_original", original)
        setattr(module, name, wrapped)


def _positive_int(value: Any) -> int:
    if isinstance(value, (bool, np.bool_, str, bytes, bytearray)):
        raise ValueError(_MESSAGE)
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ValueError(_MESSAGE)
        value = value.item()
        if isinstance(value, (bool, np.bool_, str, bytes, bytearray)):
            raise ValueError(_MESSAGE)
    try:
        result = int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(_MESSAGE) from exc
    if result < 1:
        raise ValueError(_MESSAGE)
    return result


__all__ = ["install_comparison_limit_validation"]
