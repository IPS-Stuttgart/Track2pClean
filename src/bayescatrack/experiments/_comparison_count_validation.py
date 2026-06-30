from __future__ import annotations

import operator
from functools import wraps
from typing import Any

import numpy as np

_MARKER = "_bayescatrack_comparison_count_validation"


def install_comparison_count_validation() -> None:
    from . import benchmark_comparison as module

    original = module._int_value  # pylint: disable=protected-access
    if getattr(original, _MARKER, False):
        return

    @wraps(original)
    def int_value_with_validation(row: dict[str, str], key: str) -> int:
        value = row.get(key)
        if value is None or value == "":
            return 0
        return _integer_count(value, key=key)

    setattr(int_value_with_validation, _MARKER, True)
    setattr(int_value_with_validation, "_bayescatrack_original", original)
    module._int_value = int_value_with_validation  # pylint: disable=protected-access


def _integer_count(value: Any, *, key: str) -> int:
    message = f"benchmark count field {key!r} must be an integer value"
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ValueError(message)
        value = value.item()
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(message)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0
        try:
            return int(text, 10)
        except ValueError:
            try:
                numeric = float(text)
            except ValueError as exc:
                raise ValueError(message) from exc
            if not np.isfinite(numeric) or not numeric.is_integer():
                raise ValueError(message)
            return int(numeric)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(message)
        return int(numeric)
    try:
        return int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc


__all__ = ["install_comparison_count_validation"]
