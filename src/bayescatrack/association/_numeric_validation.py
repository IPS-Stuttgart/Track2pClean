"""Shared numeric validators for association utilities."""

from __future__ import annotations

from typing import Any

import numpy as np


def validated_numeric_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def finite_positive_float(value: Any, *, name: str) -> float:
    numeric = validated_numeric_float(value, name=name)
    if numeric <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return numeric


def finite_nonnegative_float(value: Any, *, name: str) -> float:
    numeric = validated_numeric_float(value, name=name)
    if numeric < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return numeric


def finite_nonzero_float(value: Any, *, name: str) -> float:
    numeric = validated_numeric_float(value, name=name)
    if numeric == 0.0:
        raise ValueError(f"{name} values must be finite and non-zero")
    return numeric


def probability(value: Any, *, name: str, allow_zero: bool = True) -> float:
    numeric = validated_numeric_float(value, name=name)
    lower_ok = numeric >= 0.0 if allow_zero else numeric > 0.0
    if not lower_ok or numeric > 1.0:
        interval = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"{name} must be a finite value in {interval}")
    return numeric


def integer(value: Any, *, name: str) -> int:
    numeric = validated_numeric_float(value, name=name)
    if not numeric.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(numeric)


def positive_integer(value: Any, *, name: str) -> int:
    numeric = validated_numeric_float(value, name=name)
    if not numeric.is_integer() or numeric < 1.0:
        raise ValueError(f"{name} must be a positive integer")
    return int(numeric)


def nonnegative_integer(value: Any, *, name: str) -> int:
    numeric = validated_numeric_float(value, name=name)
    if not numeric.is_integer() or numeric < 0.0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(numeric)
