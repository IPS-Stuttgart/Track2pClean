"""Strict validation for global-assignment session edge ranges."""

from __future__ import annotations

import operator
from typing import Any

import numpy as np

from . import pyrecest_global_assignment as _global_assignment

_PATCH_ATTR = "_bayescatrack_session_edge_pair_validation_patch"


def install_session_edge_pair_validation() -> None:
    """Install an idempotent validator around ``session_edge_pairs``."""

    original = _global_assignment.session_edge_pairs
    if getattr(original, _PATCH_ATTR, False):
        return

    def session_edge_pairs(
        num_sessions: Any,
        *,
        max_gap: Any = 2,
    ) -> tuple[tuple[int, int], ...]:
        return original(
            _nonnegative_integer_like(num_sessions, name="num_sessions"),
            max_gap=_positive_integer_like(max_gap, name="max_gap"),
        )

    setattr(session_edge_pairs, _PATCH_ATTR, True)
    setattr(session_edge_pairs, "_bayescatrack_original", original)
    _global_assignment.session_edge_pairs = session_edge_pairs  # type: ignore[assignment]


def _nonnegative_integer_like(value: Any, *, name: str) -> int:
    integer_value = _integer_like(value, name=name)
    if integer_value < 0:
        raise ValueError(f"{name} must be non-negative")
    return integer_value


def _positive_integer_like(value: Any, *, name: str) -> int:
    integer_value = _integer_like(value, name=name)
    if integer_value < 1:
        raise ValueError(f"{name} must be at least 1")
    return integer_value


def _integer_like(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or not float(value).is_integer():
            raise ValueError(f"{name} must be an integer")
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{name} must be an integer")
        try:
            numeric_value = float(stripped)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(f"{name} must be an integer")
        return int(numeric_value)
    try:
        return int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


__all__ = ["install_session_edge_pair_validation"]
