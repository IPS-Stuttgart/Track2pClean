"""Strict validation for Track2p benchmark seed-session controls."""

from __future__ import annotations

import operator
from typing import Any

import numpy as np

_ORIGINAL_FN_ATTR = "_bayescatrack_original_resolved_seed_sessions"
_PATCH_ATTR = "_bayescatrack_seed_session_validation"


def install_seed_session_validation() -> None:
    """Install an idempotent seed-session validator on the benchmark module."""

    from bayescatrack.experiments import track2p_benchmark as _benchmark

    if getattr(_benchmark._resolved_seed_sessions, _PATCH_ATTR, False):
        return
    if not hasattr(_benchmark, _ORIGINAL_FN_ATTR):
        setattr(_benchmark, _ORIGINAL_FN_ATTR, _benchmark._resolved_seed_sessions)

    def _validated_resolved_seed_sessions(
        config: Any,
        *,
        n_sessions: Any,
    ) -> tuple[int, ...]:
        session_count = _positive_session_count(n_sessions)
        configured_seed_sessions = config.seed_sessions
        if isinstance(configured_seed_sessions, str):
            if configured_seed_sessions.casefold() != "all":
                raise ValueError("seed_sessions string value must be 'all'")
            return tuple(range(session_count))

        seed_values, uses_seed_session_fallback = _seed_session_values(
            configured_seed_sessions,
            fallback=config.seed_session,
        )
        field_name = "seed_session" if uses_seed_session_fallback else "seed_sessions"
        return tuple(
            _seed_session_index(
                seed_session,
                field_name=field_name,
                n_sessions=session_count,
            )
            for seed_session in seed_values
        )

    setattr(_validated_resolved_seed_sessions, _PATCH_ATTR, True)
    setattr(
        _validated_resolved_seed_sessions,
        "_bayescatrack_original",
        getattr(_benchmark, _ORIGINAL_FN_ATTR),
    )
    _benchmark._resolved_seed_sessions = _validated_resolved_seed_sessions


def _seed_session_values(
    configured_seed_sessions: Any,
    *,
    fallback: Any,
) -> tuple[tuple[Any, ...], bool]:
    if configured_seed_sessions is None:
        return (fallback,), True
    try:
        seed_values = tuple(configured_seed_sessions)
    except TypeError as exc:
        raise ValueError(
            "seed_sessions must be 'all' or an iterable of integer session indices"
        ) from exc
    if not seed_values:
        return (fallback,), True
    return seed_values, False


def _positive_session_count(value: Any) -> int:
    n_sessions = _integer_value(value, field_name="n_sessions")
    if n_sessions <= 0:
        raise ValueError("n_sessions must be positive")
    return n_sessions


def _seed_session_index(value: Any, *, field_name: str, n_sessions: int) -> int:
    seed_session = _integer_value(value, field_name=field_name)
    if seed_session < 0 or seed_session >= n_sessions:
        raise IndexError(
            f"seed_session {seed_session} out of bounds for {n_sessions} sessions"
        )
    return seed_session


def _integer_value(value: Any, *, field_name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must contain integer session indices")
    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(f"{field_name} must contain integer session indices")
        return int(numeric_value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{field_name} must contain integer session indices")
        try:
            numeric_value = float(stripped)
        except ValueError as exc:
            raise ValueError(
                f"{field_name} must contain integer session indices"
            ) from exc
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(f"{field_name} must contain integer session indices")
        return int(numeric_value)
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise ValueError(f"{field_name} must contain integer session indices") from exc


__all__ = ["install_seed_session_validation"]
