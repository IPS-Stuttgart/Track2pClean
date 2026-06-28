"""Strict validation for benchmark seed-session selectors."""

from __future__ import annotations

import operator
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_benchmark_seed_session_validation_patch"
_ORIGINAL_ATTR = "_bayescatrack_benchmark_seed_session_validation_original"


def install_benchmark_seed_session_validation() -> None:
    """Install strict seed-session resolution on the Track2p benchmark module."""

    from bayescatrack.experiments import track2p_benchmark as _track2p_benchmark

    current = _track2p_benchmark._resolved_seed_sessions
    if getattr(current, _PATCH_MARKER, False):
        return
    setattr(_resolved_seed_sessions, _ORIGINAL_ATTR, current)
    setattr(_resolved_seed_sessions, _PATCH_MARKER, True)
    _track2p_benchmark._resolved_seed_sessions = _resolved_seed_sessions


def _resolved_seed_sessions(config: Any, *, n_sessions: int) -> tuple[int, ...]:
    """Resolve benchmark seed-session selectors without silent scalar coercion."""

    configured_seed_sessions = config.seed_sessions
    session_count = _session_count(n_sessions)
    if isinstance(configured_seed_sessions, str):
        if configured_seed_sessions.casefold() != "all":
            raise ValueError("seed_sessions string value must be 'all'")
        return tuple(range(session_count))

    if configured_seed_sessions is None:
        raw_seed_sessions = (config.seed_session,)
        value_name = "seed_session"
    else:
        try:
            raw_seed_sessions = tuple(configured_seed_sessions)
        except TypeError as exc:
            raise ValueError(
                "seed_sessions must contain integer session indices"
            ) from exc
        if raw_seed_sessions:
            value_name = "seed_sessions"
        else:
            raw_seed_sessions = (config.seed_session,)
            value_name = "seed_session"

    normalized = tuple(
        _parse_session_index(seed_session, name=value_name)
        for seed_session in raw_seed_sessions
    )
    for seed_session in normalized:
        if seed_session < 0 or seed_session >= session_count:
            raise IndexError(
                f"seed_session {seed_session} out of bounds for {n_sessions} sessions"
            )
    return normalized


def _parse_session_index(value: Any, *, name: str) -> int:
    message = f"{name} must contain integer session indices"
    if isinstance(value, (bool, np.bool_, str, bytes, bytearray)):
        raise ValueError(message)
    try:
        value_array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if value_array.shape != ():
        raise ValueError(message)
    scalar = value_array.item()
    if isinstance(scalar, (bool, np.bool_, str, bytes, bytearray)):
        raise ValueError(message)
    if isinstance(scalar, (float, np.floating)):
        numeric_value = float(scalar)
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(message)
        return int(numeric_value)
    try:
        return int(operator.index(scalar))
    except TypeError as exc:
        raise ValueError(message) from exc


def _session_count(value: Any) -> int:
    try:
        count = _parse_session_index(value, name="n_sessions")
    except ValueError as exc:
        raise ValueError("n_sessions must contain integer session indices") from exc
    if count < 0:
        raise ValueError("n_sessions must be non-negative")
    return count


__all__ = ["install_benchmark_seed_session_validation"]
