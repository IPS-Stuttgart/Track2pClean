"""Strict validation for Track2p-policy seed-sensitivity audit seeds."""

from __future__ import annotations

import operator
from collections.abc import Sequence
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_track2pclean_seed_sensitivity_seed_validation_patch"


def install_seed_sensitivity_audit_validation() -> None:
    """Install idempotent validation for seed-sensitivity audit seed controls."""

    from bayescatrack.experiments import track2p_policy_seed_sensitivity_audit as audit

    original = audit._resolved_seed_sessions  # pylint: disable=protected-access
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def _validated_resolved_seed_sessions(
        seed_sessions: Sequence[int] | str,
        *,
        n_sessions: Any,
    ) -> tuple[int, ...]:
        session_count = _positive_session_count(n_sessions)
        values = _seed_session_values(seed_sessions)
        if values == ("all",):
            return tuple(range(session_count))
        if not values:
            raise ValueError("seed_sessions must not be empty")
        return tuple(
            _seed_session_index(value, n_sessions=session_count) for value in values
        )

    setattr(_validated_resolved_seed_sessions, _PATCH_MARKER, True)
    setattr(_validated_resolved_seed_sessions, "_track2pclean_original", original)
    audit._resolved_seed_sessions = _validated_resolved_seed_sessions  # type: ignore[assignment]  # pylint: disable=protected-access


def _seed_session_values(seed_sessions: Sequence[int] | str) -> tuple[Any, ...]:
    if isinstance(seed_sessions, str):
        if seed_sessions.casefold() == "all":
            return ("all",)
        tokens = tuple(token.strip() for token in seed_sessions.split(","))
        if not tokens or any(not token for token in tokens):
            raise ValueError("seed_sessions must be a comma-separated list of integers")
        return tokens
    try:
        return tuple(seed_sessions)
    except TypeError as exc:
        raise ValueError("seed_sessions must be 'all' or an iterable of integer session indices") from exc


def _positive_session_count(value: Any) -> int:
    n_sessions = _integer_value(value, field_name="n_sessions")
    if n_sessions <= 0:
        raise ValueError("n_sessions must be positive")
    return n_sessions


def _seed_session_index(value: Any, *, n_sessions: int) -> int:
    seed_session = _integer_value(value, field_name="seed_sessions")
    if seed_session < 0 or seed_session >= n_sessions:
        raise ValueError(f"seed_session {seed_session} out of bounds for {n_sessions} sessions")
    return seed_session


def _integer_value(value: Any, *, field_name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must contain integer session indices")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(f"{field_name} must contain integer session indices")
        return int(numeric_value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{field_name} must contain integer session indices")
        try:
            return int(text, 10)
        except ValueError as exc:
            raise ValueError(f"{field_name} must contain integer session indices") from exc
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise ValueError(f"{field_name} must contain integer session indices") from exc


__all__ = ["install_seed_sensitivity_audit_validation"]
