"""Validation hardening for growth-analysis target-session inputs."""

from __future__ import annotations

from collections.abc import Sequence
from functools import wraps
from typing import Any

_STRING_LIKE_TARGET_SESSIONS = (str, bytes, bytearray, memoryview)


def install_growth_target_sessions_validation(growth_module: Any) -> None:
    """Reject string-like target-session sequences before element-wise parsing.

    Growth APIs accept a sequence of session indices. Plain strings and binary
    buffers are also sequences in Python, but treating ``"10"`` as ``("1", "0")``
    silently selects the wrong target sessions. Install a narrow guard around
    the growth module's helper while keeping the canonical implementation in
    ``growth.py`` responsible for all normal index validation.
    """

    original_target_sessions = growth_module._target_sessions
    if getattr(original_target_sessions, "_rejects_string_like_target_sessions", False):
        return

    @wraps(original_target_sessions)
    def _target_sessions(
        *,
        n_sessions: int,
        source_session: int,
        target_sessions: Sequence[int] | None,
    ) -> tuple[int, ...]:
        if isinstance(target_sessions, _STRING_LIKE_TARGET_SESSIONS):
            raise ValueError(
                "target_sessions must be a sequence of integer-like session indices, "
                "not string-like input"
            )
        return original_target_sessions(
            n_sessions=n_sessions,
            source_session=source_session,
            target_sessions=target_sessions,
        )

    _target_sessions._rejects_string_like_target_sessions = True  # type: ignore[attr-defined]
    growth_module._target_sessions = _target_sessions
