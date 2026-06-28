"""Reject duplicate growth-analysis target-session selectors.

The radial and affine growth helpers iterate once per requested target session.
If two raw selectors normalize to the same session index, output rows and
summaries are emitted more than once and downstream aggregate counts are
inflated.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import wraps
from typing import Any

_PATCH_MARKER = "_bayescatrack_growth_target_sessions_validation_patch"


def install_growth_target_sessions_validation() -> None:
    """Install idempotent duplicate rejection for growth-analysis targets."""

    from .analysis import growth as _growth  # pylint: disable=import-outside-toplevel

    original_target_sessions = _growth._target_sessions
    if getattr(original_target_sessions, _PATCH_MARKER, False):
        return

    @wraps(original_target_sessions)
    def _target_sessions_without_duplicates(
        *,
        n_sessions: int,
        source_session: int,
        target_sessions: Sequence[Any] | None,
    ) -> tuple[int, ...]:
        targets = original_target_sessions(
            n_sessions=n_sessions,
            source_session=source_session,
            target_sessions=target_sessions,
        )
        if target_sessions is not None:
            _raise_on_duplicate_targets(targets)
        return targets

    setattr(_target_sessions_without_duplicates, _PATCH_MARKER, True)
    setattr(
        _target_sessions_without_duplicates,
        "_bayescatrack_original",
        original_target_sessions,
    )
    _growth._target_sessions = _target_sessions_without_duplicates


def _raise_on_duplicate_targets(targets: Sequence[int]) -> None:
    seen: set[int] = set()
    duplicates: set[int] = set()
    for target in targets:
        if target in seen:
            duplicates.add(int(target))
        seen.add(int(target))
    if duplicates:
        raise ValueError(
            "duplicate target_sessions are not allowed: " f"{sorted(duplicates)}"
        )


__all__ = ["install_growth_target_sessions_validation"]
