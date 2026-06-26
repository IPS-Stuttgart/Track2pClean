"""Compatibility patch for multisession solver signature variants."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any

_PATCH_MARKER = "_bayescatrack_multisession_solver_signature_validation_patch"


def install_multisession_solver_signature_validation(module: Any | None = None) -> None:
    """Install idempotent compatibility for legacy PyRecEst solver kwargs."""

    if module is None:
        from . import multisession_tracking as target_module  # pylint: disable=import-outside-toplevel
    else:
        target_module = module

    original_compatible_solver_call_attempts = target_module._compatible_solver_call_attempts
    if getattr(original_compatible_solver_call_attempts, _PATCH_MARKER, False):
        return

    def _compatible_solver_call_attempts_with_legacy_gap_names(
        solver: Callable[..., Any],
        attempts: Sequence[dict[str, Any]],
    ) -> tuple[tuple[dict[str, Any], bool], ...]:
        expanded_attempts = _with_cost_threshold_free_variants(attempts)
        try:
            signature = inspect.signature(solver)
        except (TypeError, ValueError):
            return tuple((kwargs, True) for kwargs in expanded_attempts)

        if any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            return ((expanded_attempts[0], False),)

        supported_keyword_names = {
            name
            for name, parameter in signature.parameters.items()
            if parameter.kind
            in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
        }

        compatible_attempts: list[tuple[dict[str, Any], bool]] = []
        seen_keyword_sets: set[tuple[str, ...]] = set()
        for kwargs in expanded_attempts:
            if not set(kwargs).issubset(supported_keyword_names):
                continue
            keyword_set = tuple(sorted(kwargs))
            if keyword_set in seen_keyword_sets:
                continue
            seen_keyword_sets.add(keyword_set)
            compatible_attempts.append((kwargs, False))
        return tuple(compatible_attempts)

    setattr(_compatible_solver_call_attempts_with_legacy_gap_names, _PATCH_MARKER, True)
    setattr(
        _compatible_solver_call_attempts_with_legacy_gap_names,
        "_bayescatrack_original",
        original_compatible_solver_call_attempts,
    )
    target_module._compatible_solver_call_attempts = _compatible_solver_call_attempts_with_legacy_gap_names


def _with_cost_threshold_free_variants(
    attempts: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    expanded_attempts: list[dict[str, Any]] = []
    seen_keyword_sets: set[tuple[str, ...]] = set()

    def _append_once(kwargs: Mapping[str, Any]) -> None:
        keyword_set = tuple(sorted(kwargs))
        if keyword_set in seen_keyword_sets:
            return
        seen_keyword_sets.add(keyword_set)
        expanded_attempts.append(dict(kwargs))

    for kwargs in attempts:
        _append_once(kwargs)
        if "cost_threshold" in kwargs:
            kwargs_without_threshold = dict(kwargs)
            del kwargs_without_threshold["cost_threshold"]
            _append_once(kwargs_without_threshold)

    return tuple(expanded_attempts)


__all__ = ["install_multisession_solver_signature_validation"]
