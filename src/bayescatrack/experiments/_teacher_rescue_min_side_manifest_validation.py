"""Strict manifest validation for teacher-rescue component split controls."""

from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
from numbers import Integral
from typing import Any

_PATCH_MARKER = "_bayescatrack_teacher_rescue_min_side_manifest_validation"
_ORIGINAL_ATTR = "_bayescatrack_original"


def install_teacher_rescue_min_side_manifest_validation() -> None:
    """Install idempotent validation for manifest ``min_side_observations``."""

    from bayescatrack.experiments import (  # pylint: disable=import-outside-toplevel
        _teacher_rescue_manifest_integration as base,
    )
    from bayescatrack.experiments import benchmark_manifest as manifest

    current_runner = base._run_track2p_policy_teacher_adjacent_rows
    if not _callable_chain_has_patch(current_runner):
        original_runner = current_runner

        @wraps(original_runner)
        def _run_teacher_rows_with_min_side_validation(
            config: Any, options: Mapping[str, Any]
        ) -> list[dict[str, Any]]:
            _validate_min_side_options(options)
            return original_runner(config, options)

        setattr(_run_teacher_rows_with_min_side_validation, _PATCH_MARKER, True)
        setattr(_run_teacher_rows_with_min_side_validation, _ORIGINAL_ATTR, original_runner)
        base._run_track2p_policy_teacher_adjacent_rows = (
            _run_teacher_rows_with_min_side_validation
        )

    current_benchmark_runner = manifest._run_benchmark_rows
    if not _callable_chain_has_patch(current_benchmark_runner):
        original_benchmark_runner = current_benchmark_runner

        @wraps(original_benchmark_runner)
        def _run_benchmark_rows_with_min_side_validation(
            run_spec: Any,
        ) -> list[dict[str, Any]]:
            if getattr(run_spec, "runner", None) == base.TEACHER_ADJACENT_RESCUE_RUNNER:
                _validate_min_side_options(dict(getattr(run_spec, "runner_kwargs", {}) or {}))
            return original_benchmark_runner(run_spec)

        setattr(_run_benchmark_rows_with_min_side_validation, _PATCH_MARKER, True)
        setattr(
            _run_benchmark_rows_with_min_side_validation,
            _ORIGINAL_ATTR,
            original_benchmark_runner,
        )
        manifest._run_benchmark_rows = _run_benchmark_rows_with_min_side_validation


def _validate_min_side_options(options: Mapping[str, Any]) -> None:
    if "min_side_observations" in options:
        _positive_int_value(
            options["min_side_observations"], name="min_side_observations"
        )


def _callable_chain_has_patch(function: Any) -> bool:
    seen: set[int] = set()
    current: Any = function
    while current is not None:
        current_id = id(current)
        if current_id in seen:
            return False
        if getattr(current, _PATCH_MARKER, False):
            return True
        seen.add(current_id)
        current = getattr(current, _ORIGINAL_ATTR, None)
    return False


def _positive_int_value(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    if isinstance(value, Integral):
        parsed = int(value)
    elif isinstance(value, str):
        try:
            parsed = int(value, 10)
        except ValueError as exc:
            raise ValueError(f"{name} must be a positive integer") from exc
    else:
        raise ValueError(f"{name} must be a positive integer")
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


__all__ = ["install_teacher_rescue_min_side_manifest_validation"]
