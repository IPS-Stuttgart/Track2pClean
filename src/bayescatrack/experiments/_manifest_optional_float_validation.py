"""Normalize optional-float manifest validation errors."""

from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
import math
from typing import Any

_PATCH_MARKER = "_bayescatrack_optional_float_manifest_validation_patch"
_ORIGINAL_ATTR = "_bayescatrack_original"
_BENCHMARK_EMPTY_TEXT = frozenset({"", "none", "null", "off", "disabled"})


def install_optional_float_manifest_validation() -> None:
    """Install idempotent optional-float guards for benchmark manifests."""

    from bayescatrack.experiments import (  # pylint: disable=import-outside-toplevel
        _teacher_rescue_manifest_integration as teacher_rescue,
    )
    from bayescatrack.experiments import (  # pylint: disable=import-outside-toplevel
        benchmark_manifest as manifest,
    )

    _patch_benchmark_manifest_optional_float(manifest)
    _patch_teacher_rescue_optional_float(teacher_rescue)


def _patch_benchmark_manifest_optional_float(manifest: Any) -> None:
    current = manifest._optional_float_option  # pylint: disable=protected-access
    if getattr(current, _PATCH_MARKER, False):
        return

    original = current

    @wraps(original)
    def _optional_float_option_with_validation(
        options: Mapping[str, Any], *keys: str
    ) -> float | None:
        for key in keys:
            if key not in options or options[key] is None:
                continue
            value = options[key]
            if _is_benchmark_empty_text(value):
                return None
            return _finite_float(value, name=key)
        return None

    _mark_wrapper(_optional_float_option_with_validation, original)
    manifest._optional_float_option = (  # pylint: disable=protected-access
        _optional_float_option_with_validation
    )


def _patch_teacher_rescue_optional_float(teacher_rescue: Any) -> None:
    current = teacher_rescue._optional_float_option  # pylint: disable=protected-access
    if getattr(current, _PATCH_MARKER, False):
        return

    original = current

    @wraps(original)
    def _optional_float_option_with_validation(
        options: Mapping[str, Any], *keys: str
    ) -> float | None:
        for key in keys:
            if key not in options:
                continue
            value = options[key]
            if value is None or _is_teacher_rescue_empty_text(value):
                continue
            return _finite_float(value, name=key)
        return None

    _mark_wrapper(_optional_float_option_with_validation, original)
    teacher_rescue._optional_float_option = (  # pylint: disable=protected-access
        _optional_float_option_with_validation
    )


def _mark_wrapper(wrapper: Any, original: Any) -> None:
    setattr(wrapper, _PATCH_MARKER, True)
    setattr(wrapper, _ORIGINAL_ATTR, original)


def _is_benchmark_empty_text(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in _BENCHMARK_EMPTY_TEXT


def _is_teacher_rescue_empty_text(value: Any) -> bool:
    return isinstance(value, str) and value == ""


def _finite_float(value: Any, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


__all__ = ["install_optional_float_manifest_validation"]
