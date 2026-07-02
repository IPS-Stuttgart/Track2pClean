"""Strict manifest validation for teacher-rescue feature-gate scalars.

Teacher-adjacent rescue manifests accept optional feature-gate thresholds such as
``teacher_min_registered_iou`` and their ``teacher_gate_*`` aliases.  The base
manifest integration used a membership test against ``{None, ""}`` before
coercing these values to floats.  Malformed JSON values such as lists can
therefore fail with raw Python container errors instead of a field-specific
manifest validation error.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_bayescatrack_teacher_rescue_feature_gate_manifest_validation"
_ORIGINAL_ATTR = "_bayescatrack_original"

_FEATURE_GATE_FLOAT_OPTION_GROUPS: tuple[tuple[str, ...], ...] = (
    ("teacher_min_registered_iou", "teacher_gate_min_registered_iou"),
    ("teacher_max_registered_iou", "teacher_gate_max_registered_iou"),
    ("teacher_min_threshold_margin", "teacher_gate_min_threshold_margin"),
    ("teacher_min_row_margin", "teacher_gate_min_row_margin"),
    ("teacher_min_column_margin", "teacher_gate_min_column_margin"),
    ("teacher_max_centroid_distance", "teacher_gate_max_centroid_distance"),
    ("teacher_min_area_ratio", "teacher_gate_min_area_ratio"),
    ("teacher_min_cell_probability", "teacher_gate_min_cell_probability"),
)
_TEXT_OR_BYTES_LIKE = (bytes, bytearray, memoryview, np.bytes_)


def install_teacher_rescue_feature_gate_manifest_validation() -> None:
    """Install idempotent validation for teacher-rescue feature-gate options."""

    from bayescatrack.experiments import (  # pylint: disable=import-outside-toplevel
        _teacher_rescue_manifest_integration as base,
    )

    current_runner = base._run_track2p_policy_teacher_adjacent_rows
    if _callable_chain_has_patch(current_runner):
        return

    original_runner = current_runner

    @wraps(original_runner)
    def _run_teacher_rows_with_feature_gate_validation(
        config: Any, options: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        return original_runner(config, _normalize_feature_gate_options(options))

    setattr(_run_teacher_rows_with_feature_gate_validation, _PATCH_MARKER, True)
    setattr(_run_teacher_rows_with_feature_gate_validation, _ORIGINAL_ATTR, original_runner)
    base._run_track2p_policy_teacher_adjacent_rows = (
        _run_teacher_rows_with_feature_gate_validation
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


def _normalize_feature_gate_options(options: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(options)
    for option_names in _FEATURE_GATE_FLOAT_OPTION_GROUPS:
        for name in option_names:
            if name in normalized:
                normalized[name] = _optional_finite_float(normalized[name], name=name)
    return normalized


def _optional_finite_float(value: Any, *, name: str) -> float | None:
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ValueError(_finite_float_error(name))
        value = value.item()
    if value is None or (isinstance(value, str) and value == ""):
        return None
    return _finite_float(value, name=name)


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or isinstance(value, _TEXT_OR_BYTES_LIKE):
        raise ValueError(_finite_float_error(name))
    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(_finite_float_error(name)) from exc
    if not np.isfinite(numeric_value):
        raise ValueError(_finite_float_error(name))
    return numeric_value


def _finite_float_error(name: str) -> str:
    return f"{name} must be a finite float when provided"


__all__ = ["install_teacher_rescue_feature_gate_manifest_validation"]
