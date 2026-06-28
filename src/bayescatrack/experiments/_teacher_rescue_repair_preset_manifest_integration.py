"""Manifest support for Track2p teacher-rescue repair presets.

The CLI can apply high-level teacher-rescue macros such as
``missing-seed-high-confidence``.  JSON benchmark manifests previously accepted
only the lower-level boolean and threshold knobs, so a manifest row could not use
that macro directly.  This integration expands repair presets into those
existing options before the teacher-adjacent rescue runner is invoked.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
from typing import Any

TEACHER_REPAIR_PRESET_FIELD = "teacher_repair_preset"
TEACHER_RESCUE_RUNNER = "track2p-policy-teacher-adjacent-rescue"
_PATCH_MARKER = "_bayescatrack_teacher_repair_preset_integration"


def install_teacher_rescue_repair_preset_manifest_integration() -> None:
    """Install manifest expansion for teacher-rescue repair presets."""

    from bayescatrack.experiments import _teacher_rescue_manifest_integration as base
    from bayescatrack.experiments import benchmark_manifest as manifest

    base.TEACHER_ADJACENT_RESCUE_FIELDS.add(TEACHER_REPAIR_PRESET_FIELD)
    manifest.RUNNER_SPECIFIC_FIELDS.add(TEACHER_REPAIR_PRESET_FIELD)
    manifest.RUN_SPEC_FIELDS.add(TEACHER_REPAIR_PRESET_FIELD)
    manifest.RUNNER_CONFIG_FIELDS.setdefault(
        TEACHER_RESCUE_RUNNER, set(manifest.TRACK2P_CONFIG_FIELDS)
    ).add(TEACHER_REPAIR_PRESET_FIELD)

    original_runner = base._run_track2p_policy_teacher_adjacent_rows
    if _callable_chain_has_patch(original_runner):
        manifest._bayescatrack_teacher_repair_preset_integration = True
        return

    @wraps(original_runner)
    def _run_teacher_rows_with_repair_preset(
        config: Any, options: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        return original_runner(config, _expand_teacher_repair_preset(options))

    setattr(_run_teacher_rows_with_repair_preset, _PATCH_MARKER, True)
    setattr(
        _run_teacher_rows_with_repair_preset, "_bayescatrack_original", original_runner
    )
    base._run_track2p_policy_teacher_adjacent_rows = (
        _run_teacher_rows_with_repair_preset
    )
    manifest._bayescatrack_teacher_repair_preset_integration = True


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
        current = getattr(current, "_bayescatrack_original", None)
    return False


def _expand_teacher_repair_preset(options: Mapping[str, Any]) -> dict[str, Any]:
    """Return options with high-level repair presets expanded."""

    expanded = dict(options)
    preset = str(expanded.get(TEACHER_REPAIR_PRESET_FIELD, "none")).strip().lower()
    if preset in {"", "none"}:
        return expanded

    from bayescatrack.experiments.track2p_policy_teacher_adjacent_rescue import (
        teacher_adjacent_repair_preset_kwargs,
    )

    defaults = teacher_adjacent_repair_preset_kwargs(preset)

    for key, value in defaults.items():
        if key == "min_component_observations":
            expanded[key] = max(int(value), int(expanded.get(key, 1)))
        else:
            expanded.setdefault(key, value)
    return expanded
