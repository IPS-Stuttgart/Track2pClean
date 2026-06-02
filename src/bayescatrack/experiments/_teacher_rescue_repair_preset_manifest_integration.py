"""Manifest support for Track2p teacher-rescue repair presets.

The CLI can apply high-level teacher-rescue macros such as
``missing-seed-high-confidence``.  JSON benchmark manifests previously accepted
only the lower-level boolean and threshold knobs, so a manifest row could not use
that macro directly.  This integration expands repair presets into those
existing options before the teacher-adjacent rescue runner is invoked.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

TEACHER_REPAIR_PRESET_FIELD = "teacher_repair_preset"
TEACHER_RESCUE_RUNNER = "track2p-policy-teacher-adjacent-rescue"


def install_teacher_rescue_repair_preset_manifest_integration() -> None:
    """Install manifest expansion for teacher-rescue repair presets."""

    from bayescatrack.experiments import benchmark_manifest as manifest
    from bayescatrack.experiments import _teacher_rescue_manifest_integration as base

    if getattr(manifest, "_bayescatrack_teacher_repair_preset_integration", False):
        return

    base.TEACHER_ADJACENT_RESCUE_FIELDS.add(TEACHER_REPAIR_PRESET_FIELD)
    manifest.RUNNER_SPECIFIC_FIELDS.add(TEACHER_REPAIR_PRESET_FIELD)
    manifest.RUN_SPEC_FIELDS.add(TEACHER_REPAIR_PRESET_FIELD)
    manifest.RUNNER_CONFIG_FIELDS.setdefault(
        TEACHER_RESCUE_RUNNER, set(manifest.TRACK2P_CONFIG_FIELDS)
    ).add(TEACHER_REPAIR_PRESET_FIELD)

    original_runner = base._run_track2p_policy_teacher_adjacent_rows

    def _run_teacher_rows_with_repair_preset(
        config: Any, options: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        return original_runner(config, _expand_teacher_repair_preset(options))

    base._run_track2p_policy_teacher_adjacent_rows = _run_teacher_rows_with_repair_preset
    manifest._bayescatrack_teacher_repair_preset_integration = True


def _expand_teacher_repair_preset(options: Mapping[str, Any]) -> dict[str, Any]:
    """Return options with high-level repair presets expanded."""

    expanded = dict(options)
    preset = str(expanded.get(TEACHER_REPAIR_PRESET_FIELD, "none")).strip().lower()
    if preset in {"", "none"}:
        return expanded
    if preset != "missing-seed-high-confidence":
        raise ValueError(f"Unsupported teacher_repair_preset: {preset!r}")

    expanded.setdefault("allow_seed_source_backfill", True)
    expanded.setdefault("allow_completing_seed_source_backfill", True)
    expanded.setdefault("teacher_edge_order", "dynamic-seed-confidence")
    expanded.setdefault("teacher_feature_preset", "seed-source-high-confidence")
    expanded["min_component_observations"] = max(
        2, int(expanded.get("min_component_observations", 1))
    )
    expanded.setdefault("max_applied_edits", 2)
    return expanded
