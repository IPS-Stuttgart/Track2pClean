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

    from bayescatrack.experiments import _teacher_rescue_manifest_integration as base
    from bayescatrack.experiments import benchmark_manifest as manifest

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

    base._run_track2p_policy_teacher_adjacent_rows = (
        _run_teacher_rows_with_repair_preset
    )
    manifest._bayescatrack_teacher_repair_preset_integration = True


def _expand_teacher_repair_preset(options: Mapping[str, Any]) -> dict[str, Any]:
    """Return options with high-level repair presets expanded."""

    expanded = dict(options)
    preset = str(expanded.get(TEACHER_REPAIR_PRESET_FIELD, "none")).strip().lower()
    if preset in {"", "none"}:
        return expanded
    defaults = {
        "missing-seed-high-confidence": {
            "allow_source_backfill": False,
            "allow_seed_source_backfill": True,
            "allow_completing_seed_source_backfill": True,
            "teacher_edge_order": "dynamic-seed-confidence",
            "teacher_action_filter": "seed-source-backfill",
            "teacher_feature_preset": "seed-source-high-confidence",
            "min_component_observations": 2,
            "max_applied_edits": 2,
        },
        "missing-seed-cell-confident": {
            "allow_source_backfill": False,
            "allow_seed_source_backfill": True,
            "allow_completing_seed_source_backfill": True,
            "teacher_edge_order": "dynamic-seed-confidence",
            "teacher_action_filter": "seed-source-backfill",
            "teacher_feature_preset": "seed-source-cell-confident",
            "min_component_observations": 2,
            "max_applied_edits": 3,
        },
        "missing-seed-moderate-iou": {
            "allow_source_backfill": False,
            "allow_seed_source_backfill": True,
            "allow_completing_seed_source_backfill": True,
            "teacher_edge_order": "dynamic-seed-confidence",
            "teacher_action_filter": "seed-source-backfill",
            "teacher_feature_preset": "seed-source-moderate-iou",
            "min_component_observations": 2,
            "max_applied_edits": 2,
        },
        "track2p-fn-high-confidence": {
            "teacher_action_filter": "target-extension",
            "teacher_edge_order": "dynamic-confidence",
            "teacher_feature_preset": "track2p-fn-rescue",
            "min_component_observations": 2,
            "max_applied_edits": 3,
        },
        "track2p-fn-moderate-iou-cell-confident": {
            "teacher_action_filter": "target-extension",
            "teacher_edge_order": "dynamic-confidence",
            "teacher_feature_preset": "moderate-iou-cell-confidence",
            "min_component_observations": 2,
            "max_applied_edits": 3,
        },
        "track2p-fn-moderate-iou-cell-confidence": {
            "teacher_action_filter": "target-extension",
            "teacher_edge_order": "dynamic-confidence",
            "teacher_feature_preset": "moderate-iou-cell-confidence",
            "min_component_observations": 2,
            "max_applied_edits": 3,
        },
        "residual-union-cell-confident": {
            "allow_source_backfill": False,
            "allow_seed_source_backfill": True,
            "allow_completing_seed_source_backfill": True,
            "allow_fragment_merges": False,
            "teacher_action_filter": "target-extension-or-seed-source-backfill",
            "teacher_edge_order": "dynamic-seed-confidence",
            "teacher_feature_preset": "residual-fn-cell-confident",
            "min_component_observations": 2,
            "max_applied_edits": 3,
        },
    }
    if preset not in defaults:
        raise ValueError(f"Unsupported teacher_repair_preset: {preset!r}")

    for key, value in defaults[preset].items():
        if key == "min_component_observations":
            expanded[key] = max(int(value), int(expanded.get(key, 1)))
        else:
            expanded.setdefault(key, value)
    return expanded
