"""Manifest integration for capped Track2p teacher-rescue rows.

The base teacher-rescue manifest runner accepts ``max_applied_edits``. This
module keeps compatibility with generated manifests that install the field and
the associated advanced-workbench rows separately.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

EDIT_CAP_FIELD = "max_applied_edits"


def install_teacher_rescue_edit_cap_manifest_integration() -> None:
    """Install max-applied-edit support for teacher-rescue manifests."""

    from bayescatrack.experiments import _teacher_rescue_manifest_integration as base
    from bayescatrack.experiments import benchmark_manifest as manifest

    base.TEACHER_ADJACENT_RESCUE_FIELDS.add(EDIT_CAP_FIELD)
    manifest.RUNNER_SPECIFIC_FIELDS.add(EDIT_CAP_FIELD)
    manifest.RUN_SPEC_FIELDS.add(EDIT_CAP_FIELD)
    manifest.RUNNER_CONFIG_FIELDS.setdefault(
        base.TEACHER_ADJACENT_RESCUE_RUNNER,
        set(manifest.TRACK2P_CONFIG_FIELDS | base.TEACHER_ADJACENT_RESCUE_FIELDS),
    ).add(EDIT_CAP_FIELD)

    manifest._bayescatrack_teacher_rescue_edit_cap_integration = True
    base._install_advanced_workbench_manifest_row()
    _install_advanced_workbench_edit_cap_rows()


def _install_advanced_workbench_edit_cap_rows() -> None:
    from bayescatrack.experiments import advanced_improvement_workbench as workbench

    original = workbench.track2p_result_improvement_manifest
    if getattr(original, "_bayescatrack_teacher_rescue_edit_cap_integration", False):
        return

    def _track2p_result_improvement_manifest_with_edit_caps(
        *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        manifest = original(*args, **kwargs)
        output_root = str(kwargs.get("output_root", "<OUTPUT_ROOT>"))
        _append_teacher_rescue_edit_cap_runs(manifest, output_root=output_root)
        return manifest

    setattr(
        _track2p_result_improvement_manifest_with_edit_caps,
        "_bayescatrack_teacher_rescue_edit_cap_integration",
        True,
    )
    workbench.track2p_result_improvement_manifest = (
        _track2p_result_improvement_manifest_with_edit_caps
    )


def _teacher_rescue_edit_cap_rows(output_root: str) -> tuple[dict[str, Any], ...]:
    base = _teacher_rescue_base_row()
    audit_gate = {
        "teacher_min_registered_iou": 0.10,
        "teacher_max_centroid_distance": 6.0,
        "teacher_min_area_ratio": 0.45,
    }
    local_support = {
        "teacher_min_threshold_margin": 0.0,
        "teacher_min_row_margin": 0.0,
        "teacher_min_column_margin": 0.0,
        "teacher_max_centroid_distance": 6.0,
        "teacher_min_area_ratio": 0.60,
        "teacher_require_hungarian": True,
    }
    high_confidence = {
        "teacher_min_registered_iou": 0.20,
        "teacher_min_threshold_margin": 0.05,
        "teacher_min_row_margin": 0.0,
        "teacher_min_column_margin": 0.0,
        "teacher_max_centroid_distance": 4.0,
        "teacher_min_area_ratio": 0.70,
        "teacher_min_cell_probability": 0.50,
        "teacher_require_hungarian": True,
    }
    return (
        {
            **base,
            "name": "track2p-policy-teacher-adjacent-rescue-dynamic-confidence-max1",
            "max_applied_edits": 1,
            "output": f"{output_root}/track2p_policy_teacher_adjacent_rescue_dynamic_confidence_max1.csv",
        },
        {
            **base,
            "name": "track2p-policy-teacher-adjacent-rescue-dynamic-confidence-max2",
            "max_applied_edits": 2,
            "output": f"{output_root}/track2p_policy_teacher_adjacent_rescue_dynamic_confidence_max2.csv",
        },
        {
            **base,
            **audit_gate,
            "name": "track2p-policy-teacher-adjacent-rescue-feature-gated-dynamic-confidence-max1",
            "max_applied_edits": 1,
            "output": f"{output_root}/track2p_policy_teacher_adjacent_rescue_feature_gated_dynamic_confidence_max1.csv",
        },
        {
            **base,
            **local_support,
            "name": "track2p-policy-teacher-adjacent-rescue-dynamic-confidence-local-support-max2",
            "max_applied_edits": 2,
            "output": f"{output_root}/track2p_policy_teacher_adjacent_rescue_dynamic_confidence_local_support_max2.csv",
        },
        {
            **base,
            **high_confidence,
            "name": "track2p-policy-teacher-adjacent-rescue-dynamic-confidence-high-confidence-seed-source-max2",
            "allow_seed_source_backfill": True,
            "max_applied_edits": 2,
            "output": f"{output_root}/track2p_policy_teacher_adjacent_rescue_dynamic_confidence_high_confidence_seed_source_max2.csv",
        },
    )


def _teacher_rescue_base_row() -> dict[str, Any]:
    return {
        "runner": "track2p-policy-teacher-adjacent-rescue",
        "transform_type": "affine",
        "threshold_method": "min",
        "iou_distance_threshold": 12.0,
        "cell_probability_threshold": 0.5,
        "max_gap": 1,
        "weighted_masks": False,
        "weighted_centroids": False,
        "exclude_overlapping_pixels": False,
        "apply_splits": True,
        "split_risk_threshold": 1.50,
        "split_penalty": 0.25,
        "min_side_observations": 2,
        "min_component_observations": 2,
        "allow_source_backfill": True,
        "allow_fragment_merges": True,
        "allow_completing_rescue": False,
        "allow_teacher_supported_completing_rescue": False,
        "allow_seed_source_backfill": False,
        "allow_completing_seed_source_backfill": False,
        "teacher_edge_order": "dynamic-confidence",
    }


def _append_teacher_rescue_edit_cap_runs(
    manifest: dict[str, Any], *, output_root: str
) -> None:
    runs = manifest.get("runs")
    if not isinstance(runs, list):
        return
    rows_to_insert = list(_teacher_rescue_edit_cap_rows(output_root))
    edit_cap_names = {row["name"] for row in rows_to_insert}
    runs[:] = [
        run
        for run in runs
        if not (isinstance(run, Mapping) and run.get("name") in edit_cap_names)
    ]
    insert_at = _insertion_index_after_any(
        runs,
        (
            "track2p-policy-teacher-adjacent-rescue-dynamic-confidence-seed-source",
            "track2p-policy-component-cleanup",
        ),
    )
    runs[insert_at:insert_at] = rows_to_insert
    for comparison in manifest.get("comparisons", []):
        if isinstance(comparison, dict) and isinstance(comparison.get("inputs"), dict):
            comparison["inputs"] = _insert_mapping_after_any(
                comparison["inputs"],
                after_keys=(
                    "track2p-policy-teacher-adjacent-rescue-dynamic-confidence-seed-source",
                    "track2p-policy-component-cleanup",
                ),
                items=((row["name"], row["name"]) for row in rows_to_insert),
            )


def _insertion_index_after_any(runs: list[Any], run_names: tuple[str, ...]) -> int:
    for run_name in run_names:
        for index, run in enumerate(runs):
            if isinstance(run, Mapping) and run.get("name") == run_name:
                return index + 1
    return len(runs)


def _insert_mapping_after_any(
    mapping: Mapping[str, Any], *, after_keys: tuple[str, ...], items: Any
) -> dict[str, Any]:
    pending = list(items)
    pending_keys = {key for key, _value in pending}
    target_key = next((key for key in after_keys if key in mapping), None)
    result: dict[str, Any] = {}
    inserted = False
    for key, value in mapping.items():
        if key in pending_keys:
            continue
        result[str(key)] = value
        if not inserted and key == target_key:
            result.update(dict(pending))
            inserted = True
    if not inserted:
        result.update(dict(pending))
    return result
