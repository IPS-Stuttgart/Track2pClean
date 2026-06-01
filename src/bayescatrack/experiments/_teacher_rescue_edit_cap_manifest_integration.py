"""Manifest integration for capped Track2p teacher-rescue rows.

The residual-repair runner already exposes ``--max-applied-edits`` on the
first-class Track2p teacher-adjacent rescue CLI.  This module makes the same
knob available to JSON manifests and to the generated Track2p improvement
manifest so high-confidence first-edit regimes can be evaluated reproducibly.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

EDIT_CAP_FIELD = "max_applied_edits"


def install_teacher_rescue_edit_cap_manifest_integration() -> None:
    """Install max-applied-edit support for teacher-rescue manifests."""

    from bayescatrack.experiments import benchmark_manifest as manifest
    from bayescatrack.experiments import _teacher_rescue_manifest_integration as base

    if getattr(manifest, "_bayescatrack_teacher_rescue_edit_cap_integration", False):
        return

    base.TEACHER_ADJACENT_RESCUE_FIELDS.add(EDIT_CAP_FIELD)
    manifest.RUNNER_SPECIFIC_FIELDS.add(EDIT_CAP_FIELD)
    manifest.RUN_SPEC_FIELDS.add(EDIT_CAP_FIELD)
    manifest.RUNNER_CONFIG_FIELDS.setdefault(
        base.TEACHER_ADJACENT_RESCUE_RUNNER,
        set(manifest.TRACK2P_CONFIG_FIELDS | base.TEACHER_ADJACENT_RESCUE_FIELDS),
    ).add(EDIT_CAP_FIELD)

    original_runner = base._run_track2p_policy_teacher_adjacent_rows

    def _run_track2p_policy_teacher_adjacent_rows_with_cap(
        config: Any, options: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        if EDIT_CAP_FIELD not in options:
            return original_runner(config, options)
        return _run_track2p_policy_teacher_adjacent_rows(config, options)

    base._run_track2p_policy_teacher_adjacent_rows = (
        _run_track2p_policy_teacher_adjacent_rows_with_cap
    )
    manifest._bayescatrack_teacher_rescue_edit_cap_integration = True
    _install_advanced_workbench_edit_cap_rows()


def _run_track2p_policy_teacher_adjacent_rows(
    config: Any, options: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Run teacher rescue with manifest-provided ``max_applied_edits``."""

    from bayescatrack.experiments import benchmark_manifest as manifest
    from bayescatrack.experiments import _teacher_rescue_manifest_integration as base
    from bayescatrack.experiments.track2p_policy_benchmark import (
        TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
        TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    )
    from bayescatrack.experiments.track2p_policy_component_audit import (
        ComponentCleanupConfig,
    )
    from bayescatrack.experiments.track2p_policy_teacher_adjacent_rescue import (
        TeacherEdgeFeatureGate,
        TeacherEdgeOrder,
        run_track2p_policy_teacher_adjacent_rescue,
    )

    cleanup_defaults = ComponentCleanupConfig()
    cleanup_config = ComponentCleanupConfig(
        threshold_margin_scale=manifest._float_option(
            options,
            "threshold_margin_scale",
            default=cleanup_defaults.threshold_margin_scale,
        ),
        competition_margin_scale=manifest._float_option(
            options,
            "competition_margin_scale",
            default=cleanup_defaults.competition_margin_scale,
        ),
        area_ratio_floor=manifest._float_option(
            options,
            "area_ratio_floor",
            default=cleanup_defaults.area_ratio_floor,
        ),
        centroid_distance_scale=manifest._float_option(
            options,
            "centroid_distance_scale",
            default=cleanup_defaults.centroid_distance_scale,
        ),
        split_risk_threshold=manifest._float_option(
            options,
            "split_risk_threshold",
            default=cleanup_defaults.split_risk_threshold,
        ),
        split_penalty=manifest._float_option(
            options,
            "split_penalty",
            default=cleanup_defaults.split_penalty,
        ),
        min_side_observations=int(
            options.get("min_side_observations", cleanup_defaults.min_side_observations)
        ),
        threshold_margin_weight=manifest._float_option(
            options,
            "threshold_margin_weight",
            default=cleanup_defaults.threshold_margin_weight,
        ),
        row_margin_weight=manifest._float_option(
            options,
            "row_margin_weight",
            default=cleanup_defaults.row_margin_weight,
        ),
        column_margin_weight=manifest._float_option(
            options,
            "column_margin_weight",
            default=cleanup_defaults.column_margin_weight,
        ),
        centroid_distance_weight=manifest._float_option(
            options,
            "centroid_distance_weight",
            default=cleanup_defaults.centroid_distance_weight,
        ),
        area_ratio_weight=manifest._float_option(
            options,
            "area_ratio_weight",
            default=cleanup_defaults.area_ratio_weight,
        ),
    )
    allow_source_inserts = None
    if "allow_source_inserts" in options:
        allow_source_inserts = manifest._bool_option(
            options, "allow_source_inserts", default=True
        )
    allow_source_insertions = None
    if "allow_source_insertions" in options:
        allow_source_insertions = manifest._bool_option(
            options, "allow_source_insertions", default=True
        )
    require_hungarian = False
    for require_key in (
        "teacher_require_hungarian",
        "teacher_require_hungarian_assignment",
        "teacher_gate_require_hungarian",
    ):
        if require_key in options:
            require_hungarian = manifest._bool_option(
                options, require_key, default=False
            )
            break
    teacher_feature_gate = TeacherEdgeFeatureGate(
        min_registered_iou=base._optional_float_option(
            options, "teacher_min_registered_iou", "teacher_gate_min_registered_iou"
        ),
        min_threshold_margin=base._optional_float_option(
            options, "teacher_min_threshold_margin", "teacher_gate_min_threshold_margin"
        ),
        min_row_margin=base._optional_float_option(
            options, "teacher_min_row_margin", "teacher_gate_min_row_margin"
        ),
        min_column_margin=base._optional_float_option(
            options, "teacher_min_column_margin", "teacher_gate_min_column_margin"
        ),
        max_centroid_distance=base._optional_float_option(
            options,
            "teacher_max_centroid_distance",
            "teacher_gate_max_centroid_distance",
        ),
        min_area_ratio=base._optional_float_option(
            options, "teacher_min_area_ratio", "teacher_gate_min_area_ratio"
        ),
        min_cell_probability=base._optional_float_option(
            options, "teacher_min_cell_probability", "teacher_gate_min_cell_probability"
        ),
        require_hungarian=require_hungarian,
    )
    if not teacher_feature_gate.enabled:
        teacher_feature_gate = None

    output = run_track2p_policy_teacher_adjacent_rescue(
        config,
        threshold_method=manifest._policy_threshold_method(
            options.get("threshold_method", TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD)
        ),
        iou_distance_threshold=manifest._float_option(
            options,
            "iou_distance_threshold",
            default=TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
        ),
        transform_type=config.transform_type,
        cell_probability_threshold=config.cell_probability_threshold,
        cleanup_config=cleanup_config,
        allow_completing_rescue=manifest._bool_option(
            options, "allow_completing_rescue", default=False
        ),
        allow_teacher_supported_completing_rescue=manifest._bool_option(
            options, "allow_teacher_supported_completing_rescue", default=False
        ),
        allow_completing_fragment_merges=manifest._bool_option(
            options, "allow_completing_fragment_merges", default=False
        ),
        allow_source_backfill=manifest._bool_option(
            options, "allow_source_backfill", default=True
        ),
        allow_source_inserts=allow_source_inserts,
        allow_source_insertions=allow_source_insertions,
        allow_seed_source_backfill=manifest._bool_option(
            options, "allow_seed_source_backfill", default=False
        ),
        allow_completing_seed_source_backfill=manifest._bool_option(
            options, "allow_completing_seed_source_backfill", default=False
        ),
        allow_fragment_merges=manifest._bool_option(
            options, "allow_fragment_merges", default=True
        ),
        teacher_edge_order=cast(
            TeacherEdgeOrder, str(options.get("teacher_edge_order", "structural"))
        ),
        min_component_observations=int(options.get("min_component_observations", 1)),
        max_applied_edits=int(options[EDIT_CAP_FIELD]),
        teacher_feature_gate=teacher_feature_gate,
    )
    return [result.to_dict() for result in output.results]


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
    target_key = next((key for key in after_keys if key in mapping), None)
    result: dict[str, Any] = {}
    inserted = False
    for key, value in mapping.items():
        result[str(key)] = value
        if not inserted and key == target_key:
            result.update(dict(pending))
            inserted = True
    if not inserted:
        result.update(dict(pending))
    return result
