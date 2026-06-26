"""Expose Track2p teacher-adjacent rescue in benchmark manifests.

The teacher-adjacent rescue runner is intentionally narrow and already exposed as a
first-class benchmark CLI.  This integration makes it available to JSON benchmark
manifests as well, so accuracy experiments can compare it reproducibly against
the frozen Track2pPolicy component-cleanup row.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any, cast

TEACHER_ADJACENT_RESCUE_RUNNER = "track2p-policy-teacher-adjacent-rescue"
TEACHER_ADJACENT_RESCUE_ALIASES = {
    TEACHER_ADJACENT_RESCUE_RUNNER,
    "track2p-teacher-adjacent-rescue",
}

TEACHER_ADJACENT_RESCUE_FIELDS = {
    "threshold_method",
    "iou_distance_threshold",
    "apply_splits",
    "threshold_margin_scale",
    "competition_margin_scale",
    "area_ratio_floor",
    "centroid_distance_scale",
    "split_risk_threshold",
    "split_penalty",
    "min_side_observations",
    "threshold_margin_weight",
    "row_margin_weight",
    "column_margin_weight",
    "centroid_distance_weight",
    "area_ratio_weight",
    "allow_completing_rescue",
    "allow_teacher_complete_row_rescue",
    "allow_teacher_supported_completion",
    "allow_teacher_supported_completing_rescue",
    "allow_teacher_confirmed_completing_rescue",
    "allow_completing_source_backfill",
    "allow_completing_fragment_merge",
    "allow_completing_fragment_merges",
    "allow_source_backfill",
    "allow_source_inserts",
    "allow_source_insertions",
    "allow_seed_source_backfill",
    "allow_seed_completing_backfill",
    "allow_seed_completing_rescue",
    "allow_completing_seed_source_backfill",
    "allow_fragment_merges",
    "min_component_observations",
    "max_applied_edits",
    "max_target_extension_edits",
    "max_source_backfill_edits",
    "max_seed_source_backfill_edits",
    "max_fragment_merge_edits",
    "max_completing_rescue_edits",
    "teacher_edge_order",
    "teacher_action_filter",
    "teacher_repair_preset",
    "teacher_feature_preset",
    "target_extension_feature_preset",
    "seed_source_feature_preset",
    "teacher_min_registered_iou",
    "teacher_max_registered_iou",
    "teacher_min_threshold_margin",
    "teacher_min_row_margin",
    "teacher_min_column_margin",
    "teacher_max_centroid_distance",
    "teacher_min_area_ratio",
    "teacher_min_cell_probability",
    "teacher_require_hungarian",
    "teacher_require_hungarian_assignment",
    "teacher_gate_min_registered_iou",
    "teacher_gate_max_registered_iou",
    "teacher_gate_min_threshold_margin",
    "teacher_gate_min_row_margin",
    "teacher_gate_min_column_margin",
    "teacher_gate_max_centroid_distance",
    "teacher_gate_min_area_ratio",
    "teacher_gate_min_cell_probability",
    "teacher_gate_require_hungarian",
}


def _optional_float_option(options: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        if name in options and options[name] not in {None, ""}:
            return float(options[name])
    return None


def _integer_value(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc
    raise ValueError(f"{name} must be an integer")


def _positive_int_option(options: Mapping[str, Any], name: str, *, default: int) -> int:
    value = _integer_value(options.get(name, default), name=name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _optional_nonnegative_int_option(
    options: Mapping[str, Any], *names: str
) -> int | None:
    for name in names:
        if name in options and options[name] not in {None, ""}:
            value = _integer_value(options[name], name=name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative when provided")
            return value
    return None


def install_teacher_rescue_manifest_integration() -> None:
    """Install manifest support for the Track2p teacher-adjacent rescue runner."""

    from bayescatrack.experiments import benchmark_manifest as manifest

    if getattr(manifest, "_bayescatrack_teacher_rescue_manifest_integration", False):
        _install_advanced_workbench_manifest_row()
        if getattr(manifest, "_bayescatrack_teacher_rescue_edit_cap_integration", False):
            from bayescatrack.experiments import (
                _teacher_rescue_edit_cap_manifest_integration as edit_cap,
            )

            edit_cap._install_advanced_workbench_edit_cap_rows()
        return

    manifest.RUNNER_SPECIFIC_FIELDS.update(TEACHER_ADJACENT_RESCUE_FIELDS)
    manifest.RUN_SPEC_FIELDS.update(TEACHER_ADJACENT_RESCUE_FIELDS)
    manifest.RUNNER_CONFIG_FIELDS[TEACHER_ADJACENT_RESCUE_RUNNER] = set(
        manifest.TRACK2P_CONFIG_FIELDS | TEACHER_ADJACENT_RESCUE_FIELDS
    )
    for alias in TEACHER_ADJACENT_RESCUE_ALIASES:
        manifest.RUNNER_ALIASES[alias] = TEACHER_ADJACENT_RESCUE_RUNNER
    manifest.RUNNER_CHOICES = frozenset(manifest.RUNNER_ALIASES)

    original_runner_specific_fields = manifest._runner_specific_fields
    original_runner_kwargs = manifest._runner_kwargs
    original_run_config = manifest._run_config
    original_run_benchmark_rows = manifest._run_benchmark_rows
    original_run_manifest_entry = getattr(manifest, "_run_manifest_entry", None)

    def _runner_specific_fields_with_teacher(runner: str) -> set[str]:
        if runner == TEACHER_ADJACENT_RESCUE_RUNNER:
            return set(TEACHER_ADJACENT_RESCUE_FIELDS)
        return original_runner_specific_fields(runner)

    def _runner_kwargs_with_teacher(
        run_data: Mapping[str, Any], runner: str
    ) -> dict[str, Any]:
        if runner == TEACHER_ADJACENT_RESCUE_RUNNER:
            return {
                key: run_data[key]
                for key in TEACHER_ADJACENT_RESCUE_FIELDS
                if key in run_data
            }
        return original_runner_kwargs(run_data, runner)

    def _run_config_with_teacher(
        runner: str, run_data: Mapping[str, Any], *, base_dir: Any
    ) -> Any:
        if runner == TEACHER_ADJACENT_RESCUE_RUNNER:
            config_defaults = {
                "method": "global-assignment",
                "include_non_cells": False,
                "weighted_masks": False,
                "weighted_centroids": False,
                "exclude_overlapping_pixels": False,
            }
            config_kwargs = manifest._track2p_config_kwargs(
                run_data,
                base_dir=base_dir,
                config_defaults=config_defaults,
                required=("data",),
            )
            return manifest.Track2pBenchmarkConfig(**config_kwargs)
        return original_run_config(runner, run_data, base_dir=base_dir)

    def _run_benchmark_rows_with_teacher(run_spec: Any) -> list[dict[str, Any]]:
        if run_spec.runner == TEACHER_ADJACENT_RESCUE_RUNNER:
            return _run_track2p_policy_teacher_adjacent_rows(
                run_spec.config, dict(run_spec.runner_kwargs or {})
            )
        return original_run_benchmark_rows(run_spec)

    def _run_manifest_entry_with_teacher(run_spec: Any) -> list[dict[str, Any]]:
        if run_spec.runner == TEACHER_ADJACENT_RESCUE_RUNNER:
            return _run_track2p_policy_teacher_adjacent_rows(
                run_spec.config, dict(run_spec.runner_kwargs or {})
            )
        if original_run_manifest_entry is None:  # pragma: no cover - legacy guard
            return original_run_benchmark_rows(run_spec)
        return original_run_manifest_entry(run_spec)

    manifest._runner_specific_fields = _runner_specific_fields_with_teacher
    manifest._runner_kwargs = _runner_kwargs_with_teacher
    manifest._run_config = _run_config_with_teacher
    manifest._run_benchmark_rows = _run_benchmark_rows_with_teacher
    if original_run_manifest_entry is not None:
        manifest._run_manifest_entry = _run_manifest_entry_with_teacher
    manifest._bayescatrack_teacher_rescue_manifest_integration = True

    _install_advanced_workbench_manifest_row()


def _run_track2p_policy_teacher_adjacent_rows(
    config: Any, options: Mapping[str, Any]
) -> list[dict[str, Any]]:
    from bayescatrack.experiments import benchmark_manifest as manifest
    from bayescatrack.experiments.track2p_policy_benchmark import (
        TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
        TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    )
    from bayescatrack.experiments.track2p_policy_component_audit import (
        ComponentCleanupConfig,
    )
    from bayescatrack.experiments.track2p_policy_teacher_adjacent_rescue import (
        TeacherActionFilter,
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
        min_registered_iou=_optional_float_option(
            options, "teacher_min_registered_iou", "teacher_gate_min_registered_iou"
        ),
        max_registered_iou=_optional_float_option(
            options,
            "teacher_max_registered_iou",
            "teacher_gate_max_registered_iou",
        ),
        min_threshold_margin=_optional_float_option(
            options, "teacher_min_threshold_margin", "teacher_gate_min_threshold_margin"
        ),
        min_row_margin=_optional_float_option(
            options, "teacher_min_row_margin", "teacher_gate_min_row_margin"
        ),
        min_column_margin=_optional_float_option(
            options, "teacher_min_column_margin", "teacher_gate_min_column_margin"
        ),
        max_centroid_distance=_optional_float_option(
            options,
            "teacher_max_centroid_distance",
            "teacher_gate_max_centroid_distance",
        ),
        min_area_ratio=_optional_float_option(
            options, "teacher_min_area_ratio", "teacher_gate_min_area_ratio"
        ),
        min_cell_probability=_optional_float_option(
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
        allow_teacher_complete_row_rescue=manifest._bool_option(
            options, "allow_teacher_complete_row_rescue", default=False
        ),
        allow_teacher_supported_completion=manifest._bool_option(
            options, "allow_teacher_supported_completion", default=False
        ),
        allow_teacher_supported_completing_rescue=manifest._bool_option(
            options, "allow_teacher_supported_completing_rescue", default=False
        ),
        allow_teacher_confirmed_completing_rescue=manifest._bool_option(
            options, "allow_teacher_confirmed_completing_rescue", default=False
        ),
        allow_completing_source_backfill=manifest._bool_option(
            options, "allow_completing_source_backfill", default=False
        ),
        allow_completing_fragment_merge=manifest._bool_option(
            options, "allow_completing_fragment_merge", default=False
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
        allow_seed_completing_backfill=manifest._bool_option(
            options, "allow_seed_completing_backfill", default=False
        ),
        allow_seed_completing_rescue=manifest._bool_option(
            options, "allow_seed_completing_rescue", default=False
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
        teacher_action_filter=cast(
            TeacherActionFilter, str(options.get("teacher_action_filter", "all"))
        ),
        teacher_repair_preset=str(options.get("teacher_repair_preset", "none")),
        teacher_feature_preset=str(options.get("teacher_feature_preset", "none")),
        target_extension_feature_preset=str(
            options.get("target_extension_feature_preset", "none")
        ),
        seed_source_feature_preset=str(
            options.get("seed_source_feature_preset", "none")
        ),
        min_component_observations=_positive_int_option(
            options, "min_component_observations", default=1
        ),
        max_applied_edits=_optional_nonnegative_int_option(
            options, "max_applied_edits"
        ),
        max_target_extension_edits=_optional_nonnegative_int_option(
            options, "max_target_extension_edits"
        ),
        max_source_backfill_edits=_optional_nonnegative_int_option(
            options, "max_source_backfill_edits"
        ),
        max_seed_source_backfill_edits=_optional_nonnegative_int_option(
            options, "max_seed_source_backfill_edits"
        ),
        max_fragment_merge_edits=_optional_nonnegative_int_option(
            options, "max_fragment_merge_edits"
        ),
        max_completing_rescue_edits=_optional_nonnegative_int_option(
            options, "max_completing_rescue_edits"
        ),
        teacher_feature_gate=teacher_feature_gate,
    )
    return [result.to_dict() for result in output.results]


def _install_advanced_workbench_manifest_row() -> None:
    from bayescatrack.experiments import advanced_improvement_workbench as workbench

    original = workbench.track2p_result_improvement_manifest
    if getattr(original, "_bayescatrack_teacher_rescue_manifest_integration", False):
        return

    def _track2p_result_improvement_manifest_with_teacher(
        *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        manifest = original(*args, **kwargs)
        output_root = str(kwargs.get("output_root", "<OUTPUT_ROOT>"))
        _append_teacher_rescue_runs(manifest, output_root=output_root)
        return manifest

    setattr(
        _track2p_result_improvement_manifest_with_teacher,
        "_bayescatrack_teacher_rescue_manifest_integration",
        True,
    )
    workbench.track2p_result_improvement_manifest = (
        _track2p_result_improvement_manifest_with_teacher
    )


def _teacher_rescue_manifest_rows(output_root: str) -> tuple[dict[str, Any], ...]:
    base = {
        "runner": TEACHER_ADJACENT_RESCUE_RUNNER,
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
        "allow_source_backfill": True,
        "allow_fragment_merges": True,
        "min_component_observations": 1,
    }
    variants: tuple[
        tuple[str, bool, bool, bool, bool, str | None, Mapping[str, Any], str], ...
    ] = (
        (
            "track2p-policy-teacher-adjacent-rescue",
            False,
            False,
            False,
            False,
            None,
            {},
            "track2p_policy_teacher_adjacent_rescue.csv",
        ),
        (
            "track2p-policy-teacher-adjacent-rescue-dynamic-structural",
            False,
            False,
            False,
            False,
            "dynamic-structural",
            {},
            "track2p_policy_teacher_adjacent_rescue_dynamic_structural.csv",
        ),
        (
            "track2p-policy-teacher-adjacent-rescue-confidence",
            False,
            False,
            False,
            False,
            "confidence",
            {},
            "track2p_policy_teacher_adjacent_rescue_confidence.csv",
        ),
        (
            "track2p-policy-teacher-adjacent-rescue-feature-gated-dynamic-confidence",
            False,
            False,
            False,
            False,
            "dynamic-confidence",
            {},
            "track2p_policy_teacher_adjacent_rescue_feature_gated_dynamic_confidence.csv",
        ),
        (
            "track2p-policy-teacher-adjacent-rescue-feature-gated-dynamic-confidence-max1",
            False,
            False,
            False,
            False,
            "dynamic-confidence",
            {},
            "track2p_policy_teacher_adjacent_rescue_feature_gated_dynamic_confidence_max1.csv",
        ),
        (
            "track2p-policy-teacher-adjacent-rescue-high-confidence-dynamic-confidence-max1",
            False,
            False,
            False,
            False,
            "dynamic-confidence",
            {"teacher_feature_preset": "high-confidence", "max_applied_edits": 1},
            "track2p_policy_teacher_adjacent_rescue_high_confidence_dynamic_confidence_max1.csv",
        ),
        (
            "track2p-policy-teacher-adjacent-rescue-high-confidence-dynamic-confidence-seed-source-max1",
            False,
            True,
            True,
            False,
            "dynamic-confidence",
            {"teacher_feature_preset": "high-confidence", "max_applied_edits": 1},
            "track2p_policy_teacher_adjacent_rescue_high_confidence_dynamic_confidence_seed_source_max1.csv",
        ),
        (
            "track2p-policy-teacher-adjacent-rescue-dynamic-confidence-max1",
            False,
            False,
            False,
            False,
            "dynamic-confidence",
            {"max_applied_edits": 1},
            "track2p_policy_teacher_adjacent_rescue_dynamic_confidence_max1.csv",
        ),
        (
            "track2p-policy-teacher-adjacent-rescue-dynamic-confidence-max2",
            False,
            False,
            False,
            False,
            "dynamic-confidence",
            {"max_applied_edits": 2},
            "track2p_policy_teacher_adjacent_rescue_dynamic_confidence_max2.csv",
        ),
        (
            "track2p-policy-teacher-adjacent-rescue-dynamic-confidence-seed-source",
            False,
            True,
            True,
            False,
            "dynamic-confidence",
            {},
            "track2p_policy_teacher_adjacent_rescue_dynamic_confidence_seed_source.csv",
        ),
        (
            "track2p-policy-teacher-adjacent-rescue-dynamic-confidence-first-edit-seed-source",
            False,
            True,
            True,
            False,
            "dynamic-confidence",
            {"max_applied_edits": 1},
            "track2p_policy_teacher_adjacent_rescue_dynamic_confidence_first_edit_seed_source.csv",
        ),
        (
            (
                "track2p-policy-teacher-adjacent-rescue-"
                "dynamic-seed-confidence-first-edit-seed-source"
            ),
            False,
            True,
            True,
            False,
            "dynamic-seed-confidence",
            {"max_applied_edits": 1},
            (
                "track2p_policy_teacher_adjacent_rescue_"
                "dynamic_seed_confidence_first_edit_seed_source.csv"
            ),
        ),
        (
            "track2p-policy-teacher-adjacent-rescue-dynamic-confidence-seed-source-cellgate",
            False,
            True,
            True,
            False,
            "dynamic-confidence",
            {},
            "track2p_policy_teacher_adjacent_rescue_dynamic_confidence_seed_source_cellgate.csv",
        ),
        (
            "track2p-policy-teacher-adjacent-rescue-dynamic-confidence-seed-source-cell-high-confidence-max2",
            False,
            True,
            True,
            False,
            "dynamic-confidence",
            {"teacher_feature_preset": "cell-high-confidence", "max_applied_edits": 2},
            "track2p_policy_teacher_adjacent_rescue_dynamic_confidence_seed_source_cell_high_confidence_max2.csv",
        ),
        (
            "track2p-policy-teacher-adjacent-rescue-residual-fn-dynamic-confidence",
            False,
            True,
            True,
            False,
            "dynamic-confidence",
            {
                "teacher_feature_preset": "residual-fn",
                "min_component_observations": 2,
            },
            "track2p_policy_teacher_adjacent_rescue_residual_fn_dynamic_confidence.csv",
        ),
        (
            "track2p-policy-teacher-adjacent-rescue-residual-fn-dynamic-confidence-max2",
            False,
            True,
            True,
            False,
            "dynamic-confidence",
            {
                "teacher_feature_preset": "residual-fn",
                "min_component_observations": 2,
                "max_applied_edits": 2,
            },
            (
                "track2p_policy_teacher_adjacent_rescue_residual_fn_"
                "dynamic_confidence_max2.csv"
            ),
        ),
        (
            (
                "track2p-policy-teacher-adjacent-rescue-"
                "dynamic-seed-confidence-seed-source-max2"
            ),
            False,
            True,
            True,
            False,
            "dynamic-seed-confidence",
            {
                "max_applied_edits": 2,
                "teacher_feature_preset": "high-confidence",
                "teacher_min_cell_probability": 0.60,
            },
            (
                "track2p_policy_teacher_adjacent_rescue_"
                "dynamic_seed_confidence_seed_source_max2.csv"
            ),
        ),
        (
            (
                "track2p-policy-teacher-adjacent-rescue-"
                "moderate-iou-cell-target-extension-max3"
            ),
            False,
            False,
            False,
            False,
            "dynamic-confidence",
            {
                "teacher_action_filter": "target-extension",
                "teacher_repair_preset": "track2p-fn-moderate-iou-cell-confidence",
                "teacher_feature_preset": "moderate-iou-cell-confidence",
                "min_component_observations": 2,
                "max_applied_edits": 3,
            },
            (
                "track2p_policy_teacher_adjacent_rescue_"
                "moderate_iou_cell_target_extension_max3.csv"
            ),
        ),
        (
            ("track2p-policy-teacher-adjacent-rescue-" "track2p-fn-target-extension"),
            False,
            False,
            False,
            False,
            "dynamic-confidence",
            {
                "allow_source_backfill": False,
                "allow_fragment_merges": False,
                "teacher_action_filter": "target-extension",
                "teacher_repair_preset": "track2p-fn-high-confidence",
                "teacher_feature_preset": "track2p-fn-rescue",
                "min_component_observations": 2,
                "max_applied_edits": 3,
            },
            (
                "track2p_policy_teacher_adjacent_rescue_"
                "track2p_fn_target_extension.csv"
            ),
        ),
        (
            ("track2p-policy-teacher-adjacent-rescue-" "missing-seed-high-confidence"),
            False,
            True,
            True,
            False,
            "dynamic-seed-confidence",
            {
                "allow_source_backfill": False,
                "teacher_repair_preset": "missing-seed-high-confidence",
                "teacher_feature_preset": "seed-source-high-confidence",
                "min_component_observations": 2,
                "max_applied_edits": 2,
            },
            "track2p_policy_teacher_adjacent_rescue_missing_seed_high_confidence.csv",
        ),
        (
            (
                "track2p-policy-teacher-adjacent-rescue-"
                "complete-row-action-specific-max1"
            ),
            False,
            True,
            True,
            False,
            "dynamic-seed-cell-confidence",
            {
                # Residual audits show complete-track FNs are high-leverage, but
                # previous gap/insert candidates often accepted safe edits that
                # did not move official rows.  Keep this manifest candidate
                # complete-row-only and spend at most one teacher edit so it can
                # test the missing-seed / completing-rescue hypothesis without
                # opening broad Track2p-teacher propagation.
                "allow_teacher_complete_row_rescue": True,
                "allow_source_backfill": False,
                "allow_fragment_merges": False,
                "teacher_action_filter": "completing-rescue",
                "teacher_repair_preset": "complete-row-rescue-action-specific",
                "teacher_feature_preset": "none",
                "target_extension_feature_preset": "moderate-iou-cell-confidence",
                "seed_source_feature_preset": "seed-source-cell-confident",
                "min_component_observations": 2,
                "max_applied_edits": 1,
            },
            "track2p_policy_teacher_adjacent_rescue_complete_row_action_specific_max1.csv",
        ),
        (
            "track2p-policy-teacher-adjacent-rescue-seed-source",
            False,
            True,
            False,
            False,
            None,
            {},
            "track2p_policy_teacher_adjacent_rescue_seed_source.csv",
        ),
        (
            "track2p-policy-teacher-adjacent-rescue-supported",
            False,
            False,
            False,
            False,
            None,
            {},
            "track2p_policy_teacher_adjacent_rescue_supported.csv",
        ),
        (
            "track2p-policy-teacher-adjacent-rescue-teacher-completing",
            False,
            False,
            False,
            True,
            None,
            {},
            "track2p_policy_teacher_adjacent_rescue_teacher_completing.csv",
        ),
        (
            "track2p-policy-teacher-adjacent-rescue-teacher-completing-seed-source",
            False,
            True,
            True,
            True,
            None,
            {},
            "track2p_policy_teacher_adjacent_rescue_teacher_completing_seed_source.csv",
        ),
        (
            "track2p-policy-teacher-adjacent-rescue-completing",
            True,
            False,
            False,
            False,
            None,
            {},
            "track2p_policy_teacher_adjacent_rescue_completing.csv",
        ),
        (
            "track2p-policy-teacher-adjacent-rescue-completing-seed-source",
            False,
            True,
            True,
            False,
            None,
            {},
            "track2p_policy_teacher_adjacent_rescue_completing_seed_source.csv",
        ),
    )
    return tuple(
        {
            **base,
            "name": name,
            "allow_completing_rescue": allow_completing,
            "allow_teacher_supported_completing_rescue": (
                allow_teacher_supported_completing
            ),
            "allow_completing_fragment_merges": False,
            "allow_seed_source_backfill": allow_seed_source,
            "allow_completing_seed_source_backfill": allow_completing_seed_source,
            **(
                {"min_component_observations": 2} if name.endswith("-supported") else {}
            ),
            **(
                _feature_gated_teacher_options(
                    max_applied_edits=1 if name.endswith("-max1") else None
                )
                if "-feature-gated-" in name
                else {}
            ),
            **(
                {"teacher_min_cell_probability": 0.60}
                if name.endswith("-cellgate")
                else {}
            ),
            **(
                {"teacher_edge_order": teacher_edge_order} if teacher_edge_order else {}
            ),
            **dict(extra_options),
            "output": f"{output_root}/{filename}",
        }
        for (
            name,
            allow_completing,
            allow_seed_source,
            allow_completing_seed_source,
            allow_teacher_supported_completing,
            teacher_edge_order,
            extra_options,
            filename,
        ) in variants
    )


def _feature_gated_teacher_options(
    *, max_applied_edits: int | None = None
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "teacher_min_registered_iou": 0.10,
        "teacher_max_centroid_distance": 6.0,
        "teacher_min_area_ratio": 0.45,
        "min_component_observations": 2,
    }
    if max_applied_edits is not None:
        options["max_applied_edits"] = int(max_applied_edits)
    return options


def _append_teacher_rescue_runs(manifest: dict[str, Any], *, output_root: str) -> None:
    runs = manifest.get("runs")
    if not isinstance(runs, list):
        return
    rows_to_insert = list(_teacher_rescue_manifest_rows(output_root))
    teacher_names = {row["name"] for row in rows_to_insert}
    runs[:] = [
        run
        for run in runs
        if not (isinstance(run, Mapping) and run.get("name") in teacher_names)
    ]
    insert_at = _insertion_index_after(runs, "track2p-policy-component-cleanup")
    runs[insert_at:insert_at] = rows_to_insert
    for comparison in manifest.get("comparisons", []):
        if isinstance(comparison, dict) and isinstance(comparison.get("inputs"), dict):
            comparison["inputs"] = _insert_mapping_after_many(
                comparison["inputs"],
                after_key="track2p-policy-component-cleanup",
                items=((row["name"], row["name"]) for row in rows_to_insert),
            )


def _insertion_index_after(runs: list[Any], run_name: str) -> int:
    for index, run in enumerate(runs):
        if isinstance(run, Mapping) and run.get("name") == run_name:
            return index + 1
    return len(runs)


def _insert_mapping_after_many(
    values: Mapping[str, Any], *, after_key: str, items: Any
) -> dict[str, Any]:
    materialized_items = tuple(items)
    if not materialized_items:
        return dict(values)
    inserted_keys = {key for key, _value in materialized_items}
    out: dict[str, Any] = {}
    inserted = False
    for current_key, current_value in values.items():
        if current_key in inserted_keys:
            continue
        out[current_key] = current_value
        if current_key == after_key:
            for key, value in materialized_items:
                out[key] = value
            inserted = True
    if not inserted:
        for key, value in materialized_items:
            out[key] = value
    return out
