"""Expose Track2p teacher-adjacent rescue in benchmark manifests.

The teacher-adjacent rescue runner is intentionally narrow and already exposed as a
first-class benchmark CLI.  This integration makes it available to JSON benchmark
manifests as well, so accuracy experiments can compare it reproducibly against
the frozen Track2pPolicy component-cleanup row.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

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
    "allow_completing_fragment_merges",
    "allow_source_backfill",
    "allow_source_inserts",
    "allow_seed_source_backfill",
    "allow_completing_seed_source_backfill",
    "allow_fragment_merges",
}


def install_teacher_rescue_manifest_integration() -> None:
    """Install manifest support for the Track2p teacher-adjacent rescue runner."""

    from bayescatrack.experiments import benchmark_manifest as manifest

    if getattr(manifest, "_bayescatrack_teacher_rescue_manifest_integration", False):
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
        allow_completing_fragment_merges=manifest._bool_option(
            options, "allow_completing_fragment_merges", default=False
        ),
        allow_source_backfill=manifest._bool_option(
            options, "allow_source_backfill", default=True
        ),
        allow_source_inserts=allow_source_inserts,
        allow_seed_source_backfill=manifest._bool_option(
            options, "allow_seed_source_backfill", default=False
        ),
        allow_completing_seed_source_backfill=manifest._bool_option(
            options, "allow_completing_seed_source_backfill", default=False
        ),
        allow_fragment_merges=manifest._bool_option(
            options, "allow_fragment_merges", default=True
        ),
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
    }
    variants: tuple[tuple[str, bool, bool, bool, str], ...] = (
        (
            "track2p-policy-teacher-adjacent-rescue",
            False,
            False,
            False,
            "track2p_policy_teacher_adjacent_rescue.csv",
        ),
        (
            "track2p-policy-teacher-adjacent-rescue-seed-source",
            False,
            True,
            False,
            "track2p_policy_teacher_adjacent_rescue_seed_source.csv",
        ),
        (
            "track2p-policy-teacher-adjacent-rescue-completing",
            True,
            False,
            False,
            "track2p_policy_teacher_adjacent_rescue_completing.csv",
        ),
        (
            "track2p-policy-teacher-adjacent-rescue-completing-seed-source",
            False,
            True,
            True,
            "track2p_policy_teacher_adjacent_rescue_completing_seed_source.csv",
        ),
    )
    return tuple(
        {
            **base,
            "name": name,
            "allow_completing_rescue": allow_completing,
            "allow_completing_fragment_merges": False,
            "allow_seed_source_backfill": allow_seed_source,
            "allow_completing_seed_source_backfill": allow_completing_seed_source,
            "output": f"{output_root}/{filename}",
        }
        for (
            name,
            allow_completing,
            allow_seed_source,
            allow_completing_seed_source,
            filename,
        ) in variants
    )


def _append_teacher_rescue_runs(manifest: dict[str, Any], *, output_root: str) -> None:
    runs = manifest.get("runs")
    if not isinstance(runs, list):
        return
    existing_names = {
        run.get("name") for run in runs if isinstance(run, Mapping) and "name" in run
    }
    rows_to_insert = [
        row
        for row in _teacher_rescue_manifest_rows(output_root)
        if row["name"] not in existing_names
    ]
    if rows_to_insert:
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
    materialized_items = tuple(
        (key, value) for key, value in items if key not in values
    )
    if not materialized_items:
        return dict(values)
    out: dict[str, Any] = {}
    inserted = False
    for current_key, current_value in values.items():
        out[current_key] = current_value
        if current_key == after_key:
            for key, value in materialized_items:
                out[key] = value
            inserted = True
    if not inserted:
        for key, value in materialized_items:
            out[key] = value
    return out
