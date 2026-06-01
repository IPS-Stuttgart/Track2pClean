"""Run reproducible Track2p benchmark suites from a JSON manifest."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal, cast

from bayescatrack.experiments.benchmark_comparison import (
    ComparisonInput,
    aggregate_rows,
    load_labeled_rows,
    write_comparison,
)
from bayescatrack.experiments.track2p_benchmark import (
    OutputFormat,
    Track2pBenchmarkConfig,
    run_track2p_benchmark,
    write_results,
)

ManifestObject = Mapping[str, Any]
TRACK2P_CONFIG_FIELDS = {field.name for field in fields(Track2pBenchmarkConfig)}
DEFAULT_RUNNER = "track2p"
TRACK2P_POLICY_RUNNER = "track2p-policy"
TRACK2P_POLICY_DP_RUNNER = "track2p-policy-dp"
TRACK2P_POLICY_PRUNED_RUNNER = "track2p-policy-pruned"
TRACK2P_POLICY_COMPONENT_RUNNER = "track2p-policy-component-audit"
TRACK2P_POLICY_COHERENCE_SUFFIX_RUNNER = "track2p-policy-coherence-suffix-stitch"
TRACK2P_POLICY_TEACHER_ADJACENT_RESCUE_RUNNER = "track2p-policy-teacher-adjacent-rescue"
BenchmarkRunner = Literal[
    "track2p",
    "track2p-policy",
    "track2p-policy-dp",
    "track2p-policy-pruned",
    "track2p-policy-component-audit",
    "track2p-policy-coherence-suffix-stitch",
    "track2p-policy-teacher-adjacent-rescue",
    "track2p-loso-calibration",
    "track2p-monotone-loso",
    "track2p-solver-prior-loso",
    "registration-qa",
]
RUN_METADATA_FIELDS = {"name", "runner", "output", "format"}
TRACK2P_POLICY_FIELDS = {
    "threshold_method",
    "iou_distance_threshold",
}
TRACK2P_POLICY_DP_FIELDS = TRACK2P_POLICY_FIELDS | {
    "row_top_k",
    "rescue_min_iou",
    "threshold_rescue_margin",
    "accepted_bonus",
    "rescue_penalty",
    "gap_penalty",
    "threshold_margin_weight",
    "beam_width",
    "path_candidates_per_seed",
    "path_selection_beam_width",
}
TRACK2P_POLICY_PRUNED_FIELDS = TRACK2P_POLICY_FIELDS | {
    "prune_threshold_margin",
    "prune_competition_margin",
    "prune_min_area_ratio",
    "prune_centroid_distance",
}
TRACK2P_POLICY_COMPONENT_FIELDS = TRACK2P_POLICY_FIELDS | {
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
}
TRACK2P_POLICY_COHERENCE_SUFFIX_FIELDS = (
    TRACK2P_POLICY_COMPONENT_FIELDS - {"apply_splits"}
) | {
    "suffix_path_length",
    "min_cell_probability",
    "min_area_ratio",
    "max_centroid_distance",
    "min_shifted_iou",
    "min_motion_consistency",
    "min_shape_consistency",
    "max_stitches_per_subject",
    "edge_top_k",
    "path_beam_width",
}
TRACK2P_POLICY_TEACHER_ADJACENT_RESCUE_FIELDS = (
    TRACK2P_POLICY_COMPONENT_FIELDS - {"apply_splits"}
) | {
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
    "teacher_edge_order",
    "teacher_feature_preset",
    "teacher_min_registered_iou",
    "teacher_min_threshold_margin",
    "teacher_min_row_margin",
    "teacher_min_column_margin",
    "teacher_max_centroid_distance",
    "teacher_min_area_ratio",
    "teacher_min_cell_probability",
    "teacher_require_hungarian",
    "teacher_require_hungarian_assignment",
    "teacher_gate_min_registered_iou",
    "teacher_gate_min_threshold_margin",
    "teacher_gate_min_row_margin",
    "teacher_gate_min_column_margin",
    "teacher_gate_max_centroid_distance",
    "teacher_gate_min_area_ratio",
    "teacher_gate_min_cell_probability",
    "teacher_gate_require_hungarian",
}
CONFIGURABLE_LOSO_FIELDS = {
    "feature_names",
    "sample_weight_strategy",
    "calibration_model",
    "calibration_model_kwargs",
    "calibration_model_kwargs_json",
    "hard_negative_options",
    "hard_negative_ratio",
    "hard_negative_top_k",
    "hard_negative_column_candidates",
    "hard_negative_features",
}
MONOTONE_LOSO_FIELDS = {
    "feature_names",
    "monotone_options",
    "monotone_ranker_kwargs",
    "monotone_ranker_kwargs_json",
}
SOLVER_PRIOR_FIELDS = {
    "start_costs",
    "end_costs",
    "gap_penalties",
    "cost_thresholds",
    "objective",
}
REGISTRATION_QA_CONFIG_FIELDS = {
    "data",
    "reference",
    "reference_kind",
    "allow_track2p_as_reference_for_smoke_test",
    "curated_only",
    "plane_name",
    "input_format",
    "max_gap",
    "transform_type",
    "fov_affine_mask_warp_mode",
    "cost",
    "cost_threshold",
    "include_behavior",
    "include_non_cells",
    "cell_probability_threshold",
    "weighted_masks",
    "exclude_overlapping_pixels",
    "order",
    "weighted_centroids",
    "velocity_variance",
    "regularization",
    "pairwise_cost_kwargs",
    "progress",
}
REGISTRATION_QA_SPECIFIC_FIELDS = (
    REGISTRATION_QA_CONFIG_FIELDS - TRACK2P_CONFIG_FIELDS
) | {"level"}
RUNNER_SPECIFIC_FIELDS = (
    TRACK2P_POLICY_FIELDS
    | TRACK2P_POLICY_DP_FIELDS
    | TRACK2P_POLICY_PRUNED_FIELDS
    | TRACK2P_POLICY_COMPONENT_FIELDS
    | TRACK2P_POLICY_COHERENCE_SUFFIX_FIELDS
    | TRACK2P_POLICY_TEACHER_ADJACENT_RESCUE_FIELDS
    | CONFIGURABLE_LOSO_FIELDS
    | MONOTONE_LOSO_FIELDS
    | SOLVER_PRIOR_FIELDS
    | REGISTRATION_QA_SPECIFIC_FIELDS
)
RUN_SPEC_FIELDS = (
    TRACK2P_CONFIG_FIELDS
    | REGISTRATION_QA_CONFIG_FIELDS
    | RUN_METADATA_FIELDS
    | RUNNER_SPECIFIC_FIELDS
)
RUNNER_CONFIG_FIELDS: dict[str, set[str]] = {
    DEFAULT_RUNNER: set(TRACK2P_CONFIG_FIELDS),
    TRACK2P_POLICY_RUNNER: set(TRACK2P_CONFIG_FIELDS | TRACK2P_POLICY_FIELDS),
    TRACK2P_POLICY_DP_RUNNER: set(TRACK2P_CONFIG_FIELDS | TRACK2P_POLICY_DP_FIELDS),
    TRACK2P_POLICY_PRUNED_RUNNER: set(
        TRACK2P_CONFIG_FIELDS | TRACK2P_POLICY_PRUNED_FIELDS
    ),
    TRACK2P_POLICY_COMPONENT_RUNNER: set(
        TRACK2P_CONFIG_FIELDS | TRACK2P_POLICY_COMPONENT_FIELDS
    ),
    TRACK2P_POLICY_COHERENCE_SUFFIX_RUNNER: set(
        TRACK2P_CONFIG_FIELDS | TRACK2P_POLICY_COHERENCE_SUFFIX_FIELDS
    ),
    TRACK2P_POLICY_TEACHER_ADJACENT_RESCUE_RUNNER: set(
        TRACK2P_CONFIG_FIELDS | TRACK2P_POLICY_TEACHER_ADJACENT_RESCUE_FIELDS
    ),
    "track2p-loso-calibration": set(TRACK2P_CONFIG_FIELDS | CONFIGURABLE_LOSO_FIELDS),
    "track2p-monotone-loso": set(TRACK2P_CONFIG_FIELDS | MONOTONE_LOSO_FIELDS),
    "track2p-solver-prior-loso": set(TRACK2P_CONFIG_FIELDS | SOLVER_PRIOR_FIELDS),
    "registration-qa": set(REGISTRATION_QA_CONFIG_FIELDS | {"level"}),
}
COMPARISON_FIELDS = {"name", "inputs", "output", "format", "highlight_best"}
RUNNER_ALIASES = {
    DEFAULT_RUNNER: DEFAULT_RUNNER,
    "track2p-benchmark": DEFAULT_RUNNER,
    TRACK2P_POLICY_RUNNER: TRACK2P_POLICY_RUNNER,
    TRACK2P_POLICY_DP_RUNNER: TRACK2P_POLICY_DP_RUNNER,
    TRACK2P_POLICY_PRUNED_RUNNER: TRACK2P_POLICY_PRUNED_RUNNER,
    TRACK2P_POLICY_COMPONENT_RUNNER: TRACK2P_POLICY_COMPONENT_RUNNER,
    "track2p-component-cleanup": TRACK2P_POLICY_COMPONENT_RUNNER,
    TRACK2P_POLICY_COHERENCE_SUFFIX_RUNNER: TRACK2P_POLICY_COHERENCE_SUFFIX_RUNNER,
    "track2p-coherence-suffix-stitch": TRACK2P_POLICY_COHERENCE_SUFFIX_RUNNER,
    "track2p-component-coherence-suffix-stitch": (
        TRACK2P_POLICY_COHERENCE_SUFFIX_RUNNER
    ),
    TRACK2P_POLICY_TEACHER_ADJACENT_RESCUE_RUNNER: (
        TRACK2P_POLICY_TEACHER_ADJACENT_RESCUE_RUNNER
    ),
    "track2p-teacher-adjacent-rescue": TRACK2P_POLICY_TEACHER_ADJACENT_RESCUE_RUNNER,
    "track2p-loso-calibration": "track2p-loso-calibration",
    "track2p-configurable-loso": "track2p-loso-calibration",
    "track2p-configurable-loso-calibration": "track2p-loso-calibration",
    "track2p-monotone-loso": "track2p-monotone-loso",
    "track2p-monotone-loso-calibration": "track2p-monotone-loso",
    "monotone-loso": "track2p-monotone-loso",
    "track2p-solver-prior-loso": "track2p-solver-prior-loso",
    "registration-qa": "registration-qa",
}
RUNNER_CHOICES = frozenset(RUNNER_ALIASES)


@dataclass(frozen=True)
class BenchmarkRunSpec:
    """One configured benchmark run from a manifest."""

    name: str
    config: Any
    output: Path
    runner: BenchmarkRunner = DEFAULT_RUNNER
    output_format: OutputFormat = "csv"
    runner_kwargs: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class BenchmarkComparisonSpec:
    """One aggregate comparison table from manifest run outputs."""

    name: str
    inputs: Mapping[str, str]
    output: Path
    output_format: str = "markdown"
    highlight_best: bool = False


@dataclass(frozen=True)
class BenchmarkManifest:
    """Parsed benchmark manifest with resolved filesystem paths."""

    path: Path
    runs: tuple[BenchmarkRunSpec, ...]
    comparisons: tuple[BenchmarkComparisonSpec, ...] = ()


@dataclass(frozen=True)
class BenchmarkOutputSummary:
    """Output metadata for one completed manifest artifact."""

    name: str
    output: Path
    rows: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "name": self.name,
            "output": str(self.output),
            "rows": int(self.rows),
        }


@dataclass(frozen=True)
class BenchmarkManifestResult:
    """Completed benchmark manifest outputs."""

    runs: tuple[BenchmarkOutputSummary, ...]
    comparisons: tuple[BenchmarkOutputSummary, ...]

    def to_dict(self) -> dict[str, list[dict[str, int | str]]]:
        return {
            "runs": [run.to_dict() for run in self.runs],
            "comparisons": [comparison.to_dict() for comparison in self.comparisons],
        }


def load_benchmark_manifest(
    manifest_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    progress: bool | None = None,
) -> BenchmarkManifest:
    """Load a JSON benchmark manifest and resolve paths relative to it."""

    path = Path(manifest_path)
    raw_manifest = _load_json_object(path)
    base_dir = path.resolve().parent
    output_base_dir = base_dir if output_dir is None else Path(output_dir)

    defaults = raw_manifest.get("defaults", {})
    if not isinstance(defaults, Mapping):
        raise ValueError("Manifest 'defaults' must be a JSON object")
    _reject_unknown_keys(defaults, RUN_SPEC_FIELDS, location="defaults")

    raw_runs = raw_manifest.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError("Manifest must contain a non-empty 'runs' list")

    runs = tuple(
        _parse_run_spec(
            raw_run,
            defaults=defaults,
            base_dir=base_dir,
            output_base_dir=output_base_dir,
            progress=progress,
        )
        for raw_run in raw_runs
    )
    _validate_unique_names(run.name for run in runs)

    raw_comparisons = raw_manifest.get("comparisons", [])
    if not isinstance(raw_comparisons, list):
        raise ValueError("Manifest 'comparisons' must be a JSON list when provided")
    comparisons = tuple(
        _parse_comparison_spec(
            raw_comparison,
            output_base_dir=output_base_dir,
        )
        for raw_comparison in raw_comparisons
    )
    _validate_unique_names(comparison.name for comparison in comparisons)

    return BenchmarkManifest(path=path, runs=runs, comparisons=comparisons)


def run_benchmark_manifest(manifest: BenchmarkManifest) -> BenchmarkManifestResult:
    """Run all benchmark entries and comparison tables from a manifest."""

    run_summaries: list[BenchmarkOutputSummary] = []
    run_outputs: dict[str, Path] = {}
    for run_spec in manifest.runs:
        rows = _run_benchmark_rows(run_spec)
        _write_run_rows(run_spec, rows)
        run_summaries.append(
            BenchmarkOutputSummary(
                name=run_spec.name,
                output=run_spec.output,
                rows=len(rows),
            )
        )
        run_outputs[run_spec.name] = run_spec.output

    comparison_summaries: list[BenchmarkOutputSummary] = []
    for comparison_spec in manifest.comparisons:
        comparison_inputs = _comparison_inputs(
            comparison_spec, run_outputs=run_outputs, manifest_path=manifest.path
        )
        rows = aggregate_rows(load_labeled_rows(comparison_inputs))
        write_comparison(
            rows,
            comparison_spec.output,
            comparison_spec.output_format,
            highlight_best=comparison_spec.highlight_best,
        )
        comparison_summaries.append(
            BenchmarkOutputSummary(
                name=comparison_spec.name,
                output=comparison_spec.output,
                rows=len(rows),
            )
        )

    return BenchmarkManifestResult(
        runs=tuple(run_summaries), comparisons=tuple(comparison_summaries)
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the manifest benchmark CLI parser."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark suite",
        description="Run a reproducible benchmark suite from a JSON manifest.",
    )
    parser.add_argument("manifest", type=Path, help="JSON benchmark manifest")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Resolve manifest output paths relative to this directory instead of the manifest directory",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override per-run benchmark progress reporting",
    )
    parser.add_argument(
        "--summary-format",
        choices=("json", "table"),
        default="json",
        help="Stdout summary format",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run benchmark suite CLI."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    manifest = load_benchmark_manifest(
        args.manifest, output_dir=args.output_dir, progress=args.progress
    )
    result = run_benchmark_manifest(manifest)
    if args.summary_format == "json":
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(_format_summary_table(result))
    return 0


def _parse_run_spec(
    raw_run: Any,
    *,
    defaults: ManifestObject,
    base_dir: Path,
    output_base_dir: Path,
    progress: bool | None,
) -> BenchmarkRunSpec:
    if not isinstance(raw_run, Mapping):
        raise ValueError("Each manifest run must be a JSON object")
    _reject_unknown_keys(raw_run, RUN_SPEC_FIELDS, location="runs[]")

    run_data = {**defaults, **raw_run}
    runner = _runner_name(run_data.get("runner", DEFAULT_RUNNER))
    _reject_incompatible_runner_keys(run_data, runner)
    name = str(run_data.get("name", _default_run_name(run_data)))
    output_format = _output_format(run_data.get("format", "csv"))
    output = _resolve_output_path(
        run_data.get("output"),
        default_name=f"{_slugify(name)}.{_benchmark_output_suffix(output_format)}",
        output_base_dir=output_base_dir,
    )
    config = _run_config(runner, run_data, base_dir=base_dir)
    if progress is not None:
        config = replace(config, progress=progress)
    return BenchmarkRunSpec(
        name=name,
        config=config,
        output=output,
        runner=runner,
        output_format=output_format,
        runner_kwargs=_runner_kwargs(run_data, runner),
    )


def _run_benchmark_rows(run_spec: BenchmarkRunSpec) -> list[dict[str, Any]]:
    if run_spec.runner == DEFAULT_RUNNER:
        return [
            result.to_dict()
            for result in run_track2p_benchmark(
                cast(Track2pBenchmarkConfig, run_spec.config)
            )
        ]
    if run_spec.runner == TRACK2P_POLICY_RUNNER:
        return _run_track2p_policy_rows(
            cast(Track2pBenchmarkConfig, run_spec.config),
            dict(run_spec.runner_kwargs or {}),
        )
    if run_spec.runner == TRACK2P_POLICY_DP_RUNNER:
        return _run_track2p_policy_dp_rows(
            cast(Track2pBenchmarkConfig, run_spec.config),
            dict(run_spec.runner_kwargs or {}),
        )
    if run_spec.runner == TRACK2P_POLICY_PRUNED_RUNNER:
        return _run_track2p_policy_pruned_rows(
            cast(Track2pBenchmarkConfig, run_spec.config),
            dict(run_spec.runner_kwargs or {}),
        )
    if run_spec.runner == TRACK2P_POLICY_COMPONENT_RUNNER:
        return _run_track2p_policy_component_rows(
            cast(Track2pBenchmarkConfig, run_spec.config),
            dict(run_spec.runner_kwargs or {}),
        )
    if run_spec.runner == TRACK2P_POLICY_COHERENCE_SUFFIX_RUNNER:
        return _run_track2p_policy_coherence_suffix_rows(
            cast(Track2pBenchmarkConfig, run_spec.config),
            dict(run_spec.runner_kwargs or {}),
        )
    if run_spec.runner == TRACK2P_POLICY_TEACHER_ADJACENT_RESCUE_RUNNER:
        return _run_track2p_policy_teacher_adjacent_rescue_rows(
            cast(Track2pBenchmarkConfig, run_spec.config),
            dict(run_spec.runner_kwargs or {}),
        )
    if run_spec.runner == "track2p-loso-calibration":
        return _run_configurable_loso_rows(
            cast(Track2pBenchmarkConfig, run_spec.config),
            dict(run_spec.runner_kwargs or {}),
        )
    if run_spec.runner == "track2p-monotone-loso":
        return _run_monotone_loso_rows(
            cast(Track2pBenchmarkConfig, run_spec.config),
            dict(run_spec.runner_kwargs or {}),
        )
    if run_spec.runner == "track2p-solver-prior-loso":
        return _run_solver_prior_loso_rows(
            cast(Track2pBenchmarkConfig, run_spec.config),
            dict(run_spec.runner_kwargs or {}),
        )
    if run_spec.runner == "registration-qa":
        return _run_registration_qa_rows(
            run_spec.config,
            dict(run_spec.runner_kwargs or {}),
        )
    raise ValueError(f"Unsupported benchmark manifest runner: {run_spec.runner!r}")


def _runner_name(value: Any) -> BenchmarkRunner:
    runner = str(value)
    if runner not in RUNNER_ALIASES:
        raise ValueError(
            "Manifest run runner must be one of: " + ", ".join(sorted(RUNNER_CHOICES))
        )
    return cast(BenchmarkRunner, RUNNER_ALIASES[runner])


def _reject_incompatible_runner_keys(run_data: ManifestObject, runner: str) -> None:
    allowed_specific = _runner_specific_fields(runner)
    disallowed = sorted(
        key
        for key in RUNNER_SPECIFIC_FIELDS - TRACK2P_CONFIG_FIELDS - allowed_specific
        if key in run_data
    )
    if disallowed:
        raise ValueError(
            f"Runner {runner!r} does not support manifest keys: "
            + ", ".join(disallowed)
        )


def _runner_specific_fields(runner: str) -> set[str]:
    if runner == DEFAULT_RUNNER:
        return set()
    if runner == TRACK2P_POLICY_RUNNER:
        return set(TRACK2P_POLICY_FIELDS)
    if runner == TRACK2P_POLICY_DP_RUNNER:
        return set(TRACK2P_POLICY_DP_FIELDS)
    if runner == TRACK2P_POLICY_PRUNED_RUNNER:
        return set(TRACK2P_POLICY_PRUNED_FIELDS)
    if runner == TRACK2P_POLICY_COMPONENT_RUNNER:
        return set(TRACK2P_POLICY_COMPONENT_FIELDS)
    if runner == TRACK2P_POLICY_COHERENCE_SUFFIX_RUNNER:
        return set(TRACK2P_POLICY_COHERENCE_SUFFIX_FIELDS)
    if runner == TRACK2P_POLICY_TEACHER_ADJACENT_RESCUE_RUNNER:
        return set(TRACK2P_POLICY_TEACHER_ADJACENT_RESCUE_FIELDS)
    if runner == "track2p-loso-calibration":
        return set(CONFIGURABLE_LOSO_FIELDS)
    if runner == "track2p-monotone-loso":
        return set(MONOTONE_LOSO_FIELDS)
    if runner == "track2p-solver-prior-loso":
        return set(SOLVER_PRIOR_FIELDS)
    if runner == "registration-qa":
        return set(REGISTRATION_QA_SPECIFIC_FIELDS)
    raise ValueError(f"Unsupported benchmark manifest runner: {runner!r}")


def _runner_kwargs(run_data: ManifestObject, runner: str) -> dict[str, Any]:
    if runner == DEFAULT_RUNNER:
        return {}
    if runner == TRACK2P_POLICY_RUNNER:
        return {key: run_data[key] for key in TRACK2P_POLICY_FIELDS if key in run_data}
    if runner == TRACK2P_POLICY_DP_RUNNER:
        return {
            key: run_data[key] for key in TRACK2P_POLICY_DP_FIELDS if key in run_data
        }
    if runner == TRACK2P_POLICY_PRUNED_RUNNER:
        return {
            key: run_data[key]
            for key in TRACK2P_POLICY_PRUNED_FIELDS
            if key in run_data
        }
    if runner == TRACK2P_POLICY_COMPONENT_RUNNER:
        return {
            key: run_data[key]
            for key in TRACK2P_POLICY_COMPONENT_FIELDS
            if key in run_data
        }
    if runner == TRACK2P_POLICY_COHERENCE_SUFFIX_RUNNER:
        return {
            key: run_data[key]
            for key in TRACK2P_POLICY_COHERENCE_SUFFIX_FIELDS
            if key in run_data
        }
    if runner == TRACK2P_POLICY_TEACHER_ADJACENT_RESCUE_RUNNER:
        return {
            key: run_data[key]
            for key in TRACK2P_POLICY_TEACHER_ADJACENT_RESCUE_FIELDS
            if key in run_data
        }
    if runner == "track2p-loso-calibration":
        return _configurable_loso_runner_kwargs(run_data)
    if runner == "track2p-monotone-loso":
        return _monotone_loso_runner_kwargs(run_data)
    if runner == "track2p-solver-prior-loso":
        return {key: run_data[key] for key in SOLVER_PRIOR_FIELDS if key in run_data}
    if runner == "registration-qa":
        return {"level": str(run_data["level"])} if "level" in run_data else {}
    raise ValueError(f"Unsupported benchmark manifest runner: {runner!r}")


def _configurable_loso_runner_kwargs(run_data: ManifestObject) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if "feature_names" in run_data:
        kwargs["feature_names"] = _string_tuple(
            run_data["feature_names"], location="feature_names"
        )
    if "sample_weight_strategy" in run_data:
        kwargs["sample_weight_strategy"] = str(run_data["sample_weight_strategy"])
    if "calibration_model" in run_data:
        model_kind = str(run_data["calibration_model"])
        kwargs["calibration_model"] = model_kind
        kwargs["model_kind"] = model_kind
    if "calibration_model_kwargs" in run_data:
        model_kwargs = _mapping_option(
            run_data["calibration_model_kwargs"], location="calibration_model_kwargs"
        )
        kwargs["calibration_model_kwargs"] = model_kwargs
        kwargs["model_kwargs"] = model_kwargs
    if "calibration_model_kwargs_json" in run_data:
        kwargs["calibration_model_kwargs_json"] = str(
            run_data["calibration_model_kwargs_json"]
        )
    if "hard_negative_options" in run_data:
        from bayescatrack.experiments.calibration_hard_negatives import (
            CandidateHardNegativeOptions,
        )

        kwargs["hard_negative_options"] = CandidateHardNegativeOptions(
            **_mapping_option(
                run_data["hard_negative_options"], location="hard_negative_options"
            )
        )
    for key in (
        "hard_negative_ratio",
        "hard_negative_top_k",
        "hard_negative_column_candidates",
        "hard_negative_features",
    ):
        if key in run_data:
            kwargs[key] = run_data[key]
    return kwargs


def _monotone_loso_runner_kwargs(run_data: ManifestObject) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if "feature_names" in run_data:
        kwargs["feature_names"] = _string_tuple(
            run_data["feature_names"], location="feature_names"
        )
    if "monotone_options" in run_data:
        from bayescatrack.association.monotone_ranker import MonotoneRankerOptions

        kwargs["monotone_options"] = MonotoneRankerOptions(
            **_mapping_option(run_data["monotone_options"], location="monotone_options")
        )
    return kwargs


def _parse_comparison_spec(
    raw_comparison: Any,
    *,
    output_base_dir: Path,
) -> BenchmarkComparisonSpec:
    if not isinstance(raw_comparison, Mapping):
        raise ValueError("Each manifest comparison must be a JSON object")
    _reject_unknown_keys(raw_comparison, COMPARISON_FIELDS, location="comparisons[]")

    inputs = raw_comparison.get("inputs")
    if not isinstance(inputs, Mapping) or not inputs:
        raise ValueError("Manifest comparison 'inputs' must be a non-empty JSON object")
    comparison_inputs = {str(label): str(source) for label, source in inputs.items()}

    name = str(raw_comparison.get("name", "comparison"))
    output_format = str(raw_comparison.get("format", "markdown"))
    if output_format not in {"markdown", "csv"}:
        raise ValueError("Manifest comparison format must be 'markdown' or 'csv'")
    output = _resolve_output_path(
        raw_comparison.get("output"),
        default_name=f"{_slugify(name)}.{_comparison_output_suffix(output_format)}",
        output_base_dir=output_base_dir,
    )
    raw_highlight_best = raw_comparison.get("highlight_best", False)
    if not isinstance(raw_highlight_best, bool):
        raise ValueError("Manifest comparison 'highlight_best' must be a boolean")
    return BenchmarkComparisonSpec(
        name=name,
        inputs=comparison_inputs,
        output=output,
        output_format=output_format,
        highlight_best=raw_highlight_best,
    )


def _run_config(
    runner: BenchmarkRunner, run_data: ManifestObject, *, base_dir: Path
) -> Any:
    if runner == "registration-qa":
        return _registration_qa_config(run_data, base_dir=base_dir)

    config_defaults: dict[str, Any] = {}
    required = ("data", "method")
    if runner in {
        TRACK2P_POLICY_RUNNER,
        TRACK2P_POLICY_DP_RUNNER,
        TRACK2P_POLICY_PRUNED_RUNNER,
        TRACK2P_POLICY_COMPONENT_RUNNER,
        TRACK2P_POLICY_COHERENCE_SUFFIX_RUNNER,
        TRACK2P_POLICY_TEACHER_ADJACENT_RESCUE_RUNNER,
    }:
        config_defaults = {
            "method": "global-assignment",
            "include_non_cells": False,
            "weighted_masks": False,
            "weighted_centroids": False,
            "exclude_overlapping_pixels": False,
        }
        required = ("data",)
    elif runner in {"track2p-loso-calibration", "track2p-monotone-loso"}:
        config_defaults = {
            "method": "global-assignment",
            "split": "leave-one-subject-out",
            "cost": "calibrated",
        }
        required = ("data",)
    elif runner == "track2p-solver-prior-loso":
        config_defaults = {
            "method": "global-assignment",
            "split": "leave-one-subject-out",
            "cost": "registered-iou",
        }
        required = ("data",)

    config_kwargs = _track2p_config_kwargs(
        run_data,
        base_dir=base_dir,
        config_defaults=config_defaults,
        required=required,
    )
    return Track2pBenchmarkConfig(**config_kwargs)


def _track2p_config_kwargs(
    run_data: ManifestObject,
    *,
    base_dir: Path,
    config_defaults: Mapping[str, Any] | None = None,
    required: Sequence[str] = ("data", "method"),
) -> dict[str, Any]:
    config_kwargs = dict(config_defaults or {})
    config_kwargs.update(
        {key: value for key, value in run_data.items() if key in TRACK2P_CONFIG_FIELDS}
    )
    missing_required = [key for key in required if key not in config_kwargs]
    if missing_required:
        raise ValueError(
            "Manifest run is missing required Track2p config keys: "
            + ", ".join(missing_required)
        )
    for key in ("data", "reference", "track2p_teacher_reference"):
        if key in config_kwargs and config_kwargs[key] is not None:
            config_kwargs[key] = _resolve_input_path(
                config_kwargs[key], base_dir=base_dir
            )
    return config_kwargs


def _registration_qa_config(run_data: ManifestObject, *, base_dir: Path) -> Any:
    from bayescatrack.experiments.registration_qa_report import RegistrationQAConfig

    config_kwargs = {
        key: value
        for key, value in run_data.items()
        if key in REGISTRATION_QA_CONFIG_FIELDS
    }
    if "data" not in config_kwargs:
        raise ValueError(
            "Manifest run is missing required registration-qa config key: data"
        )
    for key in ("data", "reference"):
        if key in config_kwargs and config_kwargs[key] is not None:
            config_kwargs[key] = _resolve_input_path(
                config_kwargs[key], base_dir=base_dir
            )
    return RegistrationQAConfig(**config_kwargs)


def _run_options(runner: BenchmarkRunner, run_data: ManifestObject) -> ManifestObject:
    option_fields = RUNNER_CONFIG_FIELDS[runner] - TRACK2P_CONFIG_FIELDS
    if runner == "registration-qa":
        option_fields = {"level"}
    return {key: run_data[key] for key in option_fields if key in run_data}


def _run_manifest_entry(run_spec: BenchmarkRunSpec) -> list[dict[str, Any]]:
    if run_spec.runner == "track2p":
        results = run_track2p_benchmark(cast(Track2pBenchmarkConfig, run_spec.config))
        return [result.to_dict() for result in results]
    if run_spec.runner == TRACK2P_POLICY_RUNNER:
        return _run_track2p_policy_rows(
            cast(Track2pBenchmarkConfig, run_spec.config),
            dict(run_spec.runner_kwargs or {}),
        )
    if run_spec.runner == TRACK2P_POLICY_DP_RUNNER:
        return _run_track2p_policy_dp_rows(
            cast(Track2pBenchmarkConfig, run_spec.config),
            dict(run_spec.runner_kwargs or {}),
        )
    if run_spec.runner == TRACK2P_POLICY_PRUNED_RUNNER:
        return _run_track2p_policy_pruned_rows(
            cast(Track2pBenchmarkConfig, run_spec.config),
            dict(run_spec.runner_kwargs or {}),
        )
    if run_spec.runner == TRACK2P_POLICY_COMPONENT_RUNNER:
        return _run_track2p_policy_component_rows(
            cast(Track2pBenchmarkConfig, run_spec.config),
            dict(run_spec.runner_kwargs or {}),
        )
    if run_spec.runner == TRACK2P_POLICY_COHERENCE_SUFFIX_RUNNER:
        return _run_track2p_policy_coherence_suffix_rows(
            cast(Track2pBenchmarkConfig, run_spec.config),
            dict(run_spec.runner_kwargs or {}),
        )
    if run_spec.runner == TRACK2P_POLICY_TEACHER_ADJACENT_RESCUE_RUNNER:
        return _run_track2p_policy_teacher_adjacent_rescue_rows(
            cast(Track2pBenchmarkConfig, run_spec.config),
            dict(run_spec.runner_kwargs or {}),
        )
    if run_spec.runner == "track2p-loso-calibration":
        return _run_configurable_loso_rows(
            cast(Track2pBenchmarkConfig, run_spec.config),
            dict(run_spec.runner_kwargs or {}),
        )
    if run_spec.runner == "track2p-monotone-loso":
        return _run_monotone_loso_rows(
            cast(Track2pBenchmarkConfig, run_spec.config),
            dict(run_spec.runner_kwargs or {}),
        )
    if run_spec.runner == "track2p-solver-prior-loso":
        return _run_solver_prior_loso_rows(
            cast(Track2pBenchmarkConfig, run_spec.config),
            dict(run_spec.runner_kwargs or {}),
        )
    if run_spec.runner == "registration-qa":
        return _run_registration_qa_rows(
            run_spec.config,
            dict(run_spec.runner_kwargs or {}),
        )
    raise AssertionError(f"Unhandled benchmark runner: {run_spec.runner}")


def _policy_threshold_method(value: Any) -> Literal["otsu", "min"]:
    threshold_method = str(value)
    if threshold_method not in {"otsu", "min"}:
        raise ValueError("threshold_method must be 'otsu' or 'min'")
    return cast(Literal["otsu", "min"], threshold_method)


def _run_track2p_policy_rows(
    config: Track2pBenchmarkConfig, options: ManifestObject
) -> list[dict[str, Any]]:
    from bayescatrack.experiments.track2p_policy_benchmark import (
        TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
        TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
        run_track2p_policy_benchmark,
    )

    results = run_track2p_policy_benchmark(
        config,
        threshold_method=_policy_threshold_method(
            options.get("threshold_method", TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD)
        ),
        iou_distance_threshold=_float_option(
            options,
            "iou_distance_threshold",
            default=TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
        ),
        transform_type=config.transform_type,
        cell_probability_threshold=config.cell_probability_threshold,
    )
    return [result.to_dict() for result in results]


def _run_track2p_policy_dp_rows(
    config: Track2pBenchmarkConfig, options: ManifestObject
) -> list[dict[str, Any]]:
    from bayescatrack.experiments.track2p_policy_benchmark import (
        TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
        TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    )
    from bayescatrack.experiments.track2p_policy_dp_benchmark import (
        Track2pPolicyDPConfig,
        run_track2p_policy_dp_benchmark,
    )

    dp_kwargs: dict[str, Any] = {
        "threshold_method": _policy_threshold_method(
            options.get("threshold_method", TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD)
        ),
        "iou_distance_threshold": _float_option(
            options,
            "iou_distance_threshold",
            default=TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
        ),
        "gap_penalty": float(config.gap_penalty),
        "max_gap": int(config.max_gap),
    }
    for key in (
        "row_top_k",
        "beam_width",
        "path_candidates_per_seed",
        "path_selection_beam_width",
    ):
        if key in options:
            dp_kwargs[key] = int(options[key])
    for key in (
        "rescue_min_iou",
        "threshold_rescue_margin",
        "accepted_bonus",
        "rescue_penalty",
        "threshold_margin_weight",
    ):
        if key in options:
            dp_kwargs[key] = float(options[key])

    results = run_track2p_policy_dp_benchmark(
        config,
        dp_config=Track2pPolicyDPConfig(**dp_kwargs),
        transform_type=config.transform_type,
        cell_probability_threshold=config.cell_probability_threshold,
    )
    return [result.to_dict() for result in results]


def _run_track2p_policy_pruned_rows(
    config: Track2pBenchmarkConfig, options: ManifestObject
) -> list[dict[str, Any]]:
    from bayescatrack.experiments.track2p_policy_benchmark import (
        TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
        TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    )
    from bayescatrack.experiments.track2p_policy_pruned_benchmark import (
        Track2pPolicyPruneConfig,
        run_track2p_policy_pruned_benchmark,
    )

    prune_defaults = Track2pPolicyPruneConfig()
    prune_config = Track2pPolicyPruneConfig(
        threshold_margin=_float_option(
            options,
            "prune_threshold_margin",
            default=prune_defaults.threshold_margin,
        ),
        competition_margin=_float_option(
            options,
            "prune_competition_margin",
            default=prune_defaults.competition_margin,
        ),
        min_area_ratio=_float_option(
            options,
            "prune_min_area_ratio",
            default=prune_defaults.min_area_ratio,
        ),
        centroid_distance=_float_option(
            options,
            "prune_centroid_distance",
            default=prune_defaults.centroid_distance,
        ),
    )
    results = run_track2p_policy_pruned_benchmark(
        config,
        threshold_method=_policy_threshold_method(
            options.get("threshold_method", TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD)
        ),
        iou_distance_threshold=_float_option(
            options,
            "iou_distance_threshold",
            default=TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
        ),
        prune_config=prune_config,
        transform_type=config.transform_type,
        cell_probability_threshold=config.cell_probability_threshold,
    )
    return [result.to_dict() for result in results]


def _run_track2p_policy_component_rows(
    config: Track2pBenchmarkConfig, options: ManifestObject
) -> list[dict[str, Any]]:
    from bayescatrack.experiments.track2p_policy_benchmark import (
        TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
        TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    )
    from bayescatrack.experiments.track2p_policy_component_audit import (
        ComponentCleanupConfig,
        run_track2p_policy_component_audit,
    )

    cleanup_defaults = ComponentCleanupConfig()
    cleanup_config = ComponentCleanupConfig(
        threshold_margin_scale=_float_option(
            options,
            "threshold_margin_scale",
            default=cleanup_defaults.threshold_margin_scale,
        ),
        competition_margin_scale=_float_option(
            options,
            "competition_margin_scale",
            default=cleanup_defaults.competition_margin_scale,
        ),
        area_ratio_floor=_float_option(
            options,
            "area_ratio_floor",
            default=cleanup_defaults.area_ratio_floor,
        ),
        centroid_distance_scale=_float_option(
            options,
            "centroid_distance_scale",
            default=cleanup_defaults.centroid_distance_scale,
        ),
        split_risk_threshold=_float_option(
            options,
            "split_risk_threshold",
            default=cleanup_defaults.split_risk_threshold,
        ),
        split_penalty=_float_option(
            options,
            "split_penalty",
            default=cleanup_defaults.split_penalty,
        ),
        min_side_observations=int(
            options.get("min_side_observations", cleanup_defaults.min_side_observations)
        ),
        threshold_margin_weight=_float_option(
            options,
            "threshold_margin_weight",
            default=cleanup_defaults.threshold_margin_weight,
        ),
        row_margin_weight=_float_option(
            options,
            "row_margin_weight",
            default=cleanup_defaults.row_margin_weight,
        ),
        column_margin_weight=_float_option(
            options,
            "column_margin_weight",
            default=cleanup_defaults.column_margin_weight,
        ),
        centroid_distance_weight=_float_option(
            options,
            "centroid_distance_weight",
            default=cleanup_defaults.centroid_distance_weight,
        ),
        area_ratio_weight=_float_option(
            options,
            "area_ratio_weight",
            default=cleanup_defaults.area_ratio_weight,
        ),
    )
    output = run_track2p_policy_component_audit(
        config,
        threshold_method=_policy_threshold_method(
            options.get("threshold_method", TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD)
        ),
        iou_distance_threshold=_float_option(
            options,
            "iou_distance_threshold",
            default=TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
        ),
        transform_type=config.transform_type,
        cell_probability_threshold=config.cell_probability_threshold,
        cleanup_config=cleanup_config,
        apply_splits=_bool_option(options, "apply_splits", default=True),
    )
    return [result.to_dict() for result in output.results]


def _run_track2p_policy_coherence_suffix_rows(
    config: Track2pBenchmarkConfig, options: ManifestObject
) -> list[dict[str, Any]]:
    from bayescatrack.experiments.track2p_policy_benchmark import (
        TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
        TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    )
    from bayescatrack.experiments.track2p_policy_coherence_suffix_stitch_whatif import (
        CoherenceSuffixStitchGate,
        run_track2p_policy_coherence_suffix_stitch_whatif,
    )
    from bayescatrack.experiments.track2p_policy_component_audit import (
        ComponentCleanupConfig,
    )

    cleanup_defaults = ComponentCleanupConfig()
    cleanup_config = ComponentCleanupConfig(
        threshold_margin_scale=_float_option(
            options,
            "threshold_margin_scale",
            default=cleanup_defaults.threshold_margin_scale,
        ),
        competition_margin_scale=_float_option(
            options,
            "competition_margin_scale",
            default=cleanup_defaults.competition_margin_scale,
        ),
        area_ratio_floor=_float_option(
            options,
            "area_ratio_floor",
            default=cleanup_defaults.area_ratio_floor,
        ),
        centroid_distance_scale=_float_option(
            options,
            "centroid_distance_scale",
            default=cleanup_defaults.centroid_distance_scale,
        ),
        split_risk_threshold=_float_option(
            options,
            "split_risk_threshold",
            default=cleanup_defaults.split_risk_threshold,
        ),
        split_penalty=_float_option(
            options,
            "split_penalty",
            default=cleanup_defaults.split_penalty,
        ),
        min_side_observations=int(
            options.get("min_side_observations", cleanup_defaults.min_side_observations)
        ),
        threshold_margin_weight=_float_option(
            options,
            "threshold_margin_weight",
            default=cleanup_defaults.threshold_margin_weight,
        ),
        row_margin_weight=_float_option(
            options,
            "row_margin_weight",
            default=cleanup_defaults.row_margin_weight,
        ),
        column_margin_weight=_float_option(
            options,
            "column_margin_weight",
            default=cleanup_defaults.column_margin_weight,
        ),
        centroid_distance_weight=_float_option(
            options,
            "centroid_distance_weight",
            default=cleanup_defaults.centroid_distance_weight,
        ),
        area_ratio_weight=_float_option(
            options,
            "area_ratio_weight",
            default=cleanup_defaults.area_ratio_weight,
        ),
    )
    gate_defaults = CoherenceSuffixStitchGate()
    gate = CoherenceSuffixStitchGate(
        suffix_path_length=int(
            options.get("suffix_path_length", gate_defaults.suffix_path_length)
        ),
        min_cell_probability=_float_option(
            options,
            "min_cell_probability",
            default=gate_defaults.min_cell_probability,
        ),
        min_area_ratio=_float_option(
            options,
            "min_area_ratio",
            default=gate_defaults.min_area_ratio,
        ),
        max_centroid_distance=_float_option(
            options,
            "max_centroid_distance",
            default=gate_defaults.max_centroid_distance,
        ),
        min_shifted_iou=_float_option(
            options,
            "min_shifted_iou",
            default=gate_defaults.min_shifted_iou,
        ),
        min_motion_consistency=_float_option(
            options,
            "min_motion_consistency",
            default=gate_defaults.min_motion_consistency,
        ),
        min_shape_consistency=_float_option(
            options,
            "min_shape_consistency",
            default=gate_defaults.min_shape_consistency,
        ),
        max_stitches_per_subject=int(
            options.get(
                "max_stitches_per_subject",
                gate_defaults.max_stitches_per_subject,
            )
        ),
    )
    output = run_track2p_policy_coherence_suffix_stitch_whatif(
        config,
        threshold_method=_policy_threshold_method(
            options.get("threshold_method", TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD)
        ),
        iou_distance_threshold=_float_option(
            options,
            "iou_distance_threshold",
            default=TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
        ),
        transform_type=config.transform_type,
        cell_probability_threshold=config.cell_probability_threshold,
        cleanup_config=cleanup_config,
        gate=gate,
        edge_top_k=int(options.get("edge_top_k", 25)),
        path_beam_width=int(options.get("path_beam_width", 100)),
    )
    return [
        dict(row) for row in output.result_rows if str(row.get("subject", "")) != "ALL"
    ]


def _run_track2p_policy_teacher_adjacent_rescue_rows(
    config: Track2pBenchmarkConfig, options: ManifestObject
) -> list[dict[str, Any]]:
    from bayescatrack.experiments.track2p_policy_benchmark import (
        TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
        TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD,
    )
    from bayescatrack.experiments.track2p_policy_component_audit import (
        ComponentCleanupConfig,
    )
    from bayescatrack.experiments.track2p_policy_teacher_adjacent_rescue import (
        TeacherEdgeFeatureGate,
        run_track2p_policy_teacher_adjacent_rescue,
    )

    cleanup_defaults = ComponentCleanupConfig()
    cleanup_config = ComponentCleanupConfig(
        threshold_margin_scale=_float_option(
            options,
            "threshold_margin_scale",
            default=cleanup_defaults.threshold_margin_scale,
        ),
        competition_margin_scale=_float_option(
            options,
            "competition_margin_scale",
            default=cleanup_defaults.competition_margin_scale,
        ),
        area_ratio_floor=_float_option(
            options,
            "area_ratio_floor",
            default=cleanup_defaults.area_ratio_floor,
        ),
        centroid_distance_scale=_float_option(
            options,
            "centroid_distance_scale",
            default=cleanup_defaults.centroid_distance_scale,
        ),
        split_risk_threshold=_float_option(
            options,
            "split_risk_threshold",
            default=cleanup_defaults.split_risk_threshold,
        ),
        split_penalty=_float_option(
            options,
            "split_penalty",
            default=cleanup_defaults.split_penalty,
        ),
        min_side_observations=int(
            options.get("min_side_observations", cleanup_defaults.min_side_observations)
        ),
        threshold_margin_weight=_float_option(
            options,
            "threshold_margin_weight",
            default=cleanup_defaults.threshold_margin_weight,
        ),
        row_margin_weight=_float_option(
            options,
            "row_margin_weight",
            default=cleanup_defaults.row_margin_weight,
        ),
        column_margin_weight=_float_option(
            options,
            "column_margin_weight",
            default=cleanup_defaults.column_margin_weight,
        ),
        centroid_distance_weight=_float_option(
            options,
            "centroid_distance_weight",
            default=cleanup_defaults.centroid_distance_weight,
        ),
        area_ratio_weight=_float_option(
            options,
            "area_ratio_weight",
            default=cleanup_defaults.area_ratio_weight,
        ),
    )
    allow_source_inserts = None
    if "allow_source_inserts" in options:
        allow_source_inserts = _bool_option(
            options, "allow_source_inserts", default=True
        )
    allow_source_insertions = None
    if "allow_source_insertions" in options:
        allow_source_insertions = _bool_option(
            options, "allow_source_insertions", default=True
        )
    teacher_feature_gate = TeacherEdgeFeatureGate(
        min_registered_iou=_optional_float_option(
            options,
            "teacher_min_registered_iou",
            "teacher_gate_min_registered_iou",
        ),
        min_threshold_margin=_optional_float_option(
            options,
            "teacher_min_threshold_margin",
            "teacher_gate_min_threshold_margin",
        ),
        min_row_margin=_optional_float_option(
            options,
            "teacher_min_row_margin",
            "teacher_gate_min_row_margin",
        ),
        min_column_margin=_optional_float_option(
            options,
            "teacher_min_column_margin",
            "teacher_gate_min_column_margin",
        ),
        max_centroid_distance=_optional_float_option(
            options,
            "teacher_max_centroid_distance",
            "teacher_gate_max_centroid_distance",
        ),
        min_area_ratio=_optional_float_option(
            options,
            "teacher_min_area_ratio",
            "teacher_gate_min_area_ratio",
        ),
        min_cell_probability=_optional_float_option(
            options,
            "teacher_min_cell_probability",
            "teacher_gate_min_cell_probability",
        ),
        require_hungarian=_first_bool_option(
            options,
            "teacher_require_hungarian",
            "teacher_require_hungarian_assignment",
            "teacher_gate_require_hungarian",
            default=False,
        ),
    )
    output = run_track2p_policy_teacher_adjacent_rescue(
        config,
        threshold_method=_policy_threshold_method(
            options.get("threshold_method", TRACK2P_POLICY_DEFAULT_THRESHOLD_METHOD)
        ),
        iou_distance_threshold=_float_option(
            options,
            "iou_distance_threshold",
            default=TRACK2P_POLICY_DEFAULT_IOU_DISTANCE_THRESHOLD,
        ),
        transform_type=config.transform_type,
        cell_probability_threshold=config.cell_probability_threshold,
        cleanup_config=cleanup_config,
        allow_completing_rescue=_bool_option(
            options, "allow_completing_rescue", default=False
        ),
        allow_teacher_complete_row_rescue=_bool_option(
            options, "allow_teacher_complete_row_rescue", default=False
        ),
        allow_teacher_supported_completion=_bool_option(
            options, "allow_teacher_supported_completion", default=False
        ),
        allow_teacher_supported_completing_rescue=_bool_option(
            options, "allow_teacher_supported_completing_rescue", default=False
        ),
        allow_teacher_confirmed_completing_rescue=_bool_option(
            options, "allow_teacher_confirmed_completing_rescue", default=False
        ),
        allow_completing_source_backfill=_bool_option(
            options, "allow_completing_source_backfill", default=False
        ),
        allow_completing_fragment_merge=_bool_option(
            options, "allow_completing_fragment_merge", default=False
        ),
        allow_completing_fragment_merges=_bool_option(
            options, "allow_completing_fragment_merges", default=False
        ),
        allow_source_backfill=_bool_option(
            options, "allow_source_backfill", default=True
        ),
        allow_source_inserts=allow_source_inserts,
        allow_source_insertions=allow_source_insertions,
        allow_seed_source_backfill=_bool_option(
            options, "allow_seed_source_backfill", default=False
        ),
        allow_seed_completing_backfill=_bool_option(
            options, "allow_seed_completing_backfill", default=False
        ),
        allow_seed_completing_rescue=_bool_option(
            options, "allow_seed_completing_rescue", default=False
        ),
        allow_completing_seed_source_backfill=_bool_option(
            options, "allow_completing_seed_source_backfill", default=False
        ),
        allow_fragment_merges=_bool_option(
            options, "allow_fragment_merges", default=True
        ),
        teacher_edge_order=str(options.get("teacher_edge_order", "structural")),
        min_component_observations=int(options.get("min_component_observations", 1)),
        max_applied_edits=_nonnegative_int_or_none(
            options.get("max_applied_edits"), name="max_applied_edits"
        ),
        teacher_feature_gate=teacher_feature_gate,
        teacher_feature_preset=str(options.get("teacher_feature_preset", "none")),
    )
    return [result.to_dict() for result in output.results]


def _run_configurable_loso_rows(
    config: Track2pBenchmarkConfig, options: ManifestObject
) -> list[dict[str, Any]]:
    from bayescatrack.experiments.calibration_hard_negatives import (
        CandidateHardNegativeOptions,
    )
    from bayescatrack.experiments.track2p_configurable_loso_calibration import (
        run_track2p_configurable_loso_calibration,
    )

    if isinstance(options.get("hard_negative_options"), CandidateHardNegativeOptions):
        hard_negative_options = options["hard_negative_options"]
    elif "hard_negative_options" in options:
        hard_negative_options = CandidateHardNegativeOptions(
            **_mapping_option(
                options["hard_negative_options"], location="hard_negative_options"
            )
        )
    else:
        hard_negative_options = CandidateHardNegativeOptions(
            negative_to_positive_ratio=_float_option(
                options, "hard_negative_ratio", default=4.0
            ),
            candidate_top_k_per_anchor=_positive_int_or_none(
                options.get("hard_negative_top_k", 20), name="hard_negative_top_k"
            ),
            include_column_candidates=_bool_option(
                options, "hard_negative_column_candidates", default=True
            ),
            hardness_feature_names=_string_tuple(
                options.get("hard_negative_features", ()),
                name="hard_negative_features",
                allow_empty=True,
            ),
        )

    if "model_kwargs" in options:
        model_kwargs = _mapping_option(options["model_kwargs"], location="model_kwargs")
    else:
        model_kwargs = _mapping_option(
            options,
            key="calibration_model_kwargs",
            json_key="calibration_model_kwargs_json",
        )

    kwargs: dict[str, Any] = {
        "sample_weight_strategy": str(options.get("sample_weight_strategy", "none")),
        "model_kind": str(
            options.get("model_kind", options.get("calibration_model", "logistic"))
        ),
        "model_kwargs": model_kwargs,
        "hard_negative_options": hard_negative_options,
    }
    feature_names = _feature_names_option(options)
    if feature_names is not None:
        kwargs["feature_names"] = feature_names
    return run_track2p_configurable_loso_calibration(config, **kwargs).to_rows()


def _run_monotone_loso_rows(
    config: Track2pBenchmarkConfig, options: ManifestObject
) -> list[dict[str, Any]]:
    from bayescatrack.association.monotone_ranker import MonotoneRankerOptions
    from bayescatrack.experiments.track2p_monotone_loso_calibration import (
        run_track2p_monotone_loso_calibration,
    )

    kwargs: dict[str, Any] = {}
    feature_names = _feature_names_option(options)
    if feature_names is not None:
        kwargs["feature_names"] = feature_names
    raw_monotone_options = options.get("monotone_options")
    if isinstance(raw_monotone_options, MonotoneRankerOptions):
        kwargs["monotone_options"] = raw_monotone_options
        return run_track2p_monotone_loso_calibration(config, **kwargs).to_rows()
    if raw_monotone_options is not None:
        monotone_kwargs = _mapping_option(
            raw_monotone_options, location="monotone_options"
        )
    else:
        monotone_kwargs = _mapping_option(
            options,
            key="monotone_ranker_kwargs",
            json_key="monotone_ranker_kwargs_json",
        )
    if monotone_kwargs:
        kwargs["monotone_options"] = MonotoneRankerOptions(**monotone_kwargs)
    return run_track2p_monotone_loso_calibration(config, **kwargs).to_rows()


def _run_solver_prior_loso_rows(
    config: Track2pBenchmarkConfig, options: ManifestObject
) -> list[dict[str, Any]]:
    from bayescatrack.experiments.solver_prior_tuning import (
        SolverPriorSearchConfig,
        run_track2p_loso_solver_priors,
    )

    search_kwargs: dict[str, Any] = {}
    if "start_costs" in options:
        search_kwargs["start_costs"] = _float_tuple(
            options["start_costs"], name="start_costs", positive=True
        )
    if "end_costs" in options:
        search_kwargs["end_costs"] = _float_tuple(
            options["end_costs"], name="end_costs", positive=True, allow_empty=True
        )
    if "gap_penalties" in options:
        search_kwargs["gap_penalties"] = _float_tuple(
            options["gap_penalties"], name="gap_penalties", nonnegative=True
        )
    if "cost_thresholds" in options:
        search_kwargs["cost_thresholds"] = _threshold_tuple(
            options["cost_thresholds"], name="cost_thresholds"
        )
    if "objective" in options:
        search_kwargs["objective"] = str(options["objective"])
    return run_track2p_loso_solver_priors(
        config, search=SolverPriorSearchConfig(**search_kwargs)
    ).to_rows()


def _run_registration_qa_rows(
    config: Any, options: ManifestObject
) -> list[dict[str, Any]]:
    from bayescatrack.experiments.registration_qa_report import (
        run_registration_qa_report,
        summarize_registration_backend_usage,
        summarize_registration_qa_links,
    )

    level = _registration_qa_level(options.get("level", "summary"))
    rows: Sequence[Mapping[str, Any]] = run_registration_qa_report(config)
    if level == "summary":
        rows = summarize_registration_qa_links(rows)
    elif level == "backend-audit":
        rows = summarize_registration_backend_usage(rows)
    return [dict(row) for row in rows]


def _write_run_rows(
    run_spec: BenchmarkRunSpec, rows: Sequence[Mapping[str, Any]]
) -> None:
    if run_spec.runner == "registration-qa":
        _write_registration_qa_run_rows(run_spec, rows)
        return
    write_results(rows, run_spec.output, run_spec.output_format)


def _write_registration_qa_run_rows(
    run_spec: BenchmarkRunSpec, rows: Sequence[Mapping[str, Any]]
) -> None:
    from bayescatrack.experiments.registration_qa_report import (
        write_registration_backend_audit_results,
        write_registration_qa_results,
    )

    level = _registration_qa_level(
        (run_spec.runner_kwargs or {}).get("level", "summary")
    )
    if level == "backend-audit":
        write_registration_backend_audit_results(
            rows, run_spec.output, run_spec.output_format
        )
    else:
        write_registration_qa_results(rows, run_spec.output, run_spec.output_format)


def _comparison_inputs(
    comparison_spec: BenchmarkComparisonSpec,
    *,
    run_outputs: Mapping[str, Path],
    manifest_path: Path,
) -> list[ComparisonInput]:
    inputs: list[ComparisonInput] = []
    base_dir = manifest_path.resolve().parent
    for label, source in comparison_spec.inputs.items():
        if source in run_outputs:
            source_path = run_outputs[source]
        else:
            source_path = _resolve_input_path(source, base_dir=base_dir)
        inputs.append(ComparisonInput(label=label, path=source_path))
    return inputs


def _load_json_object(path: Path) -> ManifestObject:
    if path.suffix.casefold() != ".json":
        raise ValueError(f"Benchmark manifests must be JSON files, got {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, Mapping):
        raise ValueError("Benchmark manifest must be a JSON object")
    return data


def _resolve_input_path(value: Any, *, base_dir: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return base_dir / path


def _resolve_output_path(
    value: Any, *, default_name: str, output_base_dir: Path
) -> Path:
    if value is None:
        path = Path("benchmark-results") / default_name
    else:
        path = Path(str(value))
    if path.is_absolute():
        return path
    return output_base_dir / path


def _reject_unknown_keys(
    raw_object: ManifestObject, allowed: set[str], *, location: str
) -> None:
    unknown_keys = sorted(str(key) for key in raw_object if key not in allowed)
    if unknown_keys:
        raise ValueError(
            f"Unknown keys in manifest {location}: {', '.join(unknown_keys)}"
        )


def _validate_unique_names(names: Iterable[str]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for name in names:
        if name in seen:
            duplicates.add(name)
        seen.add(name)
    if duplicates:
        raise ValueError(
            f"Manifest names must be unique: {', '.join(sorted(duplicates))}"
        )


def _default_run_name(run_data: ManifestObject) -> str:
    runner = _runner_name(run_data.get("runner", DEFAULT_RUNNER))
    if runner == "track2p-loso-calibration":
        model = run_data.get("calibration_model")
        return runner if model is None else f"{runner}-{model}"
    if runner != DEFAULT_RUNNER:
        return runner
    method = str(run_data.get("method", "run"))
    cost = run_data.get("cost")
    return method if cost is None else f"{method}-{cost}"


def _output_format(value: Any) -> OutputFormat:
    output_format = str(value)
    if output_format not in {"table", "json", "csv"}:
        raise ValueError("Manifest run format must be 'table', 'json', or 'csv'")
    return cast(OutputFormat, output_format)


def _benchmark_output_suffix(output_format: OutputFormat) -> str:
    return {"csv": "csv", "json": "json", "table": "md"}[output_format]


def _comparison_output_suffix(output_format: str) -> str:
    return "csv" if output_format == "csv" else "md"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-._")
    return slug or "benchmark"


def _mapping_option(value: Any, *, location: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Manifest {location!r} must be a JSON object")
    return dict(value)


def _string_tuple(value: Any, *, location: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(token.strip() for token in value.split(",") if token.strip())
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    raise ValueError(f"Manifest {location!r} must be a string or JSON list")


def _format_summary_table(result: BenchmarkManifestResult) -> str:
    rows: list[tuple[str, str, int, Path]] = []
    rows.extend(
        ("run", summary.name, summary.rows, summary.output) for summary in result.runs
    )
    rows.extend(
        ("comparison", summary.name, summary.rows, summary.output)
        for summary in result.comparisons
    )
    if not rows:
        return "No benchmark outputs were written."

    header = "| kind | name | rows | output |"
    separator = "| --- | --- | ---: | --- |"
    body = [header, separator]
    body.extend(
        f"| {kind} | {name} | {row_count} | {output} |"
        for kind, name, row_count, output in rows
    )
    return "\n".join(body)


def _feature_names_option(options: ManifestObject) -> tuple[str, ...] | None:
    if "feature_names" not in options:
        return None
    feature_names = _string_tuple(
        options["feature_names"], name="feature_names", allow_empty=False
    )
    if not feature_names:
        raise ValueError("feature_names must contain at least one feature")
    return feature_names


def _string_tuple(
    value: Any,
    *,
    location: str | None = None,
    name: str | None = None,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    label = name or location or "value"
    if value is None:
        values: tuple[str, ...] = ()
    elif isinstance(value, str):
        values = tuple(token.strip() for token in value.split(",") if token.strip())
    elif isinstance(value, Sequence):
        values = tuple(str(token) for token in value)
    else:
        raise ValueError(f"{label} must be a comma-separated string or JSON array")
    if not values and not allow_empty:
        raise ValueError(f"{label} must not be empty")
    return values


def _mapping_option(
    value_or_options: Any,
    *,
    location: str | None = None,
    key: str | None = None,
    json_key: str | None = None,
) -> dict[str, Any] | None:
    if key is None:
        if value_or_options is None:
            return {}
        if not isinstance(value_or_options, Mapping):
            label = location or "value"
            raise ValueError(f"Manifest {label!r} must be a JSON object")
        return dict(value_or_options)

    if json_key is None:
        raise ValueError("json_key is required when key is provided")
    options = cast(ManifestObject, value_or_options)
    has_mapping = key in options
    has_json = json_key in options
    if has_mapping and has_json:
        raise ValueError(f"Use either {key!r} or {json_key!r}, not both")
    if has_json:
        parsed = json.loads(str(options[json_key]))
    elif has_mapping:
        parsed = options[key]
    else:
        return None
    if not isinstance(parsed, Mapping):
        raise ValueError(f"{key} must be a JSON object")
    return dict(parsed)


def _float_option(options: ManifestObject, key: str, *, default: float) -> float:
    value = float(options.get(key, default))
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value


def _optional_float_option(options: ManifestObject, *keys: str) -> float | None:
    for key in keys:
        if key not in options or options[key] is None:
            continue
        value = float(options[key])
        if not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
        return value
    return None


def _bool_option(options: ManifestObject, key: str, *, default: bool) -> bool:
    value = options.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _first_bool_option(
    options: ManifestObject, *keys: str, default: bool
) -> bool:
    for key in keys:
        if key in options:
            return _bool_option(options, key, default=default)
    return default


def _nonnegative_int_or_none(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.casefold() in {"none", "null", "all"}:
        return None
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative or null")
    return parsed


def _positive_int_or_none(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.casefold() in {"none", "null", "all"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive or null")
    return parsed


def _float_tuple(
    value: Any,
    *,
    name: str,
    positive: bool = False,
    nonnegative: bool = False,
    allow_empty: bool = False,
) -> tuple[float, ...]:
    raw_values = _sequence_option(value, name=name, allow_empty=allow_empty)
    parsed = tuple(float(item) for item in raw_values)
    if not parsed and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    for item in parsed:
        if not math.isfinite(item):
            raise ValueError(f"{name} values must be finite")
        if positive and item <= 0.0:
            raise ValueError(f"{name} values must be positive")
        if nonnegative and item < 0.0:
            raise ValueError(f"{name} values must be non-negative")
    return parsed


def _threshold_tuple(value: Any, *, name: str) -> tuple[float | None, ...]:
    parsed: list[float | None] = []
    for item in _sequence_option(value, name=name, allow_empty=False):
        if item is None or (
            isinstance(item, str) and item.casefold() in {"none", "null"}
        ):
            parsed.append(None)
            continue
        threshold = float(item)
        if not math.isfinite(threshold):
            raise ValueError(f"{name} values must be finite or null")
        parsed.append(threshold)
    return tuple(parsed)


def _sequence_option(value: Any, *, name: str, allow_empty: bool) -> tuple[Any, ...]:
    if isinstance(value, str):
        values = tuple(token.strip() for token in value.split(",") if token.strip())
    elif isinstance(value, Sequence):
        values = tuple(value)
    else:
        raise ValueError(f"{name} must be a comma-separated string or JSON array")
    if not values and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    return values


def _registration_qa_level(value: Any) -> str:
    level = str(value)
    if level not in {"summary", "links", "backend-audit"}:
        raise ValueError(
            "registration-qa level must be 'summary', 'links', or 'backend-audit'"
        )
    return level


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
