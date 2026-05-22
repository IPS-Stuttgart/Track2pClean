"""Validate benchmark manifests and print their resolved plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from bayescatrack.experiments.benchmark_manifest import (
    BenchmarkComparisonSpec,
    BenchmarkManifest,
    BenchmarkRunSpec,
    load_benchmark_manifest,
)

_PLAN_CONFIG_FIELDS = (
    "data",
    "reference",
    "track2p_teacher_reference",
    "method",
    "split",
    "cost",
    "plane_name",
    "input_format",
    "transform_type",
    "max_gap",
)


def build_manifest_plan(manifest: BenchmarkManifest) -> dict[str, Any]:
    """Return a JSON-serializable plan for a parsed manifest."""

    return {
        "manifest": str(manifest.path),
        "runs": [_run_plan(run) for run in manifest.runs],
        "comparisons": [
            _comparison_plan(comparison) for comparison in manifest.comparisons
        ],
    }


def validate_manifest_input_paths(manifest: BenchmarkManifest) -> None:
    """Raise ``FileNotFoundError`` when configured input paths do not exist."""

    missing: list[str] = []
    for run in manifest.runs:
        for attribute_name in ("data", "reference", "track2p_teacher_reference"):
            value = getattr(run.config, attribute_name, None)
            if value is None:
                continue
            path = Path(value)
            if not path.exists():
                missing.append(f"{run.name}.{attribute_name}: {path}")
    if missing:
        raise FileNotFoundError(
            "Benchmark manifest references missing input paths:\n- "
            + "\n- ".join(missing)
        )


def format_manifest_plan_table(plan: Mapping[str, Any]) -> str:
    """Format a manifest plan as Markdown tables."""

    sections = [f"# Benchmark manifest plan\n\nManifest: `{plan['manifest']}`"]
    sections.append(_format_runs_table(plan.get("runs", [])))
    comparisons = plan.get("comparisons", [])
    if comparisons:
        sections.append(_format_comparisons_table(comparisons))
    else:
        sections.append("## Comparisons\n\nNo comparisons configured.")
    return "\n\n".join(sections)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the manifest validation CLI parser."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark validate-suite",
        description="Validate a benchmark suite manifest and print the resolved plan.",
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
        help="Override per-run benchmark progress reporting in the parsed plan",
    )
    parser.add_argument(
        "--check-input-paths",
        action="store_true",
        help="Also require configured data/reference paths to exist on this machine",
    )
    parser.add_argument(
        "--format",
        choices=("json", "table"),
        default="table",
        help="Stdout plan format",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the manifest validation CLI."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    manifest = load_benchmark_manifest(
        args.manifest, output_dir=args.output_dir, progress=args.progress
    )
    if args.check_input_paths:
        validate_manifest_input_paths(manifest)
    plan = build_manifest_plan(manifest)
    if args.format == "json":
        print(json.dumps(plan, indent=2, sort_keys=True))
    else:
        print(format_manifest_plan_table(plan))
    return 0


def _run_plan(run: BenchmarkRunSpec) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": run.name,
        "runner": run.runner,
        "output": str(run.output),
        "format": str(run.output_format),
        "runner_option_keys": sorted((run.runner_kwargs or {}).keys()),
    }
    for field_name in _PLAN_CONFIG_FIELDS:
        value = getattr(run.config, field_name, None)
        if value is None:
            continue
        row[field_name] = str(value) if isinstance(value, Path) else value
    return row


def _comparison_plan(comparison: BenchmarkComparisonSpec) -> dict[str, Any]:
    return {
        "name": comparison.name,
        "output": str(comparison.output),
        "format": comparison.output_format,
        "highlight_best": comparison.highlight_best,
        "inputs": dict(comparison.inputs),
    }


def _format_runs_table(runs: list[dict[str, Any]]) -> str:
    body = [
        "## Runs",
        "",
        "| name | runner | method | cost | split | output | options |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in runs:
        body.append(
            "| "
            f"{row['name']} | "
            f"{row['runner']} | "
            f"{row.get('method', '')} | "
            f"{row.get('cost', '')} | "
            f"{row.get('split', '')} | "
            f"{row['output']} | "
            f"{', '.join(row.get('runner_option_keys', []))} |"
        )
    return "\n".join(body)


def _format_comparisons_table(comparisons: list[dict[str, Any]]) -> str:
    body = [
        "## Comparisons",
        "",
        "| name | format | output | inputs |",
        "| --- | --- | --- | --- |",
    ]
    for row in comparisons:
        inputs = row.get("inputs", {})
        if isinstance(inputs, Mapping):
            input_summary = ", ".join(
                f"{label}={source}" for label, source in inputs.items()
            )
        else:
            input_summary = ""
        body.append(
            "| "
            f"{row['name']} | "
            f"{row['format']} | "
            f"{row['output']} | "
            f"{input_summary} |"
        )
    return "\n".join(body)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
