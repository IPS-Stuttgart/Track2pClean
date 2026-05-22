"""One-command Track2p result-improvement suite and objective selection."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bayescatrack.experiments.advanced_improvement_workbench import (
    ActiveLabelConfig,
    select_active_label_candidates,
    track2p_result_improvement_manifest,
    write_csv_rows,
)
from bayescatrack.experiments.benchmark_manifest import (
    load_benchmark_manifest,
    run_benchmark_manifest,
)
from bayescatrack.experiments.structured_objective_tuning import (
    nested_select_variants_by_track_metric,
    select_best_variants_by_track_metric,
)


@dataclass(frozen=True)
class ResultImprovementSelectionConfig:
    """Configuration for the result-improvement suite runner."""

    data_root: Path
    output_root: Path
    reference_root: Path | None = None
    manifest_output: Path | None = None
    selection_output: Path | None = None
    max_gap: int = 2
    transform_type: str = "fov-affine"
    metric: str = "complete_track_f1"
    group_by: str = "variant"
    nested_held_out_field: str | None = "subject"
    tie_breakers: tuple[str, ...] = ("pairwise_f1", "pairwise_precision")
    progress: bool | None = True
    active_label_input: Path | None = None
    active_label_output: Path | None = None
    max_active_label_rows: int = 500


def run_result_improvement_selection(
    config: ResultImprovementSelectionConfig,
) -> dict[str, Any]:
    """Generate, run, and select a Track2p result-improvement suite."""

    manifest_path = _manifest_output_path(config)
    manifest = track2p_result_improvement_manifest(
        data_root=str(config.data_root.resolve()),
        reference_root=(
            None
            if config.reference_root is None
            else str(config.reference_root.resolve())
        ),
        output_root=str(config.output_root.resolve()),
        max_gap=int(config.max_gap),
        transform_type=config.transform_type,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    suite = load_benchmark_manifest(manifest_path, progress=config.progress)
    suite_result = run_benchmark_manifest(suite)
    benchmark_csvs = tuple(
        summary.output
        for summary in suite_result.runs
        if summary.output.suffix.casefold() == ".csv"
        and summary.name
        not in {"registration-qa", "track2p-baseline", "oracle-gt-links"}
        and "teacher-prior" not in summary.name
    )
    selection_rows = select_structured_objective_rows(
        benchmark_csvs,
        metric=config.metric,
        group_by=config.group_by,
        nested_held_out_field=config.nested_held_out_field,
        tie_breakers=config.tie_breakers,
    )
    selection_path = _selection_output_path(config)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(
        json.dumps(selection_rows, indent=2) + "\n", encoding="utf-8"
    )

    active_label_rows = 0
    if config.active_label_input is not None and config.active_label_output is not None:
        active_rows = select_active_label_candidates(
            _read_csv_rows(config.active_label_input),
            config=ActiveLabelConfig(max_rows=int(config.max_active_label_rows)),
        )
        write_csv_rows(active_rows, config.active_label_output)
        active_label_rows = len(active_rows)

    return {
        "manifest": str(manifest_path),
        "selection": str(selection_path),
        "benchmark_csvs": [str(path) for path in benchmark_csvs],
        "suite": suite_result.to_dict(),
        "selection_rows": len(selection_rows),
        "active_label_rows": active_label_rows,
    }


def select_structured_objective_rows(
    csv_paths: Sequence[Path],
    *,
    metric: str = "complete_track_f1",
    group_by: str = "variant",
    nested_held_out_field: str | None = "subject",
    tie_breakers: Sequence[str] = ("pairwise_f1", "pairwise_precision"),
) -> list[dict[str, float | int | str]]:
    """Select variants from benchmark CSVs using a structured objective."""

    rows = _read_many_csv_rows(csv_paths)
    if nested_held_out_field:
        return nested_select_variants_by_track_metric(
            rows,
            metric=metric,
            held_out_field=nested_held_out_field,
            group_by=group_by,
            tie_breakers=tuple(tie_breakers),
        )
    return select_best_variants_by_track_metric(
        rows,
        metric=metric,
        group_by=group_by,
        tie_breakers=tuple(tie_breakers),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-result-improvement",
        description=(
            "Generate the Track2p result-improvement manifest, run it, and select "
            "the best variant by a structured objective such as complete_track_f1."
        ),
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--reference-root", type=Path, default=None)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument("--selection-output", type=Path, default=None)
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument("--transform-type", default="fov-affine")
    parser.add_argument("--metric", default="complete_track_f1")
    parser.add_argument("--group-by", default="variant")
    parser.add_argument(
        "--nested-held-out-field",
        default="subject",
        help="Use fold-clean selection by this field; pass empty string to disable",
    )
    parser.add_argument("--tie-breaker", action="append", default=None)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Override per-run progress reporting in the generated suite",
    )
    parser.add_argument(
        "--active-label-input",
        type=Path,
        default=None,
        help="Optional teacher-audit or edge-ranking CSV to rank after the suite",
    )
    parser.add_argument("--active-label-output", type=Path, default=None)
    parser.add_argument("--max-active-label-rows", type=int, default=500)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    nested = str(args.nested_held_out_field).strip() or None
    result = run_result_improvement_selection(
        ResultImprovementSelectionConfig(
            data_root=args.data_root,
            reference_root=args.reference_root,
            output_root=args.output_root,
            manifest_output=args.manifest_output,
            selection_output=args.selection_output,
            max_gap=args.max_gap,
            transform_type=args.transform_type,
            metric=args.metric,
            group_by=args.group_by,
            nested_held_out_field=nested,
            tie_breakers=tuple(
                args.tie_breaker or ("pairwise_f1", "pairwise_precision")
            ),
            progress=args.progress,
            active_label_input=args.active_label_input,
            active_label_output=args.active_label_output,
            max_active_label_rows=args.max_active_label_rows,
        )
    )
    print(json.dumps(result, indent=2))
    return 0


def _manifest_output_path(config: ResultImprovementSelectionConfig) -> Path:
    if config.manifest_output is not None:
        return config.manifest_output
    return config.output_root / "track2p_result_improvements.json"


def _selection_output_path(config: ResultImprovementSelectionConfig) -> Path:
    if config.selection_output is not None:
        return config.selection_output
    return config.output_root / "nested_complete_track_selection.json"


def _read_many_csv_rows(paths: Sequence[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(_read_csv_rows(path))
    return rows


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
