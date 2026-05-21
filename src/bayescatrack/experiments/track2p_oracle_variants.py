"""CLI for Track2p oracle-ceiling variants."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from bayescatrack.evaluation.oracle_variants import score_oracle_variants
from bayescatrack.experiments.track2p_benchmark import (
    Track2pBenchmarkConfig,
    _load_reference_for_subject,
    _validate_reference_for_benchmark,
    discover_subject_dirs,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-oracle-variants",
        description="Score oracle reference-row, consecutive-link and gap-limited Track2p variants.",
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument(
        "--reference-kind",
        default="manual-gt",
        choices=("auto", "manual-gt", "track2p-output", "aligned-subject-rows"),
    )
    parser.add_argument("--plane", dest="plane_name", default="plane0")
    parser.add_argument(
        "--input-format", default="auto", choices=("auto", "suite2p", "npy")
    )
    parser.add_argument(
        "--allow-track2p-as-reference-for-smoke-test", action="store_true"
    )
    parser.add_argument("--curated-only", action="store_true")
    parser.add_argument(
        "--seed-sessions",
        default="0",
        help="Comma-separated seed-session indices, e.g. 0,1,2",
    )
    parser.add_argument(
        "--max-gaps",
        default="1,2,3",
        help="Comma-separated max-gap values for gap-limited oracle variants",
    )
    parser.add_argument(
        "--include-behavior", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--include-non-cells", action="store_true")
    parser.add_argument("--cell-probability-threshold", type=float, default=0.5)
    parser.add_argument("--weighted-masks", action="store_true")
    parser.add_argument(
        "--exclude-overlapping-pixels",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("json", "csv", "table"), default="table")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    rows = run_oracle_variant_rows(args)
    if args.output is not None:
        _write_rows(rows, args.output, args.format)
    else:
        _write_stdout(rows, args.format)
    return 0


def run_oracle_variant_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    config = Track2pBenchmarkConfig(
        data=args.data,
        method="oracle-gt-links",
        plane_name=args.plane_name,
        input_format=args.input_format,
        reference=args.reference,
        reference_kind=args.reference_kind,
        allow_track2p_as_reference_for_smoke_test=args.allow_track2p_as_reference_for_smoke_test,
        curated_only=args.curated_only,
        include_behavior=args.include_behavior,
        include_non_cells=args.include_non_cells,
        cell_probability_threshold=args.cell_probability_threshold,
        weighted_masks=args.weighted_masks,
        exclude_overlapping_pixels=args.exclude_overlapping_pixels,
    )
    rows: list[dict[str, Any]] = []
    seed_sessions = _parse_int_list(args.seed_sessions, name="--seed-sessions")
    max_gaps = _parse_int_list(args.max_gaps, name="--max-gaps")
    for subject_dir in discover_subject_dirs(args.data):
        reference = _load_reference_for_subject(
            subject_dir, data_root=args.data, config=config
        )
        _validate_reference_for_benchmark(
            reference, subject_dir=subject_dir, config=config
        )
        for result in score_oracle_variants(
            reference,
            curated_only=args.curated_only,
            seed_sessions=seed_sessions,
            max_gaps=max_gaps,
        ):
            rows.append(
                {
                    "subject": subject_dir.name,
                    "reference_source": reference.source,
                    "n_sessions": reference.n_sessions,
                    **result.to_row(),
                }
            )
    return rows


def _parse_int_list(raw: str, *, name: str) -> tuple[int, ...]:
    values = tuple(int(token.strip()) for token in raw.split(",") if token.strip())
    if not values:
        raise ValueError(f"{name} must contain at least one integer")
    return values


def _write_rows(
    rows: list[dict[str, Any]], output_path: Path, output_format: str
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        return
    if output_format == "csv":
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_fieldnames(rows))
            writer.writeheader()
            writer.writerows(rows)
        return
    output_path.write_text(_format_table(rows) + "\n", encoding="utf-8")


def _write_stdout(rows: list[dict[str, Any]], output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(rows, indent=2))
        return
    if output_format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)
        return
    print(_format_table(rows))


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "subject",
        "variant",
        "seed_session",
        "max_gap",
        "pairwise_f1",
        "complete_track_f1",
        "oracle_tracks",
        "reference_source",
    ]
    remaining = sorted({key for row in rows for key in row} - set(preferred))
    return [key for key in preferred if any(key in row for row in rows)] + remaining


def _format_table(rows: list[dict[str, Any]]) -> str:
    columns = [
        "subject",
        "variant",
        "seed_session",
        "max_gap",
        "pairwise_f1",
        "complete_track_f1",
        "oracle_tracks",
    ]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [header, separator]
    for row in rows:
        body.append(
            "| "
            + " | ".join(_format_value(row.get(column, "")) for column in columns)
            + " |"
        )
    return "\n".join(body)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
