"""CLI for heuristic Track2p prediction error taxonomy reports."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from bayescatrack.evaluation.track_error_taxonomy import classify_track_errors
from bayescatrack.experiments.track2p_benchmark import (
    Track2pBenchmarkConfig,
    _load_reference_for_subject,
    _validate_reference_for_benchmark,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-error-taxonomy",
        description="Classify false-positive and false-negative Track2p links in a prediction CSV.",
    )
    parser.add_argument(
        "--data", required=True, type=Path, help="One Track2p subject directory"
    )
    parser.add_argument(
        "--prediction",
        required=True,
        type=Path,
        help="CSV with one predicted track per row",
    )
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
    parser.add_argument("--cost-threshold", type=float, default=None)
    parser.add_argument("--ambiguity-rank-threshold", type=float, default=0.15)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("json", "csv"), default="json")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    rows_or_summary = run_error_taxonomy(args)
    if args.output is not None:
        _write_output(rows_or_summary, args.output, args.format)
    else:
        _write_stdout(rows_or_summary, args.format)
    return 0


def run_error_taxonomy(
    args: argparse.Namespace,
) -> list[dict[str, Any]] | dict[str, Any]:
    config = Track2pBenchmarkConfig(
        data=args.data,
        method="global-assignment",
        plane_name=args.plane_name,
        input_format=args.input_format,
        reference=args.reference,
        reference_kind=args.reference_kind,
        allow_track2p_as_reference_for_smoke_test=args.allow_track2p_as_reference_for_smoke_test,
        curated_only=args.curated_only,
    )
    reference = _load_reference_for_subject(
        args.data, data_root=args.data, config=config
    )
    _validate_reference_for_benchmark(reference, subject_dir=args.data, config=config)
    predicted = _read_prediction_csv(
        args.prediction, session_names=reference.session_names
    )
    reference_matrix = reference.filtered_indices(curated_only=args.curated_only)
    report = classify_track_errors(
        predicted,
        reference_matrix,
        cost_threshold=args.cost_threshold,
        ambiguity_rank_threshold=args.ambiguity_rank_threshold,
    )
    if args.summary_only:
        return {
            "subject": args.data.name,
            "prediction": str(args.prediction),
            "reference_source": reference.source,
            **report.summary(),
        }
    return [
        {
            "subject": args.data.name,
            "prediction": str(args.prediction),
            "reference_source": reference.source,
            **row,
        }
        for row in report.to_rows()
    ]


def _read_prediction_csv(path: Path, *, session_names: tuple[str, ...]) -> np.ndarray:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} does not contain a CSV header")
        missing = [name for name in session_names if name not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"Prediction CSV is missing session columns: {', '.join(missing)}"
            )
        rows: list[list[object]] = []
        for raw_row in reader:
            rows.append(
                [_parse_nullable_int(raw_row.get(name, "")) for name in session_names]
            )
    matrix = np.empty((len(rows), len(session_names)), dtype=object)
    matrix[:] = None
    for row_index, row in enumerate(rows):
        for column_index, value in enumerate(row):
            matrix[row_index, column_index] = value
    return matrix


def _parse_nullable_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.casefold() in {"", "none", "nan", "null", "-1"}:
        return None
    return int(float(text))


def _write_output(
    data: list[dict[str, Any]] | dict[str, Any], output_path: Path, output_format: str
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return
    rows = data if isinstance(data, list) else [data]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)


def _write_stdout(
    data: list[dict[str, Any]] | dict[str, Any], output_format: str
) -> None:
    if output_format == "json":
        print(json.dumps(data, indent=2))
        return
    rows = data if isinstance(data, list) else [data]
    writer = csv.DictWriter(sys.stdout, fieldnames=_fieldnames(rows))
    writer.writeheader()
    writer.writerows(rows)


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "subject",
        "kind",
        "category",
        "session_a",
        "session_b",
        "roi_a",
        "roi_b",
        "cost",
    ]
    remaining = sorted({key for row in rows for key in row} - set(preferred))
    return [key for key in preferred if any(key in row for row in rows)] + remaining


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
