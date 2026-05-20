"""Select calibrated-association features from edge-ranking summaries."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EdgeFeatureSelectionRule:
    """Thresholds for selecting features from edge-ranking summary rows."""

    min_row_hit_at_1: float | None = None
    min_column_hit_at_1: float | None = None
    min_mutual_top1_rate: float | None = None
    min_row_positive_margin_rate: float | None = None
    min_column_positive_margin_rate: float | None = None
    max_features: int | None = None


def select_features_from_edge_ranking_summaries(
    rows: Sequence[Mapping[str, Any]],
    *,
    rule: EdgeFeatureSelectionRule | None = None,
) -> list[dict[str, float | int | str]]:
    """Aggregate score-name diagnostics and return ranked selected features."""

    rule = rule or EdgeFeatureSelectionRule()
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("score_name", ""))].append(row)

    feature_rows = [_score_feature(name, group) for name, group in grouped.items() if name]
    feature_rows = [row for row in feature_rows if _passes_rule(row, rule)]
    feature_rows.sort(
        key=lambda row: (
            -float(row["selection_score"]),
            str(row["score_name"]),
        )
    )
    if rule.max_features is not None:
        feature_rows = feature_rows[: int(rule.max_features)]
    return feature_rows


def selected_feature_names(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Return only selected score names from selection rows."""

    return tuple(str(row["score_name"]) for row in rows)


def _score_feature(name: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int | str]:
    row_hit = _mean(rows, "row_hit_at_1_present")
    col_hit = _mean(rows, "column_hit_at_1_present")
    mutual = _mean(rows, "mutual_top1_rate_present")
    row_margin = _mean(rows, "row_positive_margin_rate")
    col_margin = _mean(rows, "column_positive_margin_rate")
    finite = _sum(rows, "finite_true_edges")
    gt_edges = _sum(rows, "gt_edges")
    missing = _sum(rows, "missing_edges")
    selection_score = np.nanmean([row_hit, col_hit, mutual, row_margin, col_margin])
    return {
        "score_name": name,
        "selection_score": _float_or_zero(selection_score),
        "mean_row_hit_at_1_present": _float_or_zero(row_hit),
        "mean_column_hit_at_1_present": _float_or_zero(col_hit),
        "mean_mutual_top1_rate_present": _float_or_zero(mutual),
        "mean_row_positive_margin_rate": _float_or_zero(row_margin),
        "mean_column_positive_margin_rate": _float_or_zero(col_margin),
        "summary_rows": int(len(rows)),
        "gt_edges": int(gt_edges),
        "finite_true_edges": int(finite),
        "missing_edges": int(missing),
    }


def _passes_rule(row: Mapping[str, Any], rule: EdgeFeatureSelectionRule) -> bool:
    checks = (
        ("mean_row_hit_at_1_present", rule.min_row_hit_at_1),
        ("mean_column_hit_at_1_present", rule.min_column_hit_at_1),
        ("mean_mutual_top1_rate_present", rule.min_mutual_top1_rate),
        ("mean_row_positive_margin_rate", rule.min_row_positive_margin_rate),
        ("mean_column_positive_margin_rate", rule.min_column_positive_margin_rate),
    )
    return all(threshold is None or float(row[name]) >= threshold for name, threshold in checks)


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    values = [_float(row.get(field)) for row in rows]
    values = [value for value in values if np.isfinite(value)]
    return float(np.mean(values)) if values else float("nan")


def _sum(rows: Sequence[Mapping[str, Any]], field: str) -> int:
    total = 0
    for row in rows:
        value = _float(row.get(field))
        if np.isfinite(value):
            total += int(value)
    return total


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _float_or_zero(value: Any) -> float:
    numeric = _float(value)
    return 0.0 if not np.isfinite(numeric) else float(numeric)


def _load_csv_rows(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with Path(path).open("r", newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark select-edge-ranking-features",
        description="Select feature names from edge-ranking summary CSV files.",
    )
    parser.add_argument("summary_csv", nargs="+", type=Path)
    parser.add_argument("--min-row-hit-at-1", type=float, default=None)
    parser.add_argument("--min-column-hit-at-1", type=float, default=None)
    parser.add_argument("--min-mutual-top1-rate", type=float, default=None)
    parser.add_argument("--min-row-positive-margin-rate", type=float, default=None)
    parser.add_argument("--min-column-positive-margin-rate", type=float, default=None)
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("json", "csv", "names"), default="json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rule = EdgeFeatureSelectionRule(
        min_row_hit_at_1=args.min_row_hit_at_1,
        min_column_hit_at_1=args.min_column_hit_at_1,
        min_mutual_top1_rate=args.min_mutual_top1_rate,
        min_row_positive_margin_rate=args.min_row_positive_margin_rate,
        min_column_positive_margin_rate=args.min_column_positive_margin_rate,
        max_features=args.max_features,
    )
    selected = select_features_from_edge_ranking_summaries(
        _load_csv_rows(args.summary_csv), rule=rule
    )
    if args.format == "names":
        payload = ",".join(selected_feature_names(selected)) + "\n"
    elif args.format == "json":
        payload = json.dumps(selected, indent=2) + "\n"
    else:
        if args.output is None:
            raise ValueError("--output is required for --format csv")
        _write_csv(selected, args.output)
        return 0
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
