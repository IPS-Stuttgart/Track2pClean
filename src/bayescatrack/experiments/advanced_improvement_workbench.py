"""Command-line workbench for advanced BayesCaTrack improvement experiments.

This module is intentionally standalone.  It can be invoked with
``python -m bayescatrack.experiments.advanced_improvement_workbench`` without
modifying the top-level CLI while the advanced workstreams are reviewed.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ActiveLabelConfig:
    """Weights for active manual-label candidate ranking."""

    uncertainty_weight: float = 1.0
    disagreement_weight: float = 1.0
    margin_weight: float = 1.0
    missing_edge_weight: float = 1.0
    max_rows: int = 500


@dataclass(frozen=True)
class StratifiedMetricConfig:
    """Configuration for stratified benchmark summaries."""

    group_fields: tuple[str, ...]
    metric_fields: tuple[str, ...]


def select_active_label_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: ActiveLabelConfig | None = None,
) -> list[dict[str, Any]]:
    """Rank edges/tracks that are most informative for additional manual labels."""

    cfg = config or ActiveLabelConfig()
    scored: list[dict[str, Any]] = []
    for row in rows:
        score = active_label_score(row, config=cfg)
        out = dict(row)
        out["active_label_score"] = score
        scored.append(out)
    scored.sort(key=lambda item: (-float(item["active_label_score"]), str(item)))
    return scored[: int(cfg.max_rows)]


def active_label_score(row: Mapping[str, Any], *, config: ActiveLabelConfig | None = None) -> float:
    """Return an active-label priority score from edge-ranking/teacher rows."""

    cfg = config or ActiveLabelConfig()
    row_margin = _safe_float(row.get("row_margin"), np.nan)
    column_margin = _safe_float(row.get("column_margin"), np.nan)
    finite_margins = [value for value in (row_margin, column_margin) if np.isfinite(value)]
    if finite_margins:
        low_margin_score = 1.0 / (1.0 + max(float(np.mean(finite_margins)), 0.0))
    else:
        low_margin_score = 1.0

    disagreement = 0.0
    if "in_ground_truth" in row and "in_track2p" in row:
        disagreement += float(bool(row.get("in_ground_truth")) != bool(row.get("in_track2p")))
    if "in_ground_truth" in row and "in_bayes" in row:
        disagreement += float(bool(row.get("in_ground_truth")) != bool(row.get("in_bayes")))
    if "in_track2p" in row and "in_bayes" in row:
        disagreement += 0.5 * float(bool(row.get("in_track2p")) != bool(row.get("in_bayes")))

    missing_edge = float(str(row.get("missing_reason", "")).strip() != "")
    true_score = _safe_float(row.get("true_score"), np.nan)
    uncertainty = 1.0 if not np.isfinite(true_score) else 1.0 / (1.0 + abs(true_score))

    return float(
        cfg.margin_weight * low_margin_score
        + cfg.disagreement_weight * disagreement
        + cfg.missing_edge_weight * missing_edge
        + cfg.uncertainty_weight * uncertainty
    )


def stratified_metric_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: StratifiedMetricConfig,
) -> list[dict[str, Any]]:
    """Aggregate benchmark metrics by arbitrary metadata fields."""

    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(field, "") for field in config.group_fields)
        groups[key].append(row)

    summaries: list[dict[str, Any]] = []
    for key, group_rows in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        summary: dict[str, Any] = {field: value for field, value in zip(config.group_fields, key, strict=True)}
        summary["rows"] = len(group_rows)
        for metric in config.metric_fields:
            values = np.asarray([_safe_float(row.get(metric), np.nan) for row in group_rows], dtype=float)
            values = values[np.isfinite(values)]
            summary[f"{metric}_mean"] = float(np.mean(values)) if values.size else float("nan")
            summary[f"{metric}_median"] = float(np.median(values)) if values.size else float("nan")
            summary[f"{metric}_min"] = float(np.min(values)) if values.size else float("nan")
            summary[f"{metric}_max"] = float(np.max(values)) if values.size else float("nan")
        summaries.append(summary)
    return summaries


def synthetic_stress_manifest(
    *,
    data_root: str,
    output_root: str,
    reference_root: str | None = None,
) -> dict[str, Any]:
    """Return a benchmark manifest covering controlled stress-test variants."""

    defaults: dict[str, Any] = {
        "data": data_root,
        "method": "global-assignment",
        "reference_kind": "manual-gt",
        "input_format": "suite2p",
        "include_non_cells": True,
        "weighted_masks": True,
        "transform_type": "fov-affine",
        "max_gap": 2,
        "format": "csv",
    }
    if reference_root is not None:
        defaults["reference"] = reference_root
    runs = []
    for transform in ("fov-affine", "local-affine-grid", "bspline"):
        for cost in ("registered-soft-iou", "roi-aware-shifted", "calibrated"):
            run: dict[str, Any] = {
                "name": f"stress-{transform}-{cost}",
                "transform_type": transform,
                "cost": cost,
                "output": f"{output_root}/stress-{transform}-{cost}.csv",
            }
            if cost == "calibrated":
                run["split"] = "leave-one-subject-out"
            runs.append(run)
    return {
        "defaults": defaults,
        "runs": runs,
        "comparisons": [
            {
                "name": "stress-summary",
                "inputs": {run["name"]: run["name"] for run in runs},
                "output": f"{output_root}/stress-summary.md",
                "highlight_best": True,
            }
        ],
    }


def precision_recall_threshold_table(
    probabilities: Any,
    labels: Any,
    *,
    thresholds: Sequence[float] | None = None,
) -> list[dict[str, float | int]]:
    """Return precision/recall/F1 over probability rejection thresholds."""

    probs = np.asarray(probabilities, dtype=float).reshape(-1)
    y = np.asarray(labels, dtype=int).reshape(-1)
    if probs.shape != y.shape:
        raise ValueError("probabilities and labels must have the same length")
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        pred = probs >= float(threshold)
        positive = y != 0
        tp = int(np.count_nonzero(pred & positive))
        fp = int(np.count_nonzero(pred & ~positive))
        fn = int(np.count_nonzero(~pred & positive))
        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, tp + fn)
        rows.append(
            {
                "threshold": float(threshold),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": precision,
                "recall": recall,
                "f1": _ratio(2.0 * precision * recall, precision + recall),
            }
        )
    return rows


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bayescatrack.experiments.advanced_improvement_workbench",
        description="Advanced diagnostics and manifest helpers for BayesCaTrack result improvement.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    active = subparsers.add_parser("active-labels", help="Rank edge rows for additional manual labeling")
    active.add_argument("--input", required=True, type=Path)
    active.add_argument("--output", required=True, type=Path)
    active.add_argument("--max-rows", type=int, default=500)

    stratify = subparsers.add_parser("stratify", help="Aggregate benchmark metrics by metadata fields")
    stratify.add_argument("--input", required=True, type=Path)
    stratify.add_argument("--output", required=True, type=Path)
    stratify.add_argument("--group-field", action="append", required=True)
    stratify.add_argument("--metric", action="append", required=True)

    stress = subparsers.add_parser("stress-manifest", help="Write a synthetic stress-test benchmark manifest")
    stress.add_argument("--data-root", required=True)
    stress.add_argument("--output-root", required=True)
    stress.add_argument("--reference-root", default=None)
    stress.add_argument("--output", required=True, type=Path)

    pr = subparsers.add_parser("pr-table", help="Build a precision/recall table from probability-label CSV")
    pr.add_argument("--input", required=True, type=Path)
    pr.add_argument("--output", required=True, type=Path)
    pr.add_argument("--probability-column", default="probability")
    pr.add_argument("--label-column", default="label")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "active-labels":
        rows = read_csv_rows(args.input)
        selected = select_active_label_candidates(
            rows, config=ActiveLabelConfig(max_rows=args.max_rows)
        )
        write_csv_rows(selected, args.output)
        return 0
    if args.command == "stratify":
        rows = read_csv_rows(args.input)
        summary = stratified_metric_summary(
            rows,
            config=StratifiedMetricConfig(
                group_fields=tuple(args.group_field),
                metric_fields=tuple(args.metric),
            ),
        )
        write_csv_rows(summary, args.output)
        return 0
    if args.command == "stress-manifest":
        manifest = synthetic_stress_manifest(
            data_root=args.data_root,
            output_root=args.output_root,
            reference_root=args.reference_root,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return 0
    if args.command == "pr-table":
        rows = read_csv_rows(args.input)
        probabilities = [_safe_float(row.get(args.probability_column), np.nan) for row in rows]
        labels = [int(_safe_float(row.get(args.label_column), 0.0)) for row in rows]
        write_csv_rows(precision_recall_threshold_table(probabilities, labels), args.output)
        return 0
    raise ValueError(f"Unsupported command {args.command!r}")


def _safe_float(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    return numeric if np.isfinite(numeric) else float(default)


def _ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
