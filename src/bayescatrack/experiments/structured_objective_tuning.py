"""Select benchmark variants by track-level objectives rather than pairwise loss."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def select_best_variants_by_track_metric(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str = "complete_track_f1",
    group_by: str = "variant",
    tie_breakers: Sequence[str] = ("pairwise_f1", "pairwise_precision"),
) -> list[dict[str, float | int | str]]:
    """Return variants ranked by a held-in track-level metric."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_by, ""))].append(row)
    scored = [
        _score_group(name, group, metric=metric, tie_breakers=tie_breakers)
        for name, group in grouped.items()
        if name
    ]
    scored.sort(
        key=lambda row: (
            -float(row[f"mean_{metric}"]),
            *(-float(row[f"mean_{tie}"]) for tie in tie_breakers),
            str(row[group_by]),
        )
    )
    for rank, row in enumerate(scored, start=1):
        row["rank"] = rank
    return scored


def _score_group(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    tie_breakers: Sequence[str],
) -> dict[str, float | int | str]:
    out: dict[str, float | int | str] = {
        "variant": name,
        "rows": int(len(rows)),
        f"mean_{metric}": _mean(rows, metric),
        f"median_{metric}": _median(rows, metric),
        f"min_{metric}": _min(rows, metric),
    }
    for tie in tie_breakers:
        out[f"mean_{tie}"] = _mean(rows, tie)
    return out


def _values(rows: Sequence[Mapping[str, Any]], field: str) -> np.ndarray:
    vals = []
    for row in rows:
        try:
            value = float(row.get(field, "nan"))
        except (TypeError, ValueError):
            value = float("nan")
        if np.isfinite(value):
            vals.append(value)
    return np.asarray(vals, dtype=float)


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    values = _values(rows, field)
    return float(np.mean(values)) if values.size else float("nan")


def _median(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    values = _values(rows, field)
    return float(np.median(values)) if values.size else float("nan")


def _min(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    values = _values(rows, field)
    return float(np.min(values)) if values.size else float("nan")


def _load_csv(paths: Sequence[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark select-structured-objective",
        description="Rank benchmark variants by complete-track or other structured metrics.",
    )
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--metric", default="complete_track_f1")
    parser.add_argument("--group-by", default="variant")
    parser.add_argument("--tie-breaker", action="append", default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rows = select_best_variants_by_track_metric(
        _load_csv(args.csv),
        metric=args.metric,
        group_by=args.group_by,
        tie_breakers=tuple(args.tie_breaker or ("pairwise_f1", "pairwise_precision")),
    )
    payload = json.dumps(rows, indent=2) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
