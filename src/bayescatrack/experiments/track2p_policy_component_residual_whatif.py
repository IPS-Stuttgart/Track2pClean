"""Rank single-edit what-if repairs for ComponentCleanup residual errors.

The residual audit shows that the remaining errors after Track2pPolicy
component cleanup are now few enough that blind feature stacking is less useful
than asking which official error would move the metrics if it could be repaired.
This module is intentionally diagnostic and label-aware: it does not produce a
paper-facing tracker row.  Instead, it consumes the residual audit CSV and a
ComponentCleanup benchmark CSV, then scores the oracle effect of one local edit
family at a time:

* add a residual pairwise false negative as a true-positive edge;
* veto a residual pairwise false positive;
* recover one complete-track false negative;
* remove one complete-track false positive.

The resulting table ranks the next implementation target by exact micro-F1
delta and support bucket, e.g. Track2p-supported adjacent FNs versus Bayes-only
FPs.  It is a searchlight for the next non-oracle method, not the method itself.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

TRACK2P_POLICY_COMPONENT_RESIDUAL_WHATIF_METHOD = (
    "track2p-policy-component-residual-whatif"
)

_PAIRWISE_FIELDS = (
    "pairwise_true_positives",
    "pairwise_false_positives",
    "pairwise_false_negatives",
)
_COMPLETE_FIELDS = (
    "complete_track_true_positives",
    "complete_track_false_positives",
    "complete_track_false_negatives",
)
_SUPPORT_FLAGS = (
    "is_track2p_supported",
    "is_policy_supported",
    "is_gap_rescue_supported",
    "is_component_cleanup_affected",
)


@dataclass(frozen=True)
class MicroCounts:
    """Duplicate-aware micro counts for the two official objectives."""

    pairwise_tp: int
    pairwise_fp: int
    pairwise_fn: int
    complete_tp: int
    complete_fp: int
    complete_fn: int

    @property
    def pairwise_f1(self) -> float:
        return _f1(self.pairwise_tp, self.pairwise_fp, self.pairwise_fn)

    @property
    def complete_track_f1(self) -> float:
        return _f1(self.complete_tp, self.complete_fp, self.complete_fn)

    def apply(
        self,
        *,
        pairwise_tp_delta: int = 0,
        pairwise_fp_delta: int = 0,
        pairwise_fn_delta: int = 0,
        complete_tp_delta: int = 0,
        complete_fp_delta: int = 0,
        complete_fn_delta: int = 0,
    ) -> MicroCounts:
        """Return counts after a hypothetical single edit."""

        return MicroCounts(
            pairwise_tp=max(0, self.pairwise_tp + int(pairwise_tp_delta)),
            pairwise_fp=max(0, self.pairwise_fp + int(pairwise_fp_delta)),
            pairwise_fn=max(0, self.pairwise_fn + int(pairwise_fn_delta)),
            complete_tp=max(0, self.complete_tp + int(complete_tp_delta)),
            complete_fp=max(0, self.complete_fp + int(complete_fp_delta)),
            complete_fn=max(0, self.complete_fn + int(complete_fn_delta)),
        )


def residual_whatif_rows(
    residual_rows: Sequence[Mapping[str, Any]],
    base_counts: MicroCounts,
) -> list[dict[str, float | int | str]]:
    """Return one oracle single-edit what-if row per residual error.

    The function deliberately keeps pairwise and complete-track edits separate.
    A complete-track edit is therefore a bound on the objective movement, not a
    concrete track-matrix mutation.  Concrete methods should be implemented only
    after this table identifies a support bucket worth targeting.
    """

    output: list[dict[str, float | int | str]] = []
    for index, residual in enumerate(residual_rows):
        edit = _edit_for_error_type(str(residual.get("error_type", "")))
        if edit is None:
            continue
        edited = base_counts.apply(**edit["deltas"])
        output.append(
            _candidate_row(
                residual,
                candidate_index=index,
                edit_type=str(edit["edit_type"]),
                base_counts=base_counts,
                edited_counts=edited,
                deltas=edit["deltas"],
            )
        )
    output.sort(
        key=lambda row: (
            float(row["complete_track_f1_delta"]),
            float(row["pairwise_f1_delta"]),
            int(row["is_track2p_supported"]),
            str(row["reason_bucket"]),
            str(row["track_id_or_edge"]),
        ),
        reverse=True,
    )
    return output


def residual_whatif_summary_rows(
    candidate_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, float | int | str]]:
    """Aggregate candidate rows by edit/support/reason bucket."""

    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        groups[
            (
                str(row.get("edit_type", "")),
                str(row.get("support_bucket", "")),
                str(row.get("reason_bucket", "")),
            )
        ].append(row)

    summary: list[dict[str, float | int | str]] = []
    for (edit_type, support_bucket, reason_bucket), rows in groups.items():
        best_pairwise = max(float(row["pairwise_f1_delta"]) for row in rows)
        best_complete = max(float(row["complete_track_f1_delta"]) for row in rows)
        best_pairwise_row = max(rows, key=lambda row: float(row["pairwise_f1_delta"]))
        best_complete_row = max(
            rows, key=lambda row: float(row["complete_track_f1_delta"])
        )
        summary.append(
            {
                "edit_type": edit_type,
                "support_bucket": support_bucket,
                "reason_bucket": reason_bucket,
                "candidate_count": int(len(rows)),
                "track2p_supported_count": int(
                    sum(_boolish(row.get("is_track2p_supported", 0)) for row in rows)
                ),
                "policy_supported_count": int(
                    sum(_boolish(row.get("is_policy_supported", 0)) for row in rows)
                ),
                "gap_rescue_supported_count": int(
                    sum(_boolish(row.get("is_gap_rescue_supported", 0)) for row in rows)
                ),
                "component_cleanup_affected_count": int(
                    sum(
                        _boolish(row.get("is_component_cleanup_affected", 0))
                        for row in rows
                    )
                ),
                "best_pairwise_f1_delta": best_pairwise,
                "best_complete_track_f1_delta": best_complete,
                "best_pairwise_candidate": str(best_pairwise_row["track_id_or_edge"]),
                "best_complete_candidate": str(best_complete_row["track_id_or_edge"]),
            }
        )
    summary.sort(
        key=lambda row: (
            float(row["best_complete_track_f1_delta"]),
            float(row["best_pairwise_f1_delta"]),
            int(row["candidate_count"]),
        ),
        reverse=True,
    )
    return summary


def load_base_counts(path: Path) -> MicroCounts:
    """Load and sum official count columns from a benchmark CSV."""

    rows = _read_csv_rows(path)
    missing = [
        field
        for field in (*_PAIRWISE_FIELDS, *_COMPLETE_FIELDS)
        if field not in rows[0]
    ]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Base benchmark CSV is missing count columns: {joined}")
    totals = {field: 0 for field in (*_PAIRWISE_FIELDS, *_COMPLETE_FIELDS)}
    for row in rows:
        for field in totals:
            totals[field] += _int_value(row[field])
    return MicroCounts(
        pairwise_tp=totals["pairwise_true_positives"],
        pairwise_fp=totals["pairwise_false_positives"],
        pairwise_fn=totals["pairwise_false_negatives"],
        complete_tp=totals["complete_track_true_positives"],
        complete_fp=totals["complete_track_false_positives"],
        complete_fn=totals["complete_track_false_negatives"],
    )


def write_rows(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    output_format: Literal["csv", "json"] = "csv",
) -> None:
    """Write rows as CSV or JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output_path.write_text(
            json.dumps(list(rows), indent=2) + "\n", encoding="utf-8"
        )
        return
    fieldnames = _preferred_fieldnames(rows)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the residual what-if command-line parser."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark track2p-policy-component-residual-whatif",
        description=(
            "Rank oracle single-edit repairs for the residual errors remaining "
            "after Track2pPolicy component cleanup."
        ),
    )
    parser.add_argument(
        "--residual-audit",
        type=Path,
        required=True,
        help="CSV produced by track2p-policy-component-residual-audit.",
    )
    parser.add_argument(
        "--base-benchmark",
        type=Path,
        default=None,
        help=(
            "ComponentCleanup benchmark CSV with official TP/FP/FN columns. "
            "If omitted, all --base-* count arguments must be provided."
        ),
    )
    parser.add_argument("--base-pairwise-tp", type=int, default=None)
    parser.add_argument("--base-pairwise-fp", type=int, default=None)
    parser.add_argument("--base-pairwise-fn", type=int, default=None)
    parser.add_argument("--base-complete-tp", type=int, default=None)
    parser.add_argument("--base-complete-fp", type=int, default=None)
    parser.add_argument("--base-complete-fn", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--format", choices=("csv", "json"), default="csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run residual what-if ranking from the CLI."""

    args = build_arg_parser().parse_args(argv)
    base_counts = _base_counts_from_args(args)
    residual_rows = _read_csv_rows(args.residual_audit)
    candidates = residual_whatif_rows(residual_rows, base_counts)
    write_rows(candidates, args.output, output_format=args.format)
    if args.summary_output is not None:
        write_rows(
            residual_whatif_summary_rows(candidates),
            args.summary_output,
            output_format=args.format,
        )
    return 0


def _base_counts_from_args(args: argparse.Namespace) -> MicroCounts:
    if args.base_benchmark is not None:
        return load_base_counts(args.base_benchmark)
    required = {
        "--base-pairwise-tp": args.base_pairwise_tp,
        "--base-pairwise-fp": args.base_pairwise_fp,
        "--base-pairwise-fn": args.base_pairwise_fn,
        "--base-complete-tp": args.base_complete_tp,
        "--base-complete-fp": args.base_complete_fp,
        "--base-complete-fn": args.base_complete_fn,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            "Provide --base-benchmark or explicit base counts. " f"Missing: {joined}"
        )
    return MicroCounts(
        pairwise_tp=int(args.base_pairwise_tp),
        pairwise_fp=int(args.base_pairwise_fp),
        pairwise_fn=int(args.base_pairwise_fn),
        complete_tp=int(args.base_complete_tp),
        complete_fp=int(args.base_complete_fp),
        complete_fn=int(args.base_complete_fn),
    )


def _edit_for_error_type(error_type: str) -> dict[str, Any] | None:
    if error_type == "pairwise_fn":
        return {
            "edit_type": "add_pairwise_fn_as_tp",
            "deltas": {
                "pairwise_tp_delta": 1,
                "pairwise_fn_delta": -1,
            },
        }
    if error_type == "pairwise_fp":
        return {
            "edit_type": "remove_pairwise_fp",
            "deltas": {"pairwise_fp_delta": -1},
        }
    if error_type == "complete_fn":
        return {
            "edit_type": "recover_complete_track_fn",
            "deltas": {
                "complete_tp_delta": 1,
                "complete_fn_delta": -1,
            },
        }
    if error_type == "complete_fp":
        return {
            "edit_type": "remove_complete_track_fp",
            "deltas": {"complete_fp_delta": -1},
        }
    return None


def _candidate_row(
    residual: Mapping[str, Any],
    *,
    candidate_index: int,
    edit_type: str,
    base_counts: MicroCounts,
    edited_counts: MicroCounts,
    deltas: Mapping[str, int],
) -> dict[str, float | int | str]:
    output: dict[str, float | int | str] = {
        "candidate_index": int(candidate_index),
        "edit_type": edit_type,
        "subject": str(residual.get("subject", "")),
        "error_type": str(residual.get("error_type", "")),
        "track_id_or_edge": str(residual.get("track_id_or_edge", "")),
        "reason_bucket": str(residual.get("reason_bucket", "")),
        "support_bucket": _support_bucket(residual),
        "base_pairwise_f1": base_counts.pairwise_f1,
        "new_pairwise_f1": edited_counts.pairwise_f1,
        "pairwise_f1_delta": edited_counts.pairwise_f1 - base_counts.pairwise_f1,
        "base_complete_track_f1": base_counts.complete_track_f1,
        "new_complete_track_f1": edited_counts.complete_track_f1,
        "complete_track_f1_delta": (
            edited_counts.complete_track_f1 - base_counts.complete_track_f1
        ),
        "new_pairwise_tp": int(edited_counts.pairwise_tp),
        "new_pairwise_fp": int(edited_counts.pairwise_fp),
        "new_pairwise_fn": int(edited_counts.pairwise_fn),
        "new_complete_track_tp": int(edited_counts.complete_tp),
        "new_complete_track_fp": int(edited_counts.complete_fp),
        "new_complete_track_fn": int(edited_counts.complete_fn),
    }
    for flag in _SUPPORT_FLAGS:
        output[flag] = int(_boolish(residual.get(flag, 0)))
    for key in (
        "session_a",
        "session_b",
        "roi_a",
        "roi_b",
        "registered_iou",
        "centroid_distance",
        "area_ratio",
        "row_rank",
        "column_rank",
        "row_margin",
        "column_margin",
        "threshold_margin",
        "cell_probability_a",
        "cell_probability_b",
        "component_id",
        "component_size",
        "complete_track_status",
        "nearest_gt_track_id",
        "nearest_predicted_track_id",
    ):
        if key in residual:
            output[key] = _typed_value(residual[key])
    for key in (
        "pairwise_tp_delta",
        "pairwise_fp_delta",
        "pairwise_fn_delta",
        "complete_tp_delta",
        "complete_fp_delta",
        "complete_fn_delta",
    ):
        output[key] = int(deltas.get(key, 0))
    return output


def _support_bucket(row: Mapping[str, Any]) -> str:
    track2p = _boolish(row.get("is_track2p_supported", 0))
    policy = _boolish(row.get("is_policy_supported", 0))
    gap = _boolish(row.get("is_gap_rescue_supported", 0))
    cleanup = _boolish(row.get("is_component_cleanup_affected", 0))
    labels: list[str] = []
    if track2p:
        labels.append("track2p")
    if policy:
        labels.append("policy")
    if gap:
        labels.append("gap")
    if cleanup:
        labels.append("cleanup-affected")
    return "+".join(labels) if labels else "unsupported"


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    return rows


def _preferred_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    preferred = [
        "candidate_index",
        "edit_type",
        "support_bucket",
        "reason_bucket",
        "subject",
        "error_type",
        "track_id_or_edge",
        "pairwise_f1_delta",
        "complete_track_f1_delta",
        "new_pairwise_f1",
        "new_complete_track_f1",
        "new_pairwise_tp",
        "new_pairwise_fp",
        "new_pairwise_fn",
        "new_complete_track_tp",
        "new_complete_track_fp",
        "new_complete_track_fn",
    ]
    keys = sorted({key for row in rows for key in row})
    return [key for key in preferred if key in keys] + [
        key for key in keys if key not in preferred
    ]


def _f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * int(tp) + int(fp) + int(fn)
    if denominator <= 0:
        return 0.0
    return float(2 * int(tp) / denominator)


def _boolish(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _int_value(value: Any) -> int:
    if isinstance(value, str):
        value = value.strip().replace(",", ".")
    return int(float(value))


def _typed_value(value: Any) -> float | int | str:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text == "":
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return int(number)
    return float(number)
