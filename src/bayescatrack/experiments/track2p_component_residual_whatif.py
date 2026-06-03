"""Rank residual official-error edits after Track2pPolicy component cleanup.

This module is intentionally diagnostic: it does not change a prediction matrix.
It consumes the residual audit CSV produced after the current lead row and scores
single-edit interventions that are likely to move the official metrics.  The
output is meant to decide which small repair family should be implemented next
instead of opening another broad feature sweep.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


_PAIRWISE_TP = "pairwise_true_positives"
_PAIRWISE_FP = "pairwise_false_positives"
_PAIRWISE_FN = "pairwise_false_negatives"
_COMPLETE_TP = "complete_track_true_positives"
_COMPLETE_FP = "complete_track_false_positives"
_COMPLETE_FN = "complete_track_false_negatives"


@dataclass(frozen=True)
class OfficialCounts:
    """Micro-counts used by the Track2p benchmark comparison."""

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

    def with_delta(
        self,
        *,
        pairwise_tp_delta: int = 0,
        pairwise_fp_delta: int = 0,
        pairwise_fn_delta: int = 0,
        complete_tp_delta: int = 0,
        complete_fp_delta: int = 0,
        complete_fn_delta: int = 0,
    ) -> "OfficialCounts":
        """Return counts after applying an abstract what-if edit."""
        return OfficialCounts(
            pairwise_tp=max(0, self.pairwise_tp + int(pairwise_tp_delta)),
            pairwise_fp=max(0, self.pairwise_fp + int(pairwise_fp_delta)),
            pairwise_fn=max(0, self.pairwise_fn + int(pairwise_fn_delta)),
            complete_tp=max(0, self.complete_tp + int(complete_tp_delta)),
            complete_fp=max(0, self.complete_fp + int(complete_fp_delta)),
            complete_fn=max(0, self.complete_fn + int(complete_fn_delta)),
        )


@dataclass(frozen=True)
class WhatIfCandidate:
    """One residual-error edit and its expected metric effect."""

    candidate_id: str
    edit_type: str
    subject: str
    reason_bucket: str
    pairwise_tp_delta: int = 0
    pairwise_fp_delta: int = 0
    pairwise_fn_delta: int = 0
    complete_tp_delta: int = 0
    complete_fp_delta: int = 0
    complete_fn_delta: int = 0
    source_row: Mapping[str, str] | None = None

    def scored(self, baseline: OfficialCounts) -> dict[str, Any]:
        """Return this candidate as a serializable scored row."""
        after = baseline.with_delta(
            pairwise_tp_delta=self.pairwise_tp_delta,
            pairwise_fp_delta=self.pairwise_fp_delta,
            pairwise_fn_delta=self.pairwise_fn_delta,
            complete_tp_delta=self.complete_tp_delta,
            complete_fp_delta=self.complete_fp_delta,
            complete_fn_delta=self.complete_fn_delta,
        )
        row = {
            "candidate_id": self.candidate_id,
            "edit_type": self.edit_type,
            "subject": self.subject,
            "reason_bucket": self.reason_bucket,
            "pairwise_tp_delta": self.pairwise_tp_delta,
            "pairwise_fp_delta": self.pairwise_fp_delta,
            "pairwise_fn_delta": self.pairwise_fn_delta,
            "complete_tp_delta": self.complete_tp_delta,
            "complete_fp_delta": self.complete_fp_delta,
            "complete_fn_delta": self.complete_fn_delta,
            "baseline_pairwise_f1": baseline.pairwise_f1,
            "new_pairwise_f1": after.pairwise_f1,
            "pairwise_f1_delta": after.pairwise_f1 - baseline.pairwise_f1,
            "baseline_complete_track_f1": baseline.complete_track_f1,
            "new_complete_track_f1": after.complete_track_f1,
            "complete_track_f1_delta": (
                after.complete_track_f1 - baseline.complete_track_f1
            ),
            "new_pairwise_tp": after.pairwise_tp,
            "new_pairwise_fp": after.pairwise_fp,
            "new_pairwise_fn": after.pairwise_fn,
            "new_complete_tp": after.complete_tp,
            "new_complete_fp": after.complete_fp,
            "new_complete_fn": after.complete_fn,
        }
        if self.source_row:
            for key in (
                "error_type",
                "session_a",
                "session_b",
                "roi_a",
                "roi_b",
                "track2p_supported",
                "policy_supported",
                "gap_rescue_supported",
                "component_status",
                "component_id",
            ):
                if key in self.source_row:
                    row[key] = self.source_row[key]
        return row


def load_baseline_counts(path: Path) -> OfficialCounts:
    """Load aggregate micro-counts from a benchmark row CSV."""
    rows = _read_csv_rows(path)
    if not rows:
        raise ValueError(f"baseline CSV {path} does not contain any rows")
    return OfficialCounts(
        pairwise_tp=_sum_int_column(rows, _PAIRWISE_TP),
        pairwise_fp=_sum_int_column(rows, _PAIRWISE_FP),
        pairwise_fn=_sum_int_column(rows, _PAIRWISE_FN),
        complete_tp=_sum_int_column(rows, _COMPLETE_TP),
        complete_fp=_sum_int_column(rows, _COMPLETE_FP),
        complete_fn=_sum_int_column(rows, _COMPLETE_FN),
    )


def discover_candidates(
    rows: Sequence[Mapping[str, str]],
) -> list[WhatIfCandidate]:
    """Return plausible single-edit repairs from residual audit rows.

    The rules are deliberately conservative and diagnostic.  They estimate the
    metric effect of an edit family; the edit still needs a real prediction-matrix
    implementation and rerun before it can become a paper-facing method.
    """
    candidates: list[WhatIfCandidate] = []
    seen: set[tuple[str, str, str, str, str, str]] = set()
    for row_index, row in enumerate(rows):
        error_type = _normalized_field(row, "error_type", "type", "error")
        reason = _normalized_field(row, "reason_bucket", "reason", "bucket")
        subject = _field(row, "subject", default="")
        candidate_key = _candidate_key(row, row_index=row_index)
        dedupe_key = (
            candidate_key,
            error_type,
            _field(row, "session_a", default=""),
            _field(row, "session_b", default=""),
            _field(row, "roi_a", default=""),
            _field(row, "roi_b", default=""),
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        track2p_supported = _truthy_field(
            row, "track2p_supported", "is_track2p_supported"
        )
        component_supported = _truthy_field(
            row,
            "component_cleanup_supported",
            "component_supported",
            "is_component_cleanup_supported",
        )
        gap_supported = _truthy_field(
            row, "gap_rescue_supported", "is_gap_rescue_supported"
        )

        if _is_pairwise_fn(error_type) and track2p_supported and not component_supported:
            candidates.append(
                WhatIfCandidate(
                    candidate_id=f"teacher-fn-rescue-{candidate_key}",
                    edit_type="track2p_supported_adjacent_fn_rescue",
                    subject=subject,
                    reason_bucket=reason,
                    pairwise_tp_delta=1,
                    pairwise_fn_delta=-1,
                    source_row=row,
                )
            )
        if _is_pairwise_fp(error_type) and not track2p_supported:
            candidates.append(
                WhatIfCandidate(
                    candidate_id=f"bayes-fp-veto-{candidate_key}",
                    edit_type="bayes_only_pairwise_fp_veto",
                    subject=subject,
                    reason_bucket=reason,
                    pairwise_fp_delta=-1,
                    source_row=row,
                )
            )
        if _is_complete_fp(error_type):
            candidates.append(
                WhatIfCandidate(
                    candidate_id=f"complete-fp-split-{candidate_key}",
                    edit_type="complete_fp_component_split_or_veto",
                    subject=subject,
                    reason_bucket=reason,
                    complete_fp_delta=-1,
                    source_row=row,
                )
            )
        if _is_complete_fn(error_type) and "seed" in reason:
            candidates.append(
                WhatIfCandidate(
                    candidate_id=f"missing-seed-recovery-{candidate_key}",
                    edit_type="missing_seed_complete_fn_recovery",
                    subject=subject,
                    reason_bucket=reason,
                    complete_tp_delta=1,
                    complete_fn_delta=-1,
                    source_row=row,
                )
            )
        elif _is_complete_fn(error_type) and track2p_supported and not gap_supported:
            candidates.append(
                WhatIfCandidate(
                    candidate_id=f"complete-fn-recovery-{candidate_key}",
                    edit_type="track2p_supported_complete_fn_recovery",
                    subject=subject,
                    reason_bucket=reason,
                    complete_tp_delta=1,
                    complete_fn_delta=-1,
                    source_row=row,
                )
            )
    return candidates


def score_candidates(
    candidates: Sequence[WhatIfCandidate], baseline: OfficialCounts
) -> list[dict[str, Any]]:
    """Score and rank candidates by complete-track then pairwise improvement."""
    rows = [candidate.scored(baseline) for candidate in candidates]
    return sorted(
        rows,
        key=lambda row: (
            float(row["complete_track_f1_delta"]),
            float(row["pairwise_f1_delta"]),
            int(row["pairwise_tp_delta"]),
            -int(row["pairwise_fp_delta"]),
            str(row["candidate_id"]),
        ),
        reverse=True,
    )


def summarize(scored_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize scored candidates by edit family."""
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in scored_rows:
        grouped.setdefault(str(row["edit_type"]), []).append(row)
    output: list[dict[str, Any]] = []
    for edit_type, rows in sorted(grouped.items()):
        best = max(
            rows,
            key=lambda row: (
                float(row["complete_track_f1_delta"]),
                float(row["pairwise_f1_delta"]),
            ),
        )
        output.append(
            {
                "edit_type": edit_type,
                "candidates": len(rows),
                "subjects": ",".join(sorted({str(row.get("subject", "")) for row in rows})),
                "best_pairwise_f1_delta": best["pairwise_f1_delta"],
                "best_complete_track_f1_delta": best["complete_track_f1_delta"],
                "best_candidate_id": best["candidate_id"],
            }
        )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m bayescatrack.experiments.track2p_component_residual_whatif",
        description=(
            "Rank residual official-error edits after Track2pPolicy component "
            "cleanup. This is a diagnostic pre-screen for the next small repair."
        ),
    )
    parser.add_argument("--residual-audit", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=0)
    args = parser.parse_args(argv)

    baseline = load_baseline_counts(args.baseline_csv)
    residual_rows = _read_csv_rows(args.residual_audit)
    scored = score_candidates(discover_candidates(residual_rows), baseline)
    if args.top_n > 0:
        scored = scored[: args.top_n]
    _write_csv(args.output, scored)
    if args.summary_output is not None:
        _write_csv(args.summary_output, summarize(scored))
    return 0


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sum_int_column(rows: Sequence[Mapping[str, str]], column: str) -> int:
    if column not in rows[0]:
        raise ValueError(f"baseline CSV is missing required column {column!r}")
    total = 0
    for row in rows:
        total += int(float(str(row.get(column, "0") or "0")))
    return total


def _candidate_key(row: Mapping[str, str], *, row_index: int) -> str:
    parts = [
        _field(row, "subject", default="subject"),
        _field(row, "session_a", default="sa"),
        _field(row, "session_b", default="sb"),
        _field(row, "roi_a", default="ra"),
        _field(row, "roi_b", default="rb"),
    ]
    if all(part not in {"", "subject", "sa", "sb", "ra", "rb"} for part in parts):
        return "-".join(parts)
    return str(row_index)


def _field(row: Mapping[str, str], *names: str, default: str = "") -> str:
    for name in names:
        if name in row and row[name] not in {None, ""}:
            return str(row[name])
    return default


def _normalized_field(row: Mapping[str, str], *names: str) -> str:
    return _field(row, *names).strip().lower().replace("-", "_").replace(" ", "_")


def _truthy_field(row: Mapping[str, str], *names: str) -> bool:
    value = _normalized_field(row, *names)
    return value in {"1", "true", "yes", "y", "supported"}


def _is_pairwise_fn(error_type: str) -> bool:
    return "pairwise" in error_type and (
        "false_negative" in error_type or error_type.endswith("fn") or "_fn" in error_type
    )


def _is_pairwise_fp(error_type: str) -> bool:
    return "pairwise" in error_type and (
        "false_positive" in error_type or error_type.endswith("fp") or "_fp" in error_type
    )


def _is_complete_fp(error_type: str) -> bool:
    return "complete" in error_type and (
        "false_positive" in error_type or error_type.endswith("fp") or "_fp" in error_type
    )


def _is_complete_fn(error_type: str) -> bool:
    return "complete" in error_type and (
        "false_negative" in error_type or error_type.endswith("fn") or "_fn" in error_type
    )


def _f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * int(tp) + int(fp) + int(fn)
    if denominator <= 0:
        return 0.0
    return float(2 * int(tp) / denominator)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
