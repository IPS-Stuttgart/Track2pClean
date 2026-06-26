"""Promotion gate for identity-history scan-pruning evidence.

The scan-pruning add-on should not be promoted from a benchmark table alone.  It
must first show stable complete-history gain against the frozen greedy controls
and then pass a label-free exposure audit that records scan-motion history risk
without reading manual-GT labels.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bayescatrack.experiments.full_mht_identity_history_scan_pruning_decision import (
    evaluate_identity_history_scan_pruning_decision,
    load_comparison_rows,
)


@dataclass(frozen=True)
class ScanPruningPromotionConfig:
    """Predeclared exposure limits for scan-history pruning promotion."""

    max_selected_non_prior_edges_per_subject: int = 3
    max_total_non_prior_edges: int = 10
    max_scan_weighted_risk_per_subject: float = 10.0
    max_total_scan_weighted_risk: float = 25.0


def load_exposure_rows(path: Path) -> list[dict[str, str]]:
    """Load label-free exposure-audit rows."""

    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No exposure rows found in {path}")
    if "subject" not in rows[0]:
        raise ValueError(f"Exposure CSV {path} is missing a 'subject' column")
    return rows


def evaluate_scan_pruning_promotion(
    comparison_rows: Sequence[Mapping[str, Any]],
    exposure_rows: Sequence[Mapping[str, Any]],
    *,
    config: ScanPruningPromotionConfig | None = None,
) -> dict[str, Any]:
    """Combine scan-pruning benchmark decision and exposure evidence."""

    cfg = config or ScanPruningPromotionConfig()
    benchmark = evaluate_identity_history_scan_pruning_decision(comparison_rows)
    exposure = evaluate_scan_pruning_exposure_gate(exposure_rows, config=cfg)
    benchmark_result = str(benchmark.get("scan_pruning_result", "incomplete"))
    exposure_result = str(exposure.get("exposure_result", "incomplete"))

    if benchmark.get("status") != "complete":
        status = "incomplete"
        recommendation = "rerun identity-history scan-pruning benchmark manifest"
    elif exposure.get("status") != "complete":
        status = "incomplete"
        recommendation = "rerun scan-pruning label-free exposure audit"
    elif (
        benchmark_result == "scan_pruning_stable_complete_history_gain"
        and exposure_result == "bounded_exposure"
    ):
        status = "promotable_after_review"
        recommendation = "promote scan-pruning only with recorded manifest and exposure outputs"
    elif benchmark_result == "scan_pruning_stable_complete_history_gain":
        status = "not_promotable_broad_exposure"
        recommendation = "keep exploratory; scan-pruning exposure is too broad"
    elif exposure_result == "bounded_exposure":
        status = "not_promotable_no_stable_gain"
        recommendation = "keep exploratory; benchmark gain is absent or knife-edge"
    else:
        status = "not_promotable"
        recommendation = "keep exploratory; benchmark and exposure gates failed"

    return {
        "status": status,
        "recommendation": recommendation,
        "benchmark_result": benchmark_result,
        "exposure_result": exposure_result,
        "benchmark": benchmark,
        "exposure": exposure,
    }


def evaluate_scan_pruning_exposure_gate(
    exposure_rows: Sequence[Mapping[str, Any]],
    *,
    config: ScanPruningPromotionConfig | None = None,
) -> dict[str, Any]:
    """Evaluate whether scan-pruning exposure remains bounded."""

    cfg = config or ScanPruningPromotionConfig()
    all_row = _all_row(exposure_rows)
    if all_row is None:
        return {
            "status": "incomplete",
            "exposure_result": "missing_all_row",
            "recommendation": "rerun exposure audit with aggregate ALL row",
        }

    required = (
        "history_scan_motion_history_weighted_risk",
        "max_scan_motion_history_weighted_risk_per_subject",
        "max_selected_non_prior_edges_per_subject",
        "history_selected_non_prior_edges",
    )
    missing = [key for key in required if key not in all_row]
    if missing:
        return {
            "status": "incomplete",
            "exposure_result": "missing_scan_pruning_columns",
            "missing_columns": missing,
            "recommendation": "rerun exposure audit with --scan-motion-history-weight",
        }

    max_non_prior = _int_metric(all_row, "max_selected_non_prior_edges_per_subject")
    total_non_prior = _int_metric(all_row, "history_selected_non_prior_edges")
    max_weighted_risk = _float_metric(
        all_row,
        "max_scan_motion_history_weighted_risk_per_subject",
    )
    total_weighted_risk = _float_metric(
        all_row,
        "history_scan_motion_history_weighted_risk",
    )

    failures: list[str] = []
    if max_non_prior > int(cfg.max_selected_non_prior_edges_per_subject):
        failures.append("max_selected_non_prior_edges_per_subject")
    if total_non_prior > int(cfg.max_total_non_prior_edges):
        failures.append("history_selected_non_prior_edges")
    if max_weighted_risk > float(cfg.max_scan_weighted_risk_per_subject):
        failures.append("max_scan_motion_history_weighted_risk_per_subject")
    if total_weighted_risk > float(cfg.max_total_scan_weighted_risk):
        failures.append("history_scan_motion_history_weighted_risk")

    return {
        "status": "complete",
        "exposure_result": "bounded_exposure" if not failures else "broad_exposure",
        "failed_limits": failures,
        "max_selected_non_prior_edges_per_subject": int(max_non_prior),
        "history_selected_non_prior_edges": int(total_non_prior),
        "max_scan_motion_history_weighted_risk_per_subject": float(max_weighted_risk),
        "history_scan_motion_history_weighted_risk": float(total_weighted_risk),
        "limit_max_selected_non_prior_edges_per_subject": int(
            cfg.max_selected_non_prior_edges_per_subject
        ),
        "limit_history_selected_non_prior_edges": int(cfg.max_total_non_prior_edges),
        "limit_max_scan_motion_history_weighted_risk_per_subject": float(
            cfg.max_scan_weighted_risk_per_subject
        ),
        "limit_history_scan_motion_history_weighted_risk": float(
            cfg.max_total_scan_weighted_risk
        ),
    }


def format_promotion_markdown(decision: Mapping[str, Any]) -> str:
    """Format a compact scan-pruning promotion-gate Markdown note."""

    exposure = dict(decision.get("exposure", {}))
    lines = [
        "# FullMHT Identity-History Scan-Pruning Promotion Gate",
        "",
        f"Status: `{decision.get('status', '')}`",
        f"Benchmark result: `{decision.get('benchmark_result', '')}`",
        f"Exposure result: `{decision.get('exposure_result', '')}`",
        f"Recommendation: {decision.get('recommendation', '')}",
        "",
        "| exposure metric | value | limit |",
        "| --- | ---: | ---: |",
    ]
    for metric, limit_key in (
        (
            "max_selected_non_prior_edges_per_subject",
            "limit_max_selected_non_prior_edges_per_subject",
        ),
        ("history_selected_non_prior_edges", "limit_history_selected_non_prior_edges"),
        (
            "max_scan_motion_history_weighted_risk_per_subject",
            "limit_max_scan_motion_history_weighted_risk_per_subject",
        ),
        (
            "history_scan_motion_history_weighted_risk",
            "limit_history_scan_motion_history_weighted_risk",
        ),
    ):
        lines.append(
            "| {metric} | {value} | {limit} |".format(
                metric=metric,
                value=exposure.get(metric, ""),
                limit=exposure.get(limit_key, ""),
            )
        )
    failed = ", ".join(str(item) for item in exposure.get("failed_limits", ()))
    lines.extend(["", f"Failed exposure limits: {failed or 'none'}"])
    return "\n".join(lines)


def write_promotion(
    decision: Mapping[str, Any], output: Path, *, output_format: str
) -> None:
    """Write the promotion gate as Markdown or JSON."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output.write_text(json.dumps(dict(decision), indent=2) + "\n", encoding="utf-8")
        return
    output.write_text(format_promotion_markdown(decision) + "\n", encoding="utf-8")


def _all_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    for row in rows:
        if str(row.get("subject", "")) == "ALL":
            return row
    return None


def _int_metric(row: Mapping[str, Any], key: str) -> int:
    try:
        return int(float(row.get(key, 0)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Exposure metric {key!r} is not numeric: {row.get(key)!r}") from exc


def _float_metric(row: Mapping[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Exposure metric {key!r} is not numeric: {row.get(key)!r}") from exc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m "
            "bayescatrack.experiments.full_mht_identity_history_scan_pruning_promotion_gate"
        ),
        description="Combine identity-history scan-pruning benchmark and exposure gates.",
    )
    parser.add_argument("comparison_csv", type=Path)
    parser.add_argument("exposure_csv", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--max-non-prior-per-subject", type=int, default=3)
    parser.add_argument("--max-total-non-prior", type=int, default=10)
    parser.add_argument("--max-scan-weighted-risk-per-subject", type=float, default=10.0)
    parser.add_argument("--max-total-scan-weighted-risk", type=float, default=25.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = ScanPruningPromotionConfig(
        max_selected_non_prior_edges_per_subject=max(
            0,
            int(args.max_non_prior_per_subject),
        ),
        max_total_non_prior_edges=max(0, int(args.max_total_non_prior)),
        max_scan_weighted_risk_per_subject=max(
            0.0,
            float(args.max_scan_weighted_risk_per_subject),
        ),
        max_total_scan_weighted_risk=max(
            0.0,
            float(args.max_total_scan_weighted_risk),
        ),
    )
    decision = evaluate_scan_pruning_promotion(
        load_comparison_rows(args.comparison_csv),
        load_exposure_rows(args.exposure_csv),
        config=cfg,
    )
    if args.output is not None:
        write_promotion(decision, args.output, output_format=str(args.format))
    elif args.format == "json":
        print(json.dumps(decision, indent=2))
    else:
        print(format_promotion_markdown(decision))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
